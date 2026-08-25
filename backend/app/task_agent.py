"""LangGraph browser-task agent: complete one human goal end to end.

The graph loops ``observe → (vision analyst when needed) → plan → act`` until
the planner reports done/fail, the human says stop, or the step budget runs
out:

    START ─▶ observe ─▶ [vision] ─▶ plan ─▶ act ─┐
               ▲            ▲         │          │
               │            └── look ─┘          │
               ├─────────────────────────────────┘
               └─▶ finalize ─▶ END   (stop / limit / done / fail / error)

Observability is Langfuse-only: the runner opens the run trace, each node
works in its own span, the planner and Vision Analyst log generations, every
browser action is a ``tool:*`` span, and noteworthy moments (human notes,
payment gate, app walls) are Langfuse events.  There are no code logs here.

Human-in-the-loop is first-class: OTPs, account choices, and every
payment-looking click go through the guidance channel, never guessed.
"""

from __future__ import annotations

import operator
import os
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from task_planner import GeminiTaskPlanner, TaskDecision
from task_tools import BrowserTools
from vision_analyst import VisionAnalyst


RUNTIME_DIR = Path(__file__).resolve().parent.parent / "runtime"
UI_SCREENSHOT_PATH = RUNTIME_DIR / "latest.png"

MAX_PROMPT_ACTIONS = 90
MAX_HISTORY_LINES = 14
# Fewer extracted actions than this on a rendered page usually means the
# content is canvas/image-drawn (seat maps, map pickers) — look before acting.
MIN_TRUSTED_DOM_ACTIONS = 4

SYSTEM_RULES = """RULES
- Return exactly ONE decision for the single next browser action.
- Prefer clicking listed elements via action_type "click" with their action_id.
- "click_text" (value = the exact visible text) clicks any element by its text, searching the page AND every iframe. Use it for anything under IFRAME ELEMENTS, and as the fallback whenever the element you need is not in the action list. Login/signup/payment widgets usually live inside iframes — their controls only appear under IFRAME ELEMENTS and are only reachable with click_text or fill.
- "fill" types into a field; target = the field's visible label, placeholder, name, or id; value = the text. Fields are searched in every iframe too; inside login iframes even an unnamed lone input will be found.
- LOGIN/OTP PLAYBOOK: open the login control → prefer email/OTP flows → fill the email → click the send-OTP control → "ask_human" for the code (NEVER invent or guess one) → "enter_otp" (value = the code the human gave; it types into split code boxes correctly). If the human says no code arrived, click the resend control, wait, and ask again. If the site says the account/email is not registered, ask the human what to do.
- After typing into a search or location box, the site shows a suggestions dropdown — click the right suggestion (click_text with its visible text); do not just press Enter.
- If the listed elements look incomplete or the page is visual (seat map, map picker, canvas, image menu, unexpected layout), use "look" to request a screenshot. When a screenshot is attached you may use "click_at" with x and y in 0-1000 coordinates of the viewport (0,0 = top-left).
- Use "ask_human" (value = your question) for OTPs / verification codes, login choices, ambiguous options, or confirmations. The human's earlier answers appear under HUMAN NOTES — use them.
- NEVER click payment / place-order / buy buttons and NEVER enter card, UPI, or bank details without first getting explicit permission via ask_human. Prefer stopping at the final order-review step and finishing with "done".
- If a WARNING says the site only supports this feature in its mobile app (or another platform limit blocks the goal), do NOT keep retrying: finish with "fail" and state exactly how far you got and why the rest is impossible on this website.
- "press_key": value is a key like Enter or Escape. "scroll": value is "down" or "up". "navigate": value is a full URL. "wait" pauses ~2 seconds for slow content.
- When the goal is fully achieved use "done" with a one-sentence summary in value. If the goal is truly impossible, use "fail" with the reason in value.
- Unused fields: action_id -1, x -1, y -1, empty strings for target/value.
- Do not repeat an action that already failed; try a different approach: click_text instead of click, "look", or scroll."""

# Copy that betrays a platform wall: the goal cannot be finished on this
# website no matter what the agent clicks.  Surfaced to the planner so it
# reports the truth instead of looping.
APP_WALL_PATTERNS = (
    "only supported on the mobile app",
    "ordering is only supported on the mobile",
    "download the app to order",
    "only available on the app",
    "use the app to continue",
)

STOP_WORDS = frozenset({"stop", "abort", "cancel"})


class TaskState(TypedDict, total=False):
    """One task run. History and human notes accumulate across the loop."""

    step: int
    stop_requested: bool
    observation: Any
    context: dict[str, Any]
    vision: dict[str, Any]
    decision: TaskDecision | None
    planner_failed: bool
    history: Annotated[list[dict[str, Any]], operator.add]
    human_notes: Annotated[list[str], operator.add]
    outcome: dict[str, Any]


class TaskAgent:
    """Drive one BrowserController through a natural-language task."""

    def __init__(self, browser, planner: GeminiTaskPlanner, goal: str,
                 max_steps: int = 40, guidance=None,
                 on_event: Callable[[dict[str, Any]], None] | None = None,
                 ask_timeout_s: int = 180, observability=None):
        from observability.tracing import NoopObservability

        self.browser = browser
        self.planner = planner
        self.goal = goal.strip()
        self.max_steps = max_steps
        self.guidance = guidance
        self.on_event = on_event or (lambda event: None)
        self.observability = observability or NoopObservability()
        self.tools = BrowserTools(
            browser, self.observability, guidance=guidance,
            ask_timeout_s=ask_timeout_s, on_status=self._emit,
        )
        self.vision_analyst = VisionAnalyst(browser, self.observability)
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(TaskState)
        workflow.add_node("observe", self._observe)
        workflow.add_node("vision", self._vision)
        workflow.add_node("plan", self._plan)
        workflow.add_node("act", self._act)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "observe")
        workflow.add_conditional_edges(
            "observe", self._route_after_observe, ["vision", "plan", "finalize"]
        )
        workflow.add_edge("vision", "plan")
        workflow.add_conditional_edges(
            "plan", self._route_after_plan, ["vision", "act", "finalize"]
        )
        workflow.add_edge("act", "observe")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    # ------------------------------------------------------------------ run

    def run(self) -> dict[str, Any]:
        state = self.graph.invoke(
            {"step": 0, "history": [], "human_notes": []},
            # LangGraph counts node traversals; one step is ≤4 of them.
            {"recursion_limit": self.max_steps * 6 + 25},
        )
        return state.get("outcome") or {
            "status": "incomplete", "summary": "The task loop ended unexpectedly.",
            "steps": state.get("step", 0), "history": state.get("history", []),
        }

    # ---------------------------------------------------------------- observe

    def _observe(self, state: TaskState) -> dict[str, Any]:
        step = state.get("step", 0) + 1
        stop_requested, notes = self._drain_guidance()
        with self.observability.span("Observe Page", input={"step": step}) as span:
            observation = None
            try:
                observation = self.browser.observe()
            except Exception as exc:
                self.observability.record_exception(
                    exc, context={"active_action": "Observe Page"}
                )
            context = self._page_context()
            span.update(output={
                "url": self._current_url(),
                "title": getattr(observation, "page_title", None),
                "dom_actions": len(getattr(observation, "available_actions", None) or []),
                "iframes": len(context["frames"]),
                "otp_detected": context["otp_detected"],
                "app_wall": context["app_wall"],
            })
        return {
            "step": step,
            "stop_requested": stop_requested or state.get("stop_requested", False),
            "observation": observation,
            "context": context,
            "vision": {},
            "decision": None,
            "planner_failed": False,
            "human_notes": notes,
        }

    def _route_after_observe(self, state: TaskState) -> str:
        if state.get("stop_requested") or state.get("step", 0) > self.max_steps:
            return "finalize"
        if self._needs_vision(state):
            return "vision"
        return "plan"

    def _drain_guidance(self) -> tuple[bool, list[str]]:
        if self.guidance is None:
            return False, []
        stop, notes = False, []
        for message in self.guidance.drain():
            if message.strip().casefold() in STOP_WORDS:
                stop = True
            else:
                notes.append(f"Human said: {message}")
                self.observability.event(name="human_note", input={"message": message})
        if stop:
            self.observability.event(name="human_stop", level="WARNING")
        return stop, notes

    def _page_context(self) -> dict[str, Any]:
        """Frame contents, OTP-widget presence, and platform-wall warnings.

        This is the observation layer the QA extractor lacks: login and
        payment UI live in child iframes, and some goals are impossible on
        the website at all (app-only ordering).
        """
        context: dict[str, Any] = {"frames": [], "otp_detected": False, "app_wall": ""}
        try:
            context["frames"] = self.browser.frame_elements()
        except Exception:
            pass
        try:
            context["otp_detected"] = self.browser.find_otp_boxes() is not None
        except Exception:
            pass
        try:
            body = self.browser.page.locator("body").inner_text(timeout=1_000)
            lowered = " ".join(body.split()).casefold()
            for pattern in APP_WALL_PATTERNS:
                if pattern in lowered:
                    context["app_wall"] = pattern
                    self.observability.event(
                        name="app_wall_detected",
                        metadata={"pattern": pattern, "url": self._current_url()},
                        level="WARNING",
                    )
                    break
        except Exception:
            pass
        return context

    def _needs_vision(self, state: TaskState) -> bool:
        history = state.get("history", [])
        if history and not history[-1]["success"]:
            return True
        observation = state.get("observation")
        if observation is None:
            return True
        actions = list(observation.available_actions or [])
        frame_clickables = sum(
            len(frame.get("clickables", []))
            for frame in state.get("context", {}).get("frames", [])
        )
        return len(actions) + frame_clickables < MIN_TRUSTED_DOM_ACTIONS

    # ----------------------------------------------------------------- vision

    def _vision(self, state: TaskState) -> dict[str, Any]:
        decision = state.get("decision")
        if decision is not None and decision.action_type == "look":
            reason = "planner asked to look"
        elif state.get("observation") is None:
            reason = "page observation failed"
        elif state.get("history") and not state["history"][-1]["success"]:
            reason = "last action failed"
        else:
            reason = "DOM extraction too thin to trust"
        return {"vision": self.vision_analyst.analyze(reason)}

    # ------------------------------------------------------------------ plan

    def _plan(self, state: TaskState) -> dict[str, Any]:
        vision = state.get("vision") or {}
        decision = self.planner.decide(self._build_prompt(state), vision.get("jpeg"))
        return {"decision": decision, "planner_failed": decision is None}

    def _route_after_plan(self, state: TaskState) -> str:
        decision = state.get("decision")
        if decision is None or decision.action_type in ("done", "fail"):
            return "finalize"
        # Honour one "look" per step: analyze, then replan with the image.
        if decision.action_type == "look" and not state.get("vision"):
            return "vision"
        return "act"

    # ------------------------------------------------------------------- act

    def _act(self, state: TaskState) -> dict[str, Any]:
        decision = state["decision"]
        result = self.tools.run(decision)
        entry = {
            "step": state.get("step", 0),
            "description": decision.describe(),
            "reasoning": decision.reasoning,
            "success": result["success"],
            "url": self._current_url(),
        }
        self._emit(
            f"Step {entry['step']}: {decision.describe()} "
            f"→ {'ok' if result['success'] else 'FAILED'}"
        )
        return {"history": [entry], "human_notes": result["notes"]}

    # -------------------------------------------------------------- finalize

    def _finalize(self, state: TaskState) -> dict[str, Any]:
        decision = state.get("decision")
        if state.get("stop_requested"):
            status, summary = "stopped", "Stopped by the human."
        elif state.get("planner_failed"):
            status = "error"
            summary = "The planning model is unavailable or kept returning invalid decisions."
        elif decision is not None and decision.action_type == "done":
            status, summary = "done", decision.value or "Task completed."
        elif decision is not None and decision.action_type == "fail":
            status, summary = "failed", decision.value or "Task reported as impossible."
        else:
            status = "incomplete"
            summary = f"Reached the {self.max_steps}-step limit before finishing the task."
        outcome = {
            "status": status,
            "summary": summary,
            "steps": min(state.get("step", 0), self.max_steps),
            "history": state.get("history", []),
        }
        self.observability.event(
            name="task_outcome",
            output={"status": status, "summary": summary, "steps": outcome["steps"]},
            level="DEFAULT" if status == "done" else "WARNING",
        )
        self._emit(f"Task {status}: {summary}")
        return {"outcome": outcome}

    # ----------------------------------------------------------------- prompt

    def _build_prompt(self, state: TaskState) -> str:
        observation = state.get("observation")
        context = state.get("context") or {}
        vision = state.get("vision") or {}
        parts = [
            "You are an autonomous browser agent operating a real Chrome browser to complete one task for a human.",
            f"\nTASK: {self.goal}",
            f"\nCURRENT PAGE\nURL: {self._current_url()}",
        ]
        if context.get("app_wall"):
            parts.append(
                f'\nWARNING: the page says "{context["app_wall"]}" — this part of the goal '
                "cannot be completed on the website. Report honestly with \"fail\" instead of retrying."
            )
        if observation is not None:
            parts.append(f"Title: {observation.page_title}")
            if observation.ui_context != "NORMAL_PAGE":
                parts.append(
                    f"Active surface: {observation.ui_context}"
                    + (f' "{observation.active_container}"' if observation.active_container else "")
                    + " — only elements on this surface are listed and clickable."
                )
            parts.append("\nINTERACTIVE ELEMENTS (action_id: label [type])")
            actions = (observation.available_actions or [])[:MAX_PROMPT_ACTIONS]
            if actions:
                parts.extend(f'{action.id}: "{action.text}" [{action.type}]' for action in actions)
                if len(observation.available_actions or []) > MAX_PROMPT_ACTIONS:
                    parts.append(f"... and {len(observation.available_actions) - MAX_PROMPT_ACTIONS} more")
            else:
                parts.append("(none extracted — the page content may be visual; consider 'look')")
            input_lines = self._format_inputs(observation.inputs)
            if input_lines:
                parts.append("\nINPUT FIELDS")
                parts.extend(input_lines)
        else:
            parts.append("(page observation failed — use 'look' or 'wait')")

        for frame in context.get("frames", []):
            title = frame.get("name") or frame.get("url") or "iframe"
            parts.append(f"\nIFRAME ELEMENTS ({title}) — reach these with click_text / fill only")
            if frame.get("clickables"):
                parts.extend(f'- clickable: "{text}"' for text in frame["clickables"][:25])
            for field in frame.get("inputs", [])[:10]:
                label = field.get("placeholder") or field.get("name") or "(unnamed input)"
                parts.append(f'- input: "{label}" [{field.get("type") or "text"}]')
        if context.get("otp_detected"):
            parts.append(
                "\nOTP code boxes are visible on this page. Once the human has given the code, "
                "use enter_otp with it. If you have no code yet, ask_human for it."
            )

        if vision.get("summary"):
            parts.append("\nVISUAL ANALYSIS (what the attached screenshot shows)")
            parts.append(f"Page type: {vision.get('page_type', 'unknown')} — {vision['summary']}")
            if vision.get("buttons"):
                parts.append("Visible buttons: " + ", ".join(vision["buttons"][:15]))
            for warning in vision.get("warnings", [])[:5]:
                parts.append(f"- warning: {warning}")

        history = state.get("history", [])
        if history:
            parts.append("\nRECENT STEPS (oldest first)")
            parts.extend(
                f"{entry['step']}. {entry['description']} → {'ok' if entry['success'] else 'FAILED'}"
                for entry in history[-MAX_HISTORY_LINES:]
            )
        notes = state.get("human_notes", [])
        if notes:
            parts.append("\nHUMAN NOTES")
            parts.extend(f"- {note}" for note in notes[-10:])
        if vision.get("jpeg") is not None:
            parts.append("\nA screenshot of the current viewport is attached.")
        parts.append("\n" + SYSTEM_RULES)
        return "\n".join(parts)

    @staticmethod
    def _format_inputs(inputs) -> list[str]:
        lines = []
        for item in (inputs or [])[:25]:
            if isinstance(item, dict):
                label = (item.get("placeholder") or item.get("name")
                         or item.get("id") or item.get("label") or "")
                kind = item.get("type") or "text"
                if label:
                    lines.append(f'- "{label}" [{kind}]')
            else:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text[:80]}")
        return lines

    # ------------------------------------------------------------------ misc

    def _current_url(self) -> str:
        return self.browser.current_url() if self.browser.page else ""

    def _emit(self, status: str) -> None:
        event: dict[str, Any] = {
            "type": "task_step",
            "status": status,
            "url": self._current_url(),
        }
        try:
            UI_SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = UI_SCREENSHOT_PATH.with_name("latest.tmp.png")
            self.browser.screenshot(str(tmp_path))
            os.replace(tmp_path, UI_SCREENSHOT_PATH)
            event["screenshot_path"] = str(UI_SCREENSHOT_PATH)
        except Exception:
            pass
        self.on_event(event)
