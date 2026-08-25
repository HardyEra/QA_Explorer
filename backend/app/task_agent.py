"""LLM-driven browser task agent: complete one human goal end to end.

The loop per step: drain live human guidance → observe the page (DOM first)
→ decide the single next action with Gemini → execute it → record history.
A screenshot is attached to planning only when it earns its cost: the DOM
extraction looks too thin to be trusted (canvas/map-heavy pages), the last
action failed, or the planner explicitly asked to "look".

Human-in-the-loop is first-class: OTPs, account choices, and every
payment-looking click go through the guidance channel, never guessed.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from task_planner import GeminiTaskPlanner, TaskDecision
from vision_observer import VisionObserver


logger = logging.getLogger(__name__)

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "runtime"
UI_SCREENSHOT_PATH = RUNTIME_DIR / "latest.png"

MAX_PROMPT_ACTIONS = 90
MAX_HISTORY_LINES = 14
# Fewer extracted actions than this on a rendered page usually means the
# content is canvas/image-drawn (seat maps, map pickers) — look before acting.
MIN_TRUSTED_DOM_ACTIONS = 4

# A click whose label matches any of these is a money-committing step: the
# agent must get an explicit human "yes" first, regardless of what the
# planner decided.  Deliberately aggressive — a false positive costs one
# question; a false negative costs real money.  Word boundaries keep "cod"
# from matching "code" and "pay" from matching "paypal-free" copy.
PAYMENT_PATTERN = re.compile(
    r"\b(pay|payment|place order|buy now|purchase|add money|upi|netbanking|"
    r"cards?|wallets?|cod|paytm|gpay|phonepe|razorpay)\b"
)
CONFIRM_WORDS = frozenset({"yes", "y", "ok", "okay", "confirm", "proceed", "go ahead", "haan"})

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


class TaskAgent:
    """Drive one BrowserController through a natural-language task."""

    def __init__(self, browser, planner: GeminiTaskPlanner, goal: str,
                 max_steps: int = 40, guidance=None,
                 on_event: Callable[[dict[str, Any]], None] | None = None,
                 ask_timeout_s: int = 180):
        self.browser = browser
        self.planner = planner
        self.goal = goal.strip()
        self.max_steps = max_steps
        self.guidance = guidance
        self.on_event = on_event or (lambda event: None)
        self.ask_timeout_s = ask_timeout_s
        self.history: list[dict[str, Any]] = []
        self.human_notes: list[str] = []
        self.last_step_failed = False

    # ------------------------------------------------------------------ run

    def run(self) -> dict[str, Any]:
        outcome = {"status": "incomplete", "summary": "", "steps": 0, "history": self.history}
        for step in range(1, self.max_steps + 1):
            outcome["steps"] = step
            if self._drain_guidance_wants_stop():
                outcome.update(status="stopped", summary="Stopped by the human.")
                self._emit(f"Step {step}: stopped by the human")
                return outcome

            observation = self._observe()
            context = self._page_context()
            want_vision = self._needs_vision(observation, context)
            screenshot = self._viewport_jpeg() if want_vision else None
            decision = self.planner.decide(
                self._build_prompt(observation, context, screenshot is not None), screenshot)

            # One "look" per step: honour the request, replan with the image.
            if decision is not None and decision.action_type == "look" and screenshot is None:
                logger.info("Planner asked to look at the page; replanning with a screenshot")
                screenshot = self._viewport_jpeg()
                decision = self.planner.decide(
                    self._build_prompt(observation, context, screenshot is not None), screenshot)

            if decision is None:
                outcome.update(status="error", summary="The planning model is unavailable or kept returning invalid decisions.")
                self._emit(f"Step {step}: planner failed", level="error")
                return outcome

            logger.info("Step %s decision: %s — %s", step, decision.describe(), decision.reasoning)

            if decision.action_type == "done":
                outcome.update(status="done", summary=decision.value or "Task completed.")
                self._emit(f"Done: {outcome['summary']}")
                return outcome
            if decision.action_type == "fail":
                outcome.update(status="failed", summary=decision.value or "Task reported as impossible.")
                self._emit(f"Failed: {outcome['summary']}", level="error")
                return outcome

            success = self._execute(decision)
            self.last_step_failed = not success
            self._record(step, decision, success)
            self._emit(f"Step {step}: {decision.describe()} → {'ok' if success else 'FAILED'}")

        outcome.update(status="incomplete",
                       summary=f"Reached the {self.max_steps}-step limit before finishing the task.")
        self._emit(outcome["summary"], level="error")
        return outcome

    # ---------------------------------------------------------------- observe

    def _observe(self):
        try:
            return self.browser.observe()
        except Exception:
            logger.warning("Observation failed; continuing with an empty page view", exc_info=True)
            return None

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
            logger.debug("Frame inspection failed", exc_info=True)
        try:
            context["otp_detected"] = self.browser.find_otp_boxes() is not None
        except Exception:
            logger.debug("OTP detection failed", exc_info=True)
        try:
            body = self.browser.page.locator("body").inner_text(timeout=1_000)
            lowered = " ".join(body.split()).casefold()
            for pattern in APP_WALL_PATTERNS:
                if pattern in lowered:
                    context["app_wall"] = pattern
                    break
        except Exception:
            logger.debug("App-wall detection failed", exc_info=True)
        return context

    def _needs_vision(self, observation, context: dict[str, Any] | None = None) -> bool:
        if self.last_step_failed:
            return True
        if observation is None:
            return True
        actions = list(observation.available_actions or [])
        frame_clickables = sum(
            len(frame["clickables"]) for frame in (context or {}).get("frames", [])
        )
        if len(actions) + frame_clickables < MIN_TRUSTED_DOM_ACTIONS:
            logger.info("Only %s DOM actions and %s iframe elements extracted; attaching a screenshot",
                        len(actions), frame_clickables)
            return True
        return False

    def _viewport_jpeg(self) -> bytes | None:
        try:
            png = self.browser.page.screenshot(type="png")
            jpeg, _original, _sent = VisionObserver._resize_screenshot(png)
            return jpeg
        except Exception:
            logger.warning("Could not capture a planning screenshot", exc_info=True)
            return None

    # ----------------------------------------------------------------- prompt

    def _build_prompt(self, observation, context: dict[str, Any], screenshot_attached: bool) -> str:
        url = self.browser.current_url() if self.browser.page else "unknown"
        parts = [
            "You are an autonomous browser agent operating a real Chrome browser to complete one task for a human.",
            f"\nTASK: {self.goal}",
            f"\nCURRENT PAGE\nURL: {url}",
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

        if self.history:
            parts.append("\nRECENT STEPS (oldest first)")
            parts.extend(
                f"{entry['step']}. {entry['description']} → {'ok' if entry['success'] else 'FAILED'}"
                for entry in self.history[-MAX_HISTORY_LINES:]
            )
        if self.human_notes:
            parts.append("\nHUMAN NOTES")
            parts.extend(f"- {note}" for note in self.human_notes[-10:])
        if screenshot_attached:
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

    # ---------------------------------------------------------------- execute

    def _execute(self, decision: TaskDecision) -> bool:
        try:
            if decision.action_type == "click":
                action = self.browser.get_action(decision.action_id) or {}
                if not self._payment_gate_allows(str(action.get("text") or "")):
                    return False
                return bool(self.browser.click_action(decision.action_id))
            if decision.action_type == "click_text":
                label = decision.value or decision.target
                if not self._payment_gate_allows(label):
                    return False
                return bool(self.browser.click_text(label))
            if decision.action_type == "enter_otp":
                return bool(self.browser.fill_otp(decision.value))
            if decision.action_type == "fill":
                return bool(self.browser.fill_input(decision.target, decision.value))
            if decision.action_type == "click_at":
                return self._click_at(decision.x, decision.y)
            if decision.action_type == "press_key":
                self.browser.press_key(decision.value or "Enter")
                return True
            if decision.action_type == "scroll":
                self.browser.scroll(-600 if decision.value.strip().casefold() == "up" else 600)
                return True
            if decision.action_type == "navigate":
                self.browser.open(decision.value)
                return True
            if decision.action_type == "wait":
                self.browser.wait_for_timeout(2_000)
                return True
            if decision.action_type == "ask_human":
                return self._ask_human(decision.value)
            if decision.action_type == "look":
                # A second look in the same step adds nothing; treat as a no-op.
                return True
        except Exception:
            logger.warning("Action %s raised; recording failure", decision.action_type, exc_info=True)
            return False
        logger.warning("Unhandled action type: %s", decision.action_type)
        return False

    def _click_at(self, x: int, y: int) -> bool:
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            logger.warning("click_at coordinates out of range: (%s, %s)", x, y)
            return False
        viewport = self.browser.page.viewport_size or {"width": 1280, "height": 800}
        px = round(x / 1000 * viewport["width"])
        py = round(y / 1000 * viewport["height"])
        self.browser.click_at(px, py)
        return True

    def _ask_human(self, question: str) -> bool:
        question = question.strip() or "I need your input to continue — what should I do?"
        if self.guidance is None:
            self.human_notes.append(f"Asked: {question} — no human channel available; no answer.")
            return False
        self._emit(f"Waiting for you: {question}")
        reply = self.guidance.ask(question, timeout_s=self.ask_timeout_s)
        if reply is None:
            self.human_notes.append(f"Asked: {question} — the human did not answer in time.")
            return False
        self.human_notes.append(f"Asked: {question} — human answered: {reply}")
        return True

    def _payment_gate_allows(self, label: str) -> bool:
        """Hard gate on money-committing clicks, whatever pathway clicks them."""
        if not PAYMENT_PATTERN.search(str(label or "").casefold()):
            return True
        question = (
            f'I am about to click "{label}", which looks like a payment or '
            f'order-committing step. Reply "yes" to proceed or anything else to skip it.'
        )
        logger.info("Payment gate triggered for %r", label)
        if self.guidance is None:
            self.human_notes.append(
                f'Blocked the payment-looking click "{label}" — no human channel to confirm.'
            )
            return False
        self._emit(f"Waiting for you: {question}")
        reply = self.guidance.ask(question, timeout_s=self.ask_timeout_s)
        if reply and reply.strip().casefold() in CONFIRM_WORDS:
            self.human_notes.append(f'Human approved clicking "{label}".')
            return True
        self.human_notes.append(
            f'Human declined (or did not answer) the payment-looking click "{label}".'
        )
        return False

    # ------------------------------------------------------------------ misc

    def _drain_guidance_wants_stop(self) -> bool:
        if self.guidance is None:
            return False
        for message in self.guidance.drain():
            if message.strip().casefold() in {"stop", "abort", "cancel"}:
                return True
            self.human_notes.append(f"Human said: {message}")
        return False

    def _record(self, step: int, decision: TaskDecision, success: bool) -> None:
        self.history.append({
            "step": step,
            "description": decision.describe(),
            "reasoning": decision.reasoning,
            "success": success,
            "url": self.browser.current_url() if self.browser.page else "",
        })

    def _emit(self, status: str, level: str = "info") -> None:
        event: dict[str, Any] = {
            "type": "task_step",
            "status": status,
            "url": self.browser.current_url() if self.browser.page else "",
        }
        try:
            UI_SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = UI_SCREENSHOT_PATH.with_name("latest.tmp.png")
            self.browser.screenshot(str(tmp_path))
            os.replace(tmp_path, UI_SCREENSHOT_PATH)
            event["screenshot_path"] = str(UI_SCREENSHOT_PATH)
        except Exception:
            logger.debug("Could not capture a UI screenshot", exc_info=True)
        (logger.error if level == "error" else logger.info)("%s", status)
        self.on_event(event)
