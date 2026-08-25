"""Deterministic browser tools for the task agent.

Every tool call is one Langfuse span named ``tool:<action>`` whose input is
the planner's decision and whose output is what actually happened — the run's
Langfuse trace is the complete log of everything the agent did.

The payment gate lives here, not in the planner: any click whose label looks
money-committing needs an explicit human "yes", whatever the model decided.
Human interaction (``ask_human`` and gate confirmations) is a tool too, so
questions and answers land in the trace alongside the browser actions.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from task_planner import TaskDecision


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


class BrowserTools:
    """Execute one planned action against the browser, fully traced."""

    def __init__(self, browser, observability, guidance=None,
                 ask_timeout_s: int = 180,
                 on_status: Callable[[str], None] | None = None):
        self.browser = browser
        self.observability = observability
        self.guidance = guidance
        self.ask_timeout_s = ask_timeout_s
        self.on_status = on_status or (lambda status: None)

    def run(self, decision: TaskDecision) -> dict[str, Any]:
        """Dispatch ``decision`` to its tool. Returns {success, notes}."""
        name = decision.action_type
        handler = getattr(self, f"_tool_{name}", None)
        notes: list[str] = []
        with self.observability.span(
            f"tool:{name}",
            input={
                "target": decision.target,
                "value": self._safe_value(decision),
                "action_id": decision.action_id,
                "x": decision.x,
                "y": decision.y,
            },
        ) as span:
            if handler is None:
                self.observability.event(
                    name="unknown_action_type",
                    metadata={"action_type": name}, level="WARNING",
                )
                success = False
            else:
                try:
                    success = bool(handler(decision, notes))
                except Exception as exc:
                    self.observability.record_exception(
                        exc, context={"active_action": f"tool:{name}"}
                    )
                    success = False
            span.update(output={"success": success, "notes": notes})
        return {"success": success, "notes": notes}

    @staticmethod
    def _safe_value(decision: TaskDecision) -> str:
        # OTP codes are secrets the human just typed; keep them out of traces.
        return "***" if decision.action_type == "enter_otp" else decision.value

    # ------------------------------------------------------------- tools

    def _tool_click(self, decision: TaskDecision, notes: list[str]) -> bool:
        action = self.browser.get_action(decision.action_id) or {}
        if not self._payment_gate_allows(str(action.get("text") or ""), notes):
            return False
        return bool(self.browser.click_action(decision.action_id))

    def _tool_click_text(self, decision: TaskDecision, notes: list[str]) -> bool:
        label = decision.value or decision.target
        if not self._payment_gate_allows(label, notes):
            return False
        return bool(self.browser.click_text(label))

    def _tool_enter_otp(self, decision: TaskDecision, notes: list[str]) -> bool:
        return bool(self.browser.fill_otp(decision.value))

    def _tool_fill(self, decision: TaskDecision, notes: list[str]) -> bool:
        return bool(self.browser.fill_input(decision.target, decision.value))

    def _tool_click_at(self, decision: TaskDecision, notes: list[str]) -> bool:
        x, y = decision.x, decision.y
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            self.observability.event(
                name="click_at_out_of_range",
                metadata={"x": x, "y": y}, level="WARNING",
            )
            return False
        viewport = self.browser.page.viewport_size or {"width": 1280, "height": 800}
        self.browser.click_at(
            round(x / 1000 * viewport["width"]),
            round(y / 1000 * viewport["height"]),
        )
        return True

    def _tool_press_key(self, decision: TaskDecision, notes: list[str]) -> bool:
        self.browser.press_key(decision.value or "Enter")
        return True

    def _tool_scroll(self, decision: TaskDecision, notes: list[str]) -> bool:
        self.browser.scroll(-600 if decision.value.strip().casefold() == "up" else 600)
        return True

    def _tool_navigate(self, decision: TaskDecision, notes: list[str]) -> bool:
        self.browser.open(decision.value)
        return True

    def _tool_wait(self, decision: TaskDecision, notes: list[str]) -> bool:
        self.browser.wait_for_timeout(2_000)
        return True

    def _tool_look(self, decision: TaskDecision, notes: list[str]) -> bool:
        # The graph routes "look" to the Vision Analyst before acting; a look
        # that still reaches the toolbelt already has a screenshot — no-op.
        return True

    def _tool_ask_human(self, decision: TaskDecision, notes: list[str]) -> bool:
        question = (decision.value or "").strip() or (
            "I need your input to continue — what should I do?"
        )
        if self.guidance is None:
            notes.append(f"Asked: {question} — no human channel available; no answer.")
            return False
        self.on_status(f"Waiting for you: {question}")
        reply = self.guidance.ask(question, timeout_s=self.ask_timeout_s)
        if reply is None:
            notes.append(f"Asked: {question} — the human did not answer in time.")
            return False
        notes.append(f"Asked: {question} — human answered: {reply}")
        return True

    # ------------------------------------------------------- payment gate

    def _payment_gate_allows(self, label: str, notes: list[str]) -> bool:
        """Hard gate on money-committing clicks, whatever pathway clicks them."""
        if not PAYMENT_PATTERN.search(str(label or "").casefold()):
            return True
        if self.guidance is None:
            notes.append(
                f'Blocked the payment-looking click "{label}" — no human channel to confirm.'
            )
            self.observability.event(
                name="payment_gate", input={"label": label},
                output={"approved": False, "reason": "no human channel"},
                level="WARNING",
            )
            return False
        question = (
            f'I am about to click "{label}", which looks like a payment or '
            f'order-committing step. Reply "yes" to proceed or anything else to skip it.'
        )
        self.on_status(f"Waiting for you: {question}")
        reply = self.guidance.ask(question, timeout_s=self.ask_timeout_s)
        approved = bool(reply and reply.strip().casefold() in CONFIRM_WORDS)
        if approved:
            notes.append(f'Human approved clicking "{label}".')
        else:
            notes.append(
                f'Human declined (or did not answer) the payment-looking click "{label}".'
            )
        self.observability.event(
            name="payment_gate", input={"label": label},
            output={"approved": approved},
            level="DEFAULT" if approved else "WARNING",
        )
        return approved
