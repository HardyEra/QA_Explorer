"""Vision Analyst: the task agent's on-demand screenshot subagent.

It runs as its own LangGraph node, invoked only when the DOM cannot be
trusted (thin extraction, a failed action) or the planner explicitly asks to
look.  It captures the viewport, has Gemini describe what is actually visible,
and hands both the structured description and the JPEG to the planner.

Every invocation is one "Vision Analyst" span in Langfuse, with the Gemini
call nested inside it as a generation — so the trace shows exactly when the
agent needed its eyes and what they cost.
"""

from __future__ import annotations

from typing import Any

from vision_observer import VisionObservationError, VisionObserver


class VisionAnalyst:
    """Look at the screen only when looking earns its cost."""

    def __init__(self, browser, observability):
        self.browser = browser
        self.observability = observability
        self.observer = VisionObserver(observability=observability)
        # A quota error will not recover mid-run; stop paying for retries but
        # keep sending raw screenshots, which cost nothing extra.
        self.model_disabled = False

    def analyze(self, reason: str) -> dict[str, Any]:
        """Return {attempted, jpeg, summary, page_type, buttons, warnings}."""
        result: dict[str, Any] = {"attempted": True, "jpeg": None, "summary": ""}
        with self.observability.span("Vision Analyst", input={"reason": reason}) as span:
            result["jpeg"] = self._viewport_jpeg()
            if result["jpeg"] is None:
                span.update(output={"screenshot": False, "summary": ""},
                            status_message="viewport screenshot failed")
                return result
            if not self.model_disabled and self.observer.api_key:
                self._describe(result)
            span.update(output={
                "screenshot": True,
                "page_type": result.get("page_type", ""),
                "summary": result["summary"],
                "model_disabled": self.model_disabled,
            })
        return result

    def _describe(self, result: dict[str, Any]) -> None:
        try:
            observation = self.observer.observe(self.browser.page)
        except VisionObservationError as exc:
            reason = str(exc)
            if "429" in reason or "RESOURCE_EXHAUSTED" in reason.upper():
                self.model_disabled = True
            self.observability.event(
                name="vision_model_unavailable",
                metadata={"reason": reason[:300], "disabled_for_run": self.model_disabled},
                level="WARNING",
            )
            return
        except Exception as exc:
            self.observability.record_exception(
                exc, context={"active_action": "Vision Analyst"}
            )
            return
        result.update(
            page_type=observation.page_type,
            summary=observation.summary,
            buttons=observation.buttons,
            warnings=observation.warnings,
        )

    def _viewport_jpeg(self) -> bytes | None:
        try:
            png = self.browser.page.screenshot(type="png")
            jpeg, _original, _sent = VisionObserver._resize_screenshot(png)
            return jpeg
        except Exception as exc:
            self.observability.record_exception(
                exc, context={"active_action": "Vision Analyst screenshot"}
            )
            return None
