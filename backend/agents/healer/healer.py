"""Healer agent: decide whether a failed step is a broken test or a real bug.

Called only on failure, with the failing step and a snapshot of what is
actually visible on the page.  It may propose one revised step (self-healing a
stale label) or declare the failure a genuine defect.  Without a model, every
failure is conservatively reported as a bug — never silently retried.
"""

from __future__ import annotations

import json
import logging

from agents.llm_client import JsonModelClient
from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

HEALER_PROMPT = """You are a QA automation healer. A deterministic test step failed and you must
decide whether the TEST is broken (a stale selector/label that you can fix) or
the APPLICATION is broken (a real bug to report).

Test case: {title}
Failed step: {step}
Error: {error}

What is actually visible on the page right now:
URL: {url}
Clickable controls: {controls}
Input fields: {fields}

Rules:
- Choose "retry" ONLY when a visible control or field clearly matches the
  step's intent under a different label; put that exact label in revised_step.
- The revised step must keep the same step type as the failed step.
- If nothing visible matches the intent, choose "bug".

Return exactly one JSON object:
{{"decision": "retry", "revised_step": {{"type": "click", "target": "Sign in"}}, "reason": "..."}}
or
{{"decision": "bug", "reason": "..."}}"""


class Healer:
    """Propose a single repair for a failed step, or classify it as a bug."""

    def __init__(self, observability=None, model_client: JsonModelClient | None = None):
        self.observability = observability or NoopObservability()
        self.model_client = model_client or JsonModelClient(self.observability)

    def heal(self, test_title: str, step: dict, error: str, page_snapshot: dict) -> dict:
        with self.observability.span(
            "Healer", input={"test": test_title, "step": step, "error": error[:300]}
        ) as span:
            verdict = self._model_verdict(test_title, step, error, page_snapshot)
            span.update(output=verdict)
        logger.info("Healer decision for %r: %s", test_title, verdict["decision"])
        return verdict

    def _model_verdict(self, test_title: str, step: dict, error: str, snapshot: dict) -> dict:
        payload = self.model_client.complete_json(
            "Healer.diagnose_failure",
            HEALER_PROMPT.format(
                title=test_title,
                step=json.dumps(step, ensure_ascii=False),
                error=error[:500],
                url=snapshot.get("url", ""),
                controls=", ".join(snapshot.get("controls", [])[:40]) or "None visible",
                fields=", ".join(snapshot.get("fields", [])[:20]) or "None visible",
            ),
        )
        if not payload:
            return {"decision": "bug", "reason": "healing model unavailable; treating failure as a defect"}

        decision = str(payload.get("decision") or "").casefold()
        reason = str(payload.get("reason") or "").strip() or "no reason given"
        revised = payload.get("revised_step")
        if (
            decision == "retry"
            and isinstance(revised, dict)
            and revised.get("type") == step.get("type")
            and str(revised.get("target") or "").strip()
        ):
            revised_step = {"type": step["type"], "target": str(revised["target"]).strip()}
            if step["type"] == "fill":
                revised_step["value"] = str(revised.get("value", step.get("value", "")))
            return {"decision": "retry", "revised_step": revised_step, "reason": reason}
        return {"decision": "bug", "reason": reason}
