"""Coverage Critic agent: adversarial review of the designed test suite.

The critic never writes test cases itself.  It reports gaps, and the
orchestrator routes those gaps back to the Test Designer for another round.
Its deterministic core — requirements with no test — always runs; the model
adds qualitative gaps (missing negative/edge cases) on top when available.
"""

from __future__ import annotations

import json
import logging

from agents.llm_client import JsonModelClient
from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

MAX_MODEL_GAPS = 10

CRITIC_PROMPT = """You are a QA lead reviewing a test suite for coverage gaps.

Requirements:
{requirements}

Designed test cases (titles and the requirement each traces to):
{test_cases}

Identify concrete coverage gaps ONLY where the requirements imply them:
- requirements with no test case
- missing negative tests (invalid input, denied permissions) that the
  acceptance criteria call for
- missing edge cases explicitly present in the requirements

Do not invent requirements. If coverage is adequate, approve it.

Return exactly one JSON object:
{{
  "approved": true,
  "gaps": [
    {{"feature": "authentication", "requirement_id": "prd-R2", "note": "no negative login test"}}
  ]
}}"""


class CoverageCritic:
    """Judge whether the designed suite covers the requirements."""

    def __init__(self, observability=None, model_client: JsonModelClient | None = None):
        self.observability = observability or NoopObservability()
        self.model_client = model_client or JsonModelClient(self.observability)

    def review(self, requirements: list[dict], test_cases: list[dict],
               blocked: list[dict] | None = None) -> dict:
        # A requirement the supplied account's role cannot verify is reported
        # as blocked by the Designer; demanding tests for it would loop forever.
        blocked_ids = {entry.get("requirement_id") for entry in (blocked or [])}
        reviewable = [
            requirement for requirement in requirements
            if requirement.get("id") not in blocked_ids
        ]
        with self.observability.span(
            "Coverage Critic",
            input={"requirement_count": len(reviewable), "case_count": len(test_cases),
                   "blocked_count": len(blocked_ids)},
        ) as span:
            gaps = self._untested_requirements(reviewable, test_cases)
            gaps.extend(self._model_gaps(reviewable, test_cases, known=gaps))
            verdict = {"approved": not gaps, "gaps": gaps}
            span.update(output=verdict)
        logger.info(
            "Coverage Critic verdict: %s (%s gap(s))",
            "approved" if verdict["approved"] else "gaps found",
            len(gaps),
        )
        return verdict

    @staticmethod
    def _untested_requirements(requirements: list[dict], test_cases: list[dict]) -> list[dict]:
        covered = {case.get("requirement_id") for case in test_cases if case.get("requirement_id")}
        return [
            {
                "feature": requirement.get("feature", "general"),
                "requirement_id": requirement.get("id", ""),
                "note": f"no test case covers: {requirement.get('title', '')}",
            }
            for requirement in requirements
            if requirement.get("id") and requirement["id"] not in covered
        ]

    def _model_gaps(
        self, requirements: list[dict], test_cases: list[dict], known: list[dict]
    ) -> list[dict]:
        if not requirements or not test_cases:
            return []
        payload = self.model_client.complete_json(
            "Coverage Critic.review",
            CRITIC_PROMPT.format(
                requirements=json.dumps(
                    [
                        {
                            "id": requirement.get("id"),
                            "feature": requirement.get("feature"),
                            "title": requirement.get("title"),
                            "acceptance_criteria": requirement.get("acceptance_criteria", []),
                        }
                        for requirement in requirements
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                test_cases=json.dumps(
                    [
                        {
                            "id": case.get("id"),
                            "requirement_id": case.get("requirement_id"),
                            "title": case.get("title"),
                        }
                        for case in test_cases
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
            ),
        )
        if not payload or not isinstance(payload.get("gaps"), list):
            return []
        known_requirements = {gap["requirement_id"] for gap in known}
        requirement_ids = {requirement.get("id") for requirement in requirements}
        gaps = []
        for gap in payload["gaps"][:MAX_MODEL_GAPS]:
            if not isinstance(gap, dict):
                continue
            requirement_id = str(gap.get("requirement_id") or "").strip()
            # The critic reviews, it does not invent: a gap must reference a
            # real requirement and not repeat the deterministic findings.
            if requirement_id not in requirement_ids or requirement_id in known_requirements:
                continue
            gaps.append(
                {
                    "feature": str(gap.get("feature") or "general").casefold(),
                    "requirement_id": requirement_id,
                    "note": str(gap.get("note") or "").strip() or "coverage gap",
                }
            )
        return gaps
