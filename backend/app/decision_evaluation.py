"""Deterministic, privacy-safe evaluation of QA agent decisions.

The evaluator measures whether the agents produced a grounded, sufficiently
covered test plan.  It intentionally does *not* treat a failing product test as
an agent failure: a failed test can be a valuable product finding.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_PATH = _BACKEND_ROOT / "generated" / "memory" / "decision_evaluations.jsonl"
GUIDANCE_PATH = _BACKEND_ROOT / "generated" / "memory" / "decision_guidance.md"
MAX_GUIDANCE_CHARS = 2_000


def evaluate_run(final_state: dict[str, Any]) -> dict[str, Any]:
    """Return stable scores and concrete lessons from one completed pipeline run.

    Scores are normalized to 0..1 so they can be charted directly in Langfuse.
    """
    requirements = [item for item in final_state.get("requirements", []) if item.get("id")]
    plan = final_state.get("test_plan", [])
    results = final_state.get("results", [])
    problems = final_state.get("verification_problems", [])

    requirement_ids = {item["id"] for item in requirements}
    planned_ids = {
        case.get("requirement_id") for case in plan
        if case.get("requirement_id") in requirement_ids
    }
    coverage = 1.0 if not requirement_ids else len(planned_ids) / len(requirement_ids)

    unverified_cases = sum(1 for case in plan if case.get("unverified"))
    grounding = 1.0 if not plan else max(0.0, 1.0 - (unverified_cases / len(plan)))

    executed = [result for result in results if result.get("status") in {"passed", "failed", "error"}]
    runnable = 1.0 if not plan else len(executed) / len(plan)
    passed = sum(1 for result in executed if result.get("status") == "passed")
    execution_success = 1.0 if not executed else passed / len(executed)

    # Decision quality is about decisions agents control: coverage, grounding,
    # and whether the plan actually ran. Product failures stay separate.
    decision_quality = round((coverage * 0.45) + (grounding * 0.40) + (runnable * 0.15), 3)
    outcome = "strong" if decision_quality >= 0.85 else "needs_review" if decision_quality >= 0.60 else "weak"

    lessons: list[str] = []
    missing = [item for item in requirements if item["id"] not in planned_ids]
    if missing:
        labels = ", ".join(item.get("title") or item["id"] for item in missing[:5])
        lessons.append(f"Cover currently untested requirements before adding extra scenarios: {labels}.")
    if problems or unverified_cases:
        lessons.append("Use only controls and fields grounded in the observed App Map; avoid invented targets.")
    if plan and not executed:
        lessons.append("Produce executable, verified test cases so planned coverage reaches the executor.")

    return {
        "run_id": final_state.get("run_id", "run"),
        "start_url": final_state.get("start_url", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scores": {
            "requirements_coverage": round(coverage, 3),
            "plan_grounding": round(grounding, 3),
            "plan_runnability": round(runnable, 3),
            "execution_success_rate": round(execution_success, 3),
            "agent_decision_quality": decision_quality,
        },
        "outcome": outcome,
        "comment": "; ".join(lessons) or "Plan was covered, grounded, and executable.",
        "metadata": {
            "requirements": len(requirements), "planned_cases": len(plan),
            "executed_cases": len(executed), "unverified_cases": unverified_cases,
            "verification_problems": len(problems),
        },
        "lessons": lessons,
    }


def persist_evaluation(evaluation: dict[str, Any]) -> None:
    """Keep a small local learning record; telemetry failures never block QA."""
    try:
        EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVALUATION_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
        _refresh_guidance()
    except OSError:
        logger.warning("Could not persist agent decision evaluation", exc_info=True)


def _refresh_guidance() -> None:
    """Turn recent non-product decision misses into compact future-run context."""
    try:
        recent = EVALUATION_PATH.read_text(encoding="utf-8").splitlines()[-20:]
        lessons: list[str] = []
        for line in reversed(recent):
            record = json.loads(line)
            for lesson in record.get("lessons", []):
                if lesson not in lessons:
                    lessons.append(lesson)
        text = "\n".join(f"- {lesson}" for lesson in lessons[:6])
        GUIDANCE_PATH.write_text(text + ("\n" if text else ""), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not refresh decision-learning guidance", exc_info=True)


def load_decision_guidance() -> str:
    try:
        return GUIDANCE_PATH.read_text(encoding="utf-8").strip()[:MAX_GUIDANCE_CHARS]
    except OSError:
        return ""
