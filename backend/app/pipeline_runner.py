"""Entrypoint for the orchestrated multi-agent QA pipeline.

Usage (from the repository root, virtualenv active):

    python backend/app/pipeline_runner.py --url https://www.saucedemo.com \
        --docs backend/docs --username standard_user --password secret_sauce

Programmatic use mirrors ``runner.run_exploration``: call ``run_pipeline`` with
a required PRD and optionally pass ``on_event`` for progress callbacks.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
for _path in (_BACKEND_ROOT, _BACKEND_ROOT / "app"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from logging_config import configure_logging  # noqa: E402
from observability.langfuse_client import create_observability  # noqa: E402
from observability.tracing import TraceMetadata  # noqa: E402
from runner import EventLogHandler  # noqa: E402

from graphs.orchestrator_graph import PipelineState, QAOrchestrator  # noqa: E402


logger = logging.getLogger(__name__)

DOC_SUFFIXES = {".txt", ".pdf", ".docx"}
CHECKPOINT_PATH = _BACKEND_ROOT / "runtime" / "checkpoints" / "pipeline.db"
RUNS_STORE = _BACKEND_ROOT / "generated" / "runs"
FEEDBACK_PATH = _BACKEND_ROOT / "generated" / "memory" / "feedback.md"
MAX_FEEDBACK_CHARS = 4_000


def _create_checkpointer():
    """Persist every pipeline superstep so interrupted runs can resume."""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
        return SqliteSaver(connection)
    except Exception:
        logger.warning("Checkpointing unavailable; runs will not be resumable", exc_info=True)
        return None


CUSTOM_REQUIREMENTS_PATH = _BACKEND_ROOT / "generated" / "memory" / "custom_requirements.json"
CUSTOM_TEST_CASES_PATH = _BACKEND_ROOT / "generated" / "memory" / "custom_test_cases.json"


def _load_json_list(path: Path) -> list[dict]:
    import json

    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read %s", path, exc_info=True)
    return []


def _save_json_list(path: Path, items: list[dict]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


CUSTOM_SOURCE = "added by tester"


def load_custom_requirements() -> list[dict]:
    return _load_json_list(CUSTOM_REQUIREMENTS_PATH)


def add_custom_requirement(feature: str, title: str, description: str = "",
                           priority: str = "medium") -> dict:
    """Persist a tester-written requirement; included in every future run."""
    requirements = load_custom_requirements()
    requirement = {
        "id": f"user-R{len(requirements) + 1}",
        "feature": (feature or "general").strip().casefold(),
        "title": title.strip(),
        "description": description.strip(),
        "acceptance_criteria": [],
        "priority": priority if priority in {"critical", "high", "medium", "low"} else "medium",
        "source_doc": CUSTOM_SOURCE,
    }
    requirements.append(requirement)
    _save_json_list(CUSTOM_REQUIREMENTS_PATH, requirements)
    return requirement


def parse_steps_text(text: str) -> list[dict]:
    """One step per line: ``click: Label`` / ``fill: Field = value`` /
    ``navigate: url`` / ``upload: Control = asset_name``."""
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        kind, _, rest = line.partition(":")
        kind = kind.strip().casefold()
        rest = rest.strip()
        if kind in {"fill", "upload"} and "=" in rest:
            target, _, value = rest.partition("=")
            steps.append({"type": kind, "target": target.strip(), "value": value.strip()})
        elif kind in {"navigate", "click", "fill", "upload"}:
            steps.append({"type": kind, "target": rest})
    return steps


def parse_expected_text(text: str) -> list[dict]:
    """One check per line: ``url: substring`` / ``text: visible text`` /
    ``element: control label``."""
    kinds = {"url": "url_contains", "text": "text_visible", "element": "element_visible"}
    expected = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        kind, _, value = line.partition(":")
        expectation_type = kinds.get(kind.strip().casefold())
        if expectation_type and value.strip():
            expected.append({"type": expectation_type, "value": value.strip()})
    return expected


def load_custom_test_cases() -> list[dict]:
    return _load_json_list(CUSTOM_TEST_CASES_PATH)


def add_custom_test_case(title: str, description: str, steps_text: str,
                         expected_text: str, requirement_id: str = "") -> dict | str:
    """Validate and persist a tester-written test case; returns an error string
    when the steps/checks don't parse into anything executable."""
    from agents.test_designer import normalise_case
    from agents.test_designer.models import describe_case

    cases = load_custom_test_cases()
    case = normalise_case(
        {
            "id": f"user-t{len(cases) + 1}",
            "requirement_id": requirement_id.strip(),
            "title": title.strip(),
            "description": description.strip(),
            "priority": "high",
            "steps": parse_steps_text(steps_text),
            "expected": parse_expected_text(expected_text),
        },
        fallback_id=f"user-t{len(cases) + 1}",
    )
    if case is None:
        return ("Could not parse the test case: it needs a title, at least one valid step "
                "(navigate/click/fill/upload), and at least one check (url/text/element).")
    if not case.get("description"):
        case["description"] = describe_case(case)
    cases.append(case)
    _save_json_list(CUSTOM_TEST_CASES_PATH, cases)
    return case


def load_feedback() -> str:
    """Tester corrections remembered across runs (see generated/memory/feedback.md)."""
    try:
        if FEEDBACK_PATH.is_file():
            return FEEDBACK_PATH.read_text(encoding="utf-8").strip()[:MAX_FEEDBACK_CHARS]
    except OSError:
        logger.warning("Could not read the feedback memory file", exc_info=True)
    return ""


def save_feedback(text: str) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_PATH.write_text(text.strip() + "\n", encoding="utf-8")


def _store_run(run_id: str, final_state: dict[str, Any]) -> None:
    """Persist the run's artifacts so they can be questioned and reused later."""
    import json

    record = {
        "run_id": run_id,
        "start_url": final_state.get("start_url", ""),
        "goal": final_state.get("goal", ""),
        "requirements": final_state.get("requirements", []),
        "semantic_map": final_state.get("semantic_map", {}),
        "exploration_history": final_state.get("exploration_history", []),
        "workflow": final_state.get("workflow", {}),
        "evidence_map": final_state.get("evidence_map", {}),
        "test_plan": final_state.get("test_plan", []),
        "blocked": final_state.get("blocked", []),
        "summary": final_state.get("report", {}).get("summary", {}),
        "report_artifacts": final_state.get("report", {}).get("artifacts", {}),
    }
    try:
        RUNS_STORE.mkdir(parents=True, exist_ok=True)
        (RUNS_STORE / f"{run_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError:
        logger.warning("Could not persist the run record for %s", run_id, exc_info=True)


def collect_documents(docs: str | None) -> list[str]:
    """Accept a directory or a comma-separated list of document files."""
    if not docs:
        return []
    paths: list[Path] = []
    for entry in docs.split(","):
        path = Path(entry.strip()).expanduser()
        if path.is_dir():
            paths.extend(sorted(child for child in path.iterdir() if child.suffix.casefold() in DOC_SUFFIXES))
        elif path.is_file():
            paths.append(path)
        else:
            logger.warning("Ignoring missing document path: %s", path)
    return [str(path) for path in paths]


def run_pipeline(
    *,
    start_url: str,
    docs: str | None = None,
    username: str = "",
    password: str = "",
    goal: str = "",
    application_context: str = "",
    max_steps: int = 12,
    skip_exploration: bool = False,
    preserve_session: bool = False,
    max_concurrency: int = 3,
    resume_run_id: str | None = None,
    hitl_wait_seconds: int = 0,
    guidance=None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run ingest → explore → map → design → critique → execute → report."""
    configure_logging()
    callback = on_event or (lambda event: None)
    log_handler = EventLogHandler(callback)
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(log_handler)
    observability = create_observability()
    run_id = resume_run_id or uuid4().hex[:12]
    doc_paths = collect_documents(docs)
    if not doc_paths:
        raise ValueError("A PRD is required. Upload a non-empty .txt, .pdf, or .docx product document.")
    if skip_exploration:
        raise ValueError("PRD-driven QA always explores the website before designing tests.")

    feedback = load_feedback()
    if feedback:
        logger.info("Applying %s character(s) of remembered tester feedback", len(feedback))
        application_context = (
            application_context.strip()
            + "\nTester feedback remembered from earlier runs (authoritative corrections):\n"
            + feedback
        )
    from decision_evaluation import load_decision_guidance

    decision_guidance = load_decision_guidance()
    if decision_guidance:
        logger.info("Applying learned agent-decision guidance")
        application_context = (
            application_context.strip()
            + "\nLessons from earlier agent decision evaluations (apply when relevant):\n"
            + decision_guidance
        )

    callback({"type": "started", "run_id": run_id, "status": "Pipeline starting",
              "docs": len(doc_paths), "url": start_url})
    logger.info("PRD-driven pipeline %s: %s document(s), start_url=%s", run_id, len(doc_paths), start_url)

    state: PipelineState = {
        "run_id": run_id,
        "start_url": start_url,
        "username": username,
        "password": password,
        "goal": goal.strip(),
        "application_context": application_context.strip(),
        "max_steps": max_steps,
        "skip_exploration": skip_exploration,
        "preserve_session": preserve_session,
        "hitl_wait_seconds": hitl_wait_seconds,
        "doc_paths": doc_paths,
        # Tester-written requirements and cases join every run alongside the
        # document/exploration-derived ones.
        "requirements": load_custom_requirements(),
        "test_cases": load_custom_test_cases(),
        "blocked": [],
        "results": [],
        "design_rounds": 0,
    }

    from agents import llm_client

    llm_client.reset_model_health()
    orchestrator = QAOrchestrator(
        observability, on_event=callback, checkpointer=_create_checkpointer(),
        guidance=guidance,
    )
    try:
        with observability.trace(
            "QA Pipeline Run",
            metadata=TraceMetadata(
                project_name="qa-explorer", session_id=run_id, exploration_id=run_id,
                current_page=start_url, current_goal=goal or "Autonomous QA pipeline",
            ),
            input={"start_url": start_url, "goal": goal, "max_steps": max_steps},
        ) as trace:
            final_state = orchestrator.invoke(
                state,
                max_concurrency=max_concurrency,
                thread_id=run_id,
                resume=bool(resume_run_id),
            )
            final_state = dict(final_state)
            from decision_evaluation import evaluate_run, persist_evaluation

            evaluation = evaluate_run(final_state)
            final_state["decision_evaluation"] = evaluation
            trace.update(output={"summary": final_state.get("report", {}).get("summary", {}),
                                 "evaluation": evaluation})
            for name, value in evaluation["scores"].items():
                observability.score_current_trace(
                    name, value, comment=evaluation["comment"], metadata=evaluation["metadata"],
                )
            observability.score_current_trace(
                "agent_decision_outcome", evaluation["outcome"], data_type="CATEGORICAL",
                comment=evaluation["comment"], metadata=evaluation["metadata"],
            )
            persist_evaluation(evaluation)
        report = final_state.get("report", {})
        health = llm_client.model_health()
        if health["failures"]:
            # Fallback-quality output must never masquerade as an AI-designed run.
            report["model_health"] = health
            warning = (
                f"AI model unavailable for {health['failures']} of {health['calls']} call(s) "
                f"(last error: {health['last_error']}). Results are fallback-quality — "
                "rerun once the model quota recovers."
            )
            logger.warning(warning)
            callback({"type": "warning", "run_id": run_id, "status": warning})
        _store_run(run_id, final_state)
        callback({"type": "completed", "run_id": run_id, "status": "Pipeline complete",
                  "summary": report.get("summary", {})})
        return dict(final_state)
    except Exception as exc:
        logger.exception("Pipeline %s failed", run_id)
        callback({"type": "error", "run_id": run_id, "status": f"Pipeline failed: {exc}"})
        raise
    finally:
        logging.getLogger().removeHandler(log_handler)
        observability.flush()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the multi-agent QA pipeline")
    parser.add_argument("--url", required=True, help="Application start URL")
    parser.add_argument("--docs", required=True,
                        help="Required PRD/product document(s): .txt, .pdf, or .docx")
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--goal", default="", help="Optional exploration goal")
    parser.add_argument("--context", default="", help="Optional application context")
    parser.add_argument("--max-steps", type=int, default=12, help="Discovery observation budget")
    parser.add_argument("--preserve-session", action="store_true",
                        help="Log in once and reuse the session across tests (skips per-test logins)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel budget for doc analysis, design, and test execution")
    parser.add_argument("--resume", default=None, metavar="RUN_ID",
                        help="Resume an interrupted run from its last completed phase")
    args = parser.parse_args()

    final_state = run_pipeline(
        start_url=args.url,
        docs=args.docs,
        username=args.username,
        password=args.password,
        goal=args.goal,
        application_context=args.context,
        max_steps=args.max_steps,
        preserve_session=args.preserve_session,
        max_concurrency=args.workers,
        resume_run_id=args.resume,
    )
    summary = final_state.get("report", {}).get("summary", {})
    artifacts = final_state.get("report", {}).get("artifacts", {})
    print("\n=== QA pipeline finished ===")
    print(f"Tests: {summary.get('total_tests', 0)} | passed {summary.get('passed', 0)} "
          f"| failed {summary.get('failed', 0)} | errors {summary.get('errors', 0)}")
    print(f"Requirements covered: {summary.get('requirements_tested', 0)}"
          f"/{summary.get('requirements_total', 0)}")
    if artifacts:
        print(f"Report: {artifacts.get('markdown')}")


if __name__ == "__main__":
    main()
