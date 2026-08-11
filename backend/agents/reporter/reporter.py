"""Reporter: aggregate requirements, test cases, and results into artifacts.

Deterministic on purpose — a report must never hallucinate.  Produces
``report.json`` (machine), ``report.md`` (human), and ``junit.xml`` (CI).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from agents.test_designer.models import describe_expectation, describe_step
from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

REPORTS_ROOT = Path(__file__).resolve().parent.parent.parent / "generated" / "reports"


class Reporter:
    """Build the final QA report for one pipeline run."""

    def __init__(self, observability=None, output_root: Path | None = None):
        self.observability = observability or NoopObservability()
        self.output_root = output_root or REPORTS_ROOT

    def generate(self, run_id: str, requirements: list[dict], test_plan: list[dict],
                 results: list[dict], blocked: list[dict] | None = None) -> dict:
        with self.observability.span(
            "Reporter", input={"tests": len(test_plan), "results": len(results)}
        ) as span:
            report = self._build(run_id, requirements, test_plan, results, blocked or [])
            output_dir = self.output_root / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (output_dir / "report.md").write_text(self._markdown(report), encoding="utf-8")
            (output_dir / "junit.xml").write_text(self._junit(report), encoding="utf-8")
            report["artifacts"] = {
                "json": str(output_dir / "report.json"),
                "markdown": str(output_dir / "report.md"),
                "junit": str(output_dir / "junit.xml"),
            }
            span.update(output=report["summary"])
        logger.info("Report written to %s", output_dir)
        return report

    @staticmethod
    def _explain_result(result: dict) -> str:
        """Plain-language, deterministic account of why a test passed or failed.

        Built strictly from recorded facts — a report must never speculate.
        """
        status = result.get("status")
        checks = result.get("expectations") or []
        final_url = (result.get("evidence") or {}).get("final_url")

        if status == "passed":
            held = "; ".join(describe_expectation(check) for check in checks)
            return f"Every step executed successfully, and every check held: {held}."

        failed_step = result.get("failed_step")
        if status == "failed" and failed_step:
            reason = str(result.get("failure_reason") or "").strip()
            explanation = (
                f"Stopped at step {failed_step.get('index')}: could not "
                f"{describe_step(failed_step)}."
            )
            if reason:
                explanation += f" Diagnosis: {reason}"
            return explanation

        if status == "failed":
            missed = "; ".join(
                describe_expectation(check) for check in checks if not check.get("passed")
            )
            explanation = (
                "All steps executed (the flow itself worked), but verification failed — "
                f"the test expected that {missed}, and that was not observed."
            )
            if final_url:
                explanation += f" The browser ended on {final_url}."
            return explanation

        return f"The test could not run at all: {result.get('failure_reason') or 'unknown error'}."

    @staticmethod
    def _build(run_id: str, requirements: list[dict], test_plan: list[dict],
               results: list[dict], blocked: list[dict]) -> dict:
        results = [
            {**result, "explanation": Reporter._explain_result(result)} for result in results
        ]
        by_status = {"passed": 0, "failed": 0, "error": 0}
        for result in results:
            by_status[result.get("status", "error")] = by_status.get(result.get("status", "error"), 0) + 1

        tested_requirements = {
            result.get("requirement_id") for result in results if result.get("requirement_id")
        }
        blocked_ids = {entry.get("requirement_id") for entry in blocked}
        coverage = []
        for requirement in requirements:
            requirement_id = requirement.get("id", "")
            requirement_results = [
                result for result in results if result.get("requirement_id") == requirement_id
            ]
            status = "untested"
            if requirement_results:
                status = (
                    "passed"
                    if all(result["status"] == "passed" for result in requirement_results)
                    else "failed"
                )
            elif requirement_id in blocked_ids:
                status = "blocked"
            coverage.append(
                {
                    "requirement_id": requirement_id,
                    "title": requirement.get("title", ""),
                    "feature": requirement.get("feature", ""),
                    "priority": requirement.get("priority", "medium"),
                    "status": status,
                    "test_count": len(requirement_results),
                }
            )

        bugs = [
            {
                "test_id": result.get("test_id"),
                "title": result.get("title"),
                "description": result.get("description", ""),
                "requirement_id": result.get("requirement_id"),
                "reason": result.get("failure_reason"),
                "explanation": result.get("explanation", ""),
                "failed_step": result.get("failed_step"),
                "evidence": result.get("evidence", {}),
            }
            for result in results
            if result.get("status") in {"failed", "error"}
        ]

        total = len(results)
        return {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_tests": total,
                "passed": by_status.get("passed", 0),
                "failed": by_status.get("failed", 0),
                "errors": by_status.get("error", 0),
                "pass_rate": round(by_status.get("passed", 0) / total * 100, 1) if total else 0.0,
                "requirements_total": len(requirements),
                "requirements_tested": len(tested_requirements),
                "requirements_blocked": len(blocked_ids),
                "healed_tests": sum(1 for result in results if result.get("healed")),
            },
            "coverage": coverage,
            "results": results,
            "bugs": bugs,
            "blocked": blocked,
        }

    @staticmethod
    def _markdown(report: dict) -> str:
        summary = report["summary"]
        lines = [
            f"# QA Report — run `{report['run_id']}`",
            "",
            f"Generated: {report['generated_at']}",
            "",
            "## Summary",
            "",
            f"- Tests: **{summary['total_tests']}** — "
            f"{summary['passed']} passed, {summary['failed']} failed, {summary['errors']} errors "
            f"({summary['pass_rate']}% pass rate)",
            f"- Requirements covered: {summary['requirements_tested']}/{summary['requirements_total']}"
            + (f" ({summary['requirements_blocked']} blocked — see below)"
               if summary.get("requirements_blocked") else ""),
            f"- Self-healed tests: {summary['healed_tests']}",
            "",
            "## Results",
            "",
            "| Test | Requirement | Priority | Status | Duration |",
            "|---|---|---|---|---|",
        ]
        for result in report["results"]:
            lines.append(
                f"| {result.get('title', '')} | {result.get('requirement_id') or '—'} "
                f"| {result.get('priority', '')} | {result.get('status', '')} "
                f"| {result.get('duration_ms', 0)} ms |"
            )
        if report["results"]:
            lines += ["", "## Test details", ""]
            for result in report["results"]:
                icon = "✅" if result.get("status") == "passed" else "❌"
                lines.append(f"### {icon} {result.get('title', '')} (`{result.get('test_id', '')}`)")
                if result.get("description"):
                    lines.append(f"*What it tests:* {result['description']}")
                if result.get("explanation"):
                    lines.append(f"*Outcome:* {result['explanation']}")
                lines.append("")
        if report["coverage"]:
            lines += [
                "",
                "## Requirement coverage",
                "",
                "| Requirement | Feature | Priority | Status | Tests |",
                "|---|---|---|---|---|",
            ]
            for item in report["coverage"]:
                lines.append(
                    f"| {item['title']} | {item['feature']} | {item['priority']} "
                    f"| {item['status']} | {item['test_count']} |"
                )
        if report.get("blocked"):
            lines += [
                "",
                "## Blocked requirements (need a different role or account)",
                "",
            ]
            for entry in report["blocked"]:
                lines.append(
                    f"- **{entry.get('title') or entry.get('requirement_id')}** "
                    f"(`{entry.get('requirement_id')}`): {entry.get('reason')}"
                )
        if report["bugs"]:
            lines += ["", "## Failures and suspected bugs", ""]
            for bug in report["bugs"]:
                lines.append(f"### {bug['title']} (`{bug['test_id']}`)")
                lines.append(f"- Reason: {bug['reason']}")
                if bug.get("failed_step"):
                    lines.append(f"- Failed step: `{json.dumps(bug['failed_step'])}`")
                screenshot = (bug.get("evidence") or {}).get("screenshot")
                if screenshot:
                    lines.append(f"- Screenshot: {screenshot}")
                lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _junit(report: dict) -> str:
        summary = report["summary"]
        cases = []
        for result in report["results"]:
            name = escape(result.get("title", result.get("test_id", "test")))
            seconds = (result.get("duration_ms") or 0) / 1000
            if result.get("status") == "passed":
                cases.append(f'    <testcase name="{name}" time="{seconds:.2f}"/>')
            else:
                tag = "failure" if result.get("status") == "failed" else "error"
                reason = escape(str(result.get("failure_reason") or "unknown"))
                cases.append(
                    f'    <testcase name="{name}" time="{seconds:.2f}">\n'
                    f'      <{tag} message="{reason}"/>\n'
                    f"    </testcase>"
                )
        for entry in report.get("blocked", []):
            name = escape(str(entry.get("title") or entry.get("requirement_id") or "blocked"))
            reason = escape(str(entry.get("reason") or "requires a different role"))
            cases.append(
                f'    <testcase name="{name}" time="0.00">\n'
                f'      <skipped message="{reason}"/>\n'
                f"    </testcase>"
            )
        blocked_count = len(report.get("blocked", []))
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<testsuite name="qa-explorer-{escape(report["run_id"])}" '
            f'tests="{summary["total_tests"] + blocked_count}" failures="{summary["failed"]}" '
            f'errors="{summary["errors"]}" skipped="{blocked_count}">\n'
            + "\n".join(cases) + "\n</testsuite>\n"
        )
