"""The coordinator: a LangGraph orchestrating the multi-agent QA pipeline.

Control flow is deliberately deterministic — no LLM ever decides which phase
runs next.  The graph fans out with ``Send`` (one Doc Analyst per document,
one Test Designer per feature, one Executor per test case) and fans results
back in through additive state reducers.

    START ─▶ [analyze_doc ×N] ─▶ explore ─▶ [design_feature ×F] ─▶ critic
                                                 ▲                    │
                                                 └── gaps (≤2 rounds) ┘
                                              collect_tests ─▶ [execute_test ×T] ─▶ report ─▶ END
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.cartographer import Cartographer
from agents.coverage_critic import CoverageCritic
from agents.doc_analyst import DocAnalyst
from agents.reporter import Reporter
from agents.test_designer import TestDesigner
from agents.test_executor import ExecutorAgent
from observability.tracing import NoopObservability


logger = logging.getLogger(__name__)

MAX_DESIGN_ROUNDS = 2
MAX_TEST_CASES = 30


class PipelineState(TypedDict, total=False):
    """Orchestrator-level state: compact artifacts only, never raw blobs."""

    run_id: str
    start_url: str
    username: str
    password: str
    goal: str
    application_context: str
    max_steps: int
    skip_exploration: bool
    preserve_session: bool
    session_state_path: str
    doc_paths: list[str]

    # Fan-in channels: parallel branches append, LangGraph merges.
    requirements: Annotated[list[dict[str, Any]], operator.add]
    test_cases: Annotated[list[dict[str, Any]], operator.add]
    blocked: Annotated[list[dict[str, Any]], operator.add]
    results: Annotated[list[dict[str, Any]], operator.add]

    app_map: dict[str, Any]
    semantic_map: dict[str, Any]
    design_rounds: int
    critique: dict[str, Any]
    test_plan: list[dict[str, Any]]
    report: dict[str, Any]


class QAOrchestrator:
    """Build and run the end-to-end multi-agent QA pipeline."""

    def __init__(self, observability=None, max_steps_default: int = 12, on_event=None,
                 checkpointer=None):
        self.observability = observability or NoopObservability()
        self.on_event = on_event or (lambda event: None)
        self.doc_analyst = DocAnalyst(self.observability)
        self.cartographer = Cartographer(self.observability)
        self.designer = TestDesigner(self.observability)
        self.critic = CoverageCritic(self.observability)
        self.executor = ExecutorAgent(self.observability)
        self.reporter = Reporter(self.observability)
        self.max_steps_default = max_steps_default
        self.graph = self._build(checkpointer)

    def _build(self, checkpointer=None):
        workflow = StateGraph(PipelineState)
        workflow.add_node("analyze_doc", self._analyze_doc)
        workflow.add_node("explore", self._explore)
        workflow.add_node("cartographer", self._map_features)
        workflow.add_node("design_feature", self._design_feature)
        workflow.add_node("critic", self._critic)
        workflow.add_node("collect_tests", self._collect_tests)
        workflow.add_node("execute_test", self._execute_test)
        workflow.add_node("report", self._report)

        workflow.add_conditional_edges(START, self._ingest_route, ["analyze_doc", "explore"])
        workflow.add_edge("analyze_doc", "explore")
        workflow.add_edge("explore", "cartographer")
        workflow.add_conditional_edges("cartographer", self._design_route, ["design_feature"])
        workflow.add_edge("design_feature", "critic")
        workflow.add_conditional_edges(
            "critic", self._critic_route, ["design_feature", "collect_tests"]
        )
        workflow.add_conditional_edges(
            "collect_tests", self._execute_route, ["execute_test", "report"]
        )
        workflow.add_edge("execute_test", "report")
        workflow.add_edge("report", END)
        return workflow.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # Phase 0 — ingest (fan-out per document)
    # ------------------------------------------------------------------

    def _ingest_route(self, state: PipelineState):
        doc_paths = state.get("doc_paths") or []
        if not doc_paths:
            logger.info("No documents supplied; skipping ingestion")
            return "explore"
        logger.info("Fanning out %s Doc Analyst(s)", len(doc_paths))
        return [Send("analyze_doc", {"doc_path": path}) for path in doc_paths]

    def _analyze_doc(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"requirements": self.doc_analyst.analyze(state["doc_path"])}

    # ------------------------------------------------------------------
    # Phase 1 — explore (the existing Discovery agent)
    # ------------------------------------------------------------------

    @staticmethod
    def _prd_seeded_context(state: PipelineState, stored_map: dict | None = None) -> str:
        """Give the Explorer's planner a shopping list of features to find."""
        context = state.get("application_context", "").strip()
        features = []
        seen = set()
        for requirement in state.get("requirements", []):
            label = f"{requirement.get('feature', '')}: {requirement.get('title', '')}".strip(": ")
            if label and label.casefold() not in seen:
                seen.add(label.casefold())
                features.append(label)
        if features:
            context += (
                "\nProduct features to look for while exploring (from the product documents): "
                + "; ".join(features[:15])
            )
        unexplored = QAOrchestrator._unexplored_hints(stored_map)
        if unexplored:
            context += (
                "\nNavigation entries observed on earlier runs whose pages were NEVER "
                "visited — prioritise clicking these to go deeper into the application: "
                + ", ".join(unexplored)
            )
        return context.strip()

    @staticmethod
    def _unexplored_hints(stored_map: dict | None, limit: int = 12) -> list[str]:
        """Controls seen on mapped pages whose destination pages are unmapped.

        Heuristic frontier: a short navigation-looking label (e.g. "Candidates")
        counts as unexplored when no mapped page's URL or title mentions it.
        """
        if not stored_map:
            return []
        page_text = " ".join(
            f"{page.get('url', '')} {page.get('title', '')}"
            for page in stored_map.get("pages", [])
        ).casefold()
        hints: list[str] = []
        for page in stored_map.get("pages", []):
            for label in page.get("controls", []):
                words = label.split()
                if not words or len(words) > 3 or not label[:1].isalpha():
                    continue
                token = max(words, key=len).casefold()
                if len(token) < 4 or token in page_text:
                    continue
                if label not in hints:
                    hints.append(label)
        return hints[:limit]

    def _explore(self, state: PipelineState) -> dict[str, Any]:
        from app_map_builder import AppMapBuilder
        from app_map_store import AppMapStore

        store = AppMapStore()
        if state.get("skip_exploration"):
            # Docs-only runs still benefit from every earlier exploration:
            # the stored map grounds the Designer without opening a browser.
            stored = store.load(state["start_url"])
            if stored:
                logger.info("Exploration skipped; using the stored App Map for grounding")
                return {"app_map": stored}
            logger.info("Exploration skipped by configuration; designing from requirements only")
            return {"app_map": {"start_url": state["start_url"], "pages": [], "transitions": []}}

        from browser import BrowserController
        from config import ExplorationConfig
        from explorer import Explorer

        browser = BrowserController()

        def explorer_event(event: dict[str, Any]) -> None:
            # Mirror runner.py's live-preview contract: observation events
            # carry a screenshot path the frontend can render.
            if event.get("type") == "observation":
                try:
                    from runner import SCREENSHOT_PATH

                    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    browser.screenshot(str(SCREENSHOT_PATH))
                    event = {**event, "screenshot_path": str(SCREENSHOT_PATH)}
                except Exception:
                    logger.debug("Could not capture an exploration screenshot", exc_info=True)
            self.on_event(event)

        try:
            browser.start(observability=self.observability)
            browser.open(state["start_url"])
            config = ExplorationConfig(
                start_url=state["start_url"],
                follow_external=False,
                max_steps=state.get("max_steps") or self.max_steps_default,
                current_goal=state.get("goal")
                or "Autonomously discover and execute the application's primary business-critical workflow.",
                application_context=self._prd_seeded_context(state, store.load(state["start_url"])),
                username=state.get("username", ""),
                password=state.get("password", ""),
            )
            result = Explorer(browser, config, self.observability, on_event=explorer_event).explore()
        finally:
            try:
                browser.close()
            except Exception:
                logger.warning("Could not close the exploration browser cleanly", exc_info=True)

        fresh_map = AppMapBuilder().build(
            state["start_url"], result.get("exploration_history", [])
        )
        logger.info("App Map built this run: %s page(s)", len(fresh_map["pages"]))
        # Knowledge compounds: this run's observations join every earlier run's.
        app_map = store.merge_and_save(state["start_url"], fresh_map)
        return {"app_map": app_map}

    # ------------------------------------------------------------------
    # Phase 1.5 — cartographer: connect the map to the requirements
    # ------------------------------------------------------------------

    def _map_features(self, state: PipelineState) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        requirements = state.get("requirements", [])
        # Tester-added requirements are extras, not a requirements source: with
        # no document-derived ones, the observed application still fills in.
        has_document_requirements = any(
            requirement.get("source_doc") not in {"", "added by tester"}
            and requirement.get("source_doc") != "observed application"
            for requirement in requirements
        )
        if not has_document_requirements:
            synthesized = self.cartographer.synthesize_requirements(
                state.get("app_map", {}), state.get("application_context", "")
            )
            if synthesized:
                updates["requirements"] = synthesized
                requirements = requirements + synthesized
        updates["semantic_map"] = self.cartographer.map_features(
            requirements,
            state.get("app_map", {}),
            state.get("application_context", ""),
        )
        return updates

    # ------------------------------------------------------------------
    # Phase 2 — design (fan-out per feature) and critic loop
    # ------------------------------------------------------------------

    @staticmethod
    def _runs_fresh_login(case: dict[str, Any]) -> int:
        """0 = reuses the saved session; 1 = performs its own (fresh) login."""
        from agents.test_designer.models import USERNAME_PLACEHOLDER
        from agents.test_executor.executor_agent import ExecutorAgent

        steps = case.get("steps", [])
        boundary = ExecutorAgent._login_boundary(steps)
        uses_real_credentials = boundary is not None and any(
            USERNAME_PLACEHOLDER in str(step.get("value", "")) for step in steps
        )
        return 0 if uses_real_credentials else 1

    @staticmethod
    def _stored_asset_names() -> list[str]:
        try:
            from asset_manager import AssetManager

            return sorted(AssetManager().list_assets())
        except Exception:
            logger.debug("Could not list test assets", exc_info=True)
            return []

    @staticmethod
    def _feature_locations(state: PipelineState, requirements: list[dict]) -> list[dict]:
        """The Cartographer's findings for one feature's requirement ids."""
        wanted = {requirement.get("id") for requirement in requirements}
        return [
            item for item in state.get("semantic_map", {}).get("features", [])
            if item.get("requirement_id") in wanted
        ]

    def _design_route(self, state: PipelineState):
        groups = self._group_by_feature(state.get("requirements", []))
        if not groups:
            groups = {"smoke": []}
        detected_role = state.get("semantic_map", {}).get("detected_role", "unknown")
        logger.info("Fanning out %s Test Designer(s): %s", len(groups), ", ".join(groups))
        return [
            Send(
                "design_feature",
                {
                    "feature": feature,
                    "feature_requirements": requirements,
                    "app_map": state.get("app_map", {}),
                    "start_url": state["start_url"],
                    "has_credentials": bool(state.get("username")),
                    "application_context": state.get("application_context", ""),
                    "feature_locations": self._feature_locations(state, requirements),
                    "detected_role": detected_role,
                    "assets": self._stored_asset_names(),
                    "critique_note": "",
                },
            )
            for feature, requirements in groups.items()
        ]

    def _design_feature(self, state: dict[str, Any]) -> dict[str, Any]:
        cases, blocked = self.designer.design(
            feature=state["feature"],
            requirements=state["feature_requirements"],
            app_map=state.get("app_map", {}),
            start_url=state["start_url"],
            has_credentials=state.get("has_credentials", False),
            critique=state.get("critique_note", ""),
            application_context=state.get("application_context", ""),
            feature_locations=state.get("feature_locations"),
            detected_role=state.get("detected_role", ""),
            assets=state.get("assets"),
        )
        return {"test_cases": cases, "blocked": blocked}

    def _critic(self, state: PipelineState) -> dict[str, Any]:
        verdict = self.critic.review(
            state.get("requirements", []),
            state.get("test_cases", []),
            state.get("blocked", []),
        )
        return {"critique": verdict, "design_rounds": state.get("design_rounds", 0) + 1}

    def _critic_route(self, state: PipelineState):
        verdict = state.get("critique", {})
        rounds = state.get("design_rounds", 1)
        if verdict.get("approved") or rounds >= MAX_DESIGN_ROUNDS or not verdict.get("gaps"):
            return "collect_tests"

        gap_features: dict[str, list[str]] = {}
        for gap in verdict["gaps"]:
            gap_features.setdefault(gap.get("feature", "general"), []).append(
                f"[{gap.get('requirement_id', '')}] {gap.get('note', '')}"
            )
        groups = self._group_by_feature(state.get("requirements", []))
        logger.info("Coverage round %s: redesigning %s feature(s)", rounds + 1, len(gap_features))
        return [
            Send(
                "design_feature",
                {
                    "feature": feature,
                    "feature_requirements": groups.get(feature, []),
                    "app_map": state.get("app_map", {}),
                    "start_url": state["start_url"],
                    "has_credentials": bool(state.get("username")),
                    "application_context": state.get("application_context", ""),
                    "feature_locations": self._feature_locations(state, groups.get(feature, [])),
                    "detected_role": state.get("semantic_map", {}).get("detected_role", "unknown"),
                    "assets": self._stored_asset_names(),
                    "critique_note": "\n".join(notes),
                },
            )
            for feature, notes in gap_features.items()
        ]

    def _collect_tests(self, state: PipelineState) -> dict[str, Any]:
        """Dedupe fan-in results into the final, capped test plan."""
        plan: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for case in state.get("test_cases", []):
            case_id = case.get("id", "")
            if case_id in seen_ids:
                suffix = 2
                while f"{case_id}-{suffix}" in seen_ids:
                    suffix += 1
                case = {**case, "id": f"{case_id}-{suffix}"}
            seen_ids.add(case["id"])
            plan.append(case)
        dropped = max(0, len(plan) - MAX_TEST_CASES)
        if dropped:
            logger.warning("Test plan capped at %s case(s); dropped %s", MAX_TEST_CASES, dropped)

        # Fail fast on a total AI blackout: executing fallback-only smoke
        # tests wastes minutes and produces a report that must be discarded
        # anyway. Partial degradation still executes (real tests exist).
        from agents import llm_client

        health = llm_client.model_health()
        if health["calls"] and health["failures"] >= health["calls"]:
            logger.warning(
                "Every AI call failed (%s/%s) — skipping execution of fallback-only tests; "
                "rerun when the model quota recovers",
                health["failures"], health["calls"],
            )
            self.on_event({
                "type": "warning",
                "status": "AI completely unavailable — test execution skipped. "
                          "Rerun when the model quota recovers (exploration results were saved).",
            })
            return {"test_plan": []}

        # On single-session apps a fresh login invalidates the shared saved
        # session. Run session-reusing tests first, fresh-login tests last, so
        # auth tests cannot strand the rest of the plan on a login page.
        plan.sort(key=self._runs_fresh_login)
        updates: dict[str, Any] = {"test_plan": plan[:MAX_TEST_CASES]}
        if state.get("preserve_session") and state.get("username"):
            from pathlib import Path

            from agents.test_executor.executor_agent import create_login_session

            session_path = create_login_session(
                state["start_url"],
                state.get("username", ""),
                state.get("password", ""),
                Path(__file__).resolve().parent.parent / "runtime" / "sessions"
                / f"{state.get('run_id', 'run')}.json",
                observability=self.observability,
            )
            updates["session_state_path"] = session_path or ""
        return updates

    # ------------------------------------------------------------------
    # Phase 3 — execute (fan-out per test case) and Phase 4 — report
    # ------------------------------------------------------------------

    def _execute_route(self, state: PipelineState):
        plan = state.get("test_plan", [])
        if not plan:
            logger.warning("No executable test cases; reporting an empty run")
            return "report"
        logger.info("Fanning out %s Executor(s)", len(plan))
        return [
            Send(
                "execute_test",
                {
                    "test_case": case,
                    "start_url": state["start_url"],
                    "username": state.get("username", ""),
                    "password": state.get("password", ""),
                    "run_id": state.get("run_id", "run"),
                    "session_state_path": state.get("session_state_path", ""),
                },
            )
            for case in plan
        ]

    def _execute_test(self, state: dict[str, Any]) -> dict[str, Any]:
        result = self.executor.run(
            state["test_case"],
            start_url=state["start_url"],
            username=state.get("username", ""),
            password=state.get("password", ""),
            run_id=state.get("run_id", "run"),
            session_state_path=state.get("session_state_path") or None,
        )
        return {"results": [result]}

    def _report(self, state: PipelineState) -> dict[str, Any]:
        report = self.reporter.generate(
            run_id=state.get("run_id", "run"),
            requirements=state.get("requirements", []),
            test_plan=state.get("test_plan", []),
            results=state.get("results", []),
            blocked=state.get("blocked", []),
        )
        return {"report": report}

    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_feature(requirements: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for requirement in requirements:
            groups.setdefault(requirement.get("feature", "general"), []).append(requirement)
        return groups

    def invoke(
        self,
        state: PipelineState | None,
        max_concurrency: int = 3,
        thread_id: str | None = None,
        resume: bool = False,
    ) -> PipelineState:
        """Run the pipeline; ``max_concurrency`` caps parallel browsers/LLM calls.

        With a checkpointer and a ``thread_id``, every superstep is persisted;
        ``resume=True`` re-invokes the same thread with no input, continuing an
        interrupted run from its last completed phase.
        """
        config: dict[str, Any] = {"max_concurrency": max_concurrency, "recursion_limit": 200}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        return self.graph.invoke(None if resume else state, config=config)
