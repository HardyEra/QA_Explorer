"""LangGraph workflow for autonomous browser exploration."""

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from executor import Executor
from exploration_memory import ExplorationMemory
from planner import Planner
from action_classifier import ActionClassifier, ActionRanker, ClassifiedAction
from observability.decorators import traced_node
from observability.tracing import TraceMetadata
from vision_observer import VisionObserver


logger = logging.getLogger(__name__)


class ExplorationState(TypedDict, total=False):
    """Data passed between the exploration workflow nodes."""

    observation: Any
    plan: dict[str, Any]
    available_actions: list[Any]
    classified_actions: list[ClassifiedAction]
    ranked_actions: list[ClassifiedAction]
    planner_candidates: list[ClassifiedAction]
    source_url: str
    execution_succeeded: bool
    exploration_history: list[dict[str, Any]]
    in_scope: bool
    iterations: int


class Explorer:
    """Explore an application through an explicit LangGraph state machine."""

    def __init__(self, browser, config, observability, on_event: Callable[[dict[str, Any]], None] | None = None):
        self.browser = browser
        self.config = config
        self.observability = observability
        # Browser-level errors need the workflow name even when they occur
        # below an Explorer node's tracing context.
        self.browser.workflow_name = config.current_goal
        self.planner = Planner(observability)
        self.vision_observer = VisionObserver(observability=observability)
        self.executor = Executor(browser, observability)
        self.memory = ExplorationMemory()
        # This is the single canonical, serializable record of Discovery's
        # observed pages and attempted business actions.  It is returned to
        # orchestration after the graph reaches END.
        self.execution_history: list[dict[str, Any]] = []
        self.action_classifier = ActionClassifier()
        self.action_ranker = ActionRanker()
        self.on_event = on_event or (lambda event: None)
        self.browser.set_new_tab_policy(config.explore_new_tabs)
        self.browser.set_navigation_policy(config.start_url, config.follow_external)
        self.graph = self._build_graph()

    def trace_metadata(self, state: ExplorationState | None = None) -> dict[str, Any]:
        """Snapshot the required discovery metadata for a child observation."""
        current_page = self.browser.current_url() if self.browser.page else None
        return TraceMetadata(
            project_name=self.config.project_name,
            session_id=self.config.session_id,
            exploration_id=self.config.exploration_id,
            current_page=current_page,
            visited_pages=sorted(self.memory.visited_pages),
            visited_actions=sorted(self.memory.executed_actions),
            current_goal=self.config.current_goal,
        ).model_dump(mode="json")

    def _build_graph(self):
        workflow = StateGraph(ExplorationState)
        workflow.add_node("observe", self._observe)
        workflow.add_node("check_scope", self._check_scope)
        workflow.add_node("prepare_candidates", self._prepare_candidates)
        workflow.add_node("plan", self._plan)
        workflow.add_node("execute", self._execute)
        workflow.add_node("record_result", self._record_result)

        workflow.add_edge(START, "observe")
        workflow.add_edge("observe", "check_scope")
        workflow.add_conditional_edges(
            "check_scope",
            self._scope_route,
            {
                "prepare_candidates": "prepare_candidates",
                "observe": "observe",
                "complete": END,
            },
        )
        workflow.add_conditional_edges(
            "prepare_candidates",
            self._action_route,
            {"plan": "plan", "complete": END},
        )
        workflow.add_conditional_edges(
            "plan",
            self._plan_route,
            {"execute": "execute", "complete": END},
        )
        workflow.add_edge("execute", "record_result")
        workflow.add_edge("record_result", "observe")

        return workflow.compile()

    @traced_node("Discovery Agent.observe")
    def _observe(self, state: ExplorationState) -> ExplorationState:
        observation = self.browser.observe()
        # Vision is supplementary perception: a Gemini outage, missing key,
        # or invalid model response must never prevent DOM-first exploration.
        if self.vision_observer.api_key:
            try:
                observation.vision_observation = self.vision_observer.observe(self.browser.page)
                logger.info(
                    "Vision observation attached: page_type=%s dialogs=%s buttons=%s",
                    observation.vision_observation.page_type,
                    len(observation.vision_observation.dialogs),
                    len(observation.vision_observation.buttons),
                )
            except Exception as exc:
                logger.warning("Vision observation unavailable; continuing without it: %s", exc)
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    # Quota will not recover mid-run; stop paying a failed
                    # request's latency on every remaining observation.
                    logger.warning("Vision disabled for the rest of this run (quota exhausted)")
                    self.vision_observer.api_key = None
        else:
            logger.info("Vision observation skipped: GEMINI_API_KEY is not configured")
        # Observed (not merely executed) controls and fields travel with the
        # navigation event so the App Map can ground the Test Designer in the
        # page's real labels and input names.
        observed_controls = [
            " ".join(str(getattr(action, "text", "")).split())
            for action in observation.actions[:20]
        ]
        observed_inputs = [
            {
                "type": str(item.get("type") or ""),
                "name": str(item.get("name") or ""),
                "placeholder": str(item.get("placeholder") or ""),
            }
            for item in observation.inputs[:12]
            if isinstance(item, dict)
        ]
        self.execution_history.append(
            {
                "action_type": "navigation",
                "success": True,
                "page_title": observation.page.title,
                "url": observation.page.url,
                "controls": [control for control in observed_controls if control],
                "inputs": observed_inputs,
                "timestamp": self._timestamp(),
            }
        )
        self.on_event({"type": "observation", "url": observation.page.url,
                       "status": "Observing page", "iteration": state.get("iterations", 0) + 1})
        logger.info(
            "Observation %s: title=%r url=%s actions=%s",
            state.get("iterations", 0) + 1,
            observation.page.title,
            observation.page.url,
            len(observation.actions),
        )
        print(observation.page.title)
        print(observation.page.url)
        print("\nAvailable Actions:")
        for action in observation.actions:
            print(action)

        return {
            "observation": observation,
            "iterations": state.get("iterations", 0) + 1,
        }

    @traced_node("Discovery Agent.check_scope")
    def _check_scope(self, state: ExplorationState) -> dict[str, Any]:
        observation = state["observation"]
        if not self.config.in_scope(observation.page.url):
            logger.warning("Out-of-scope URL encountered: %s", observation.page.url)
            print("\nOutside exploration scope.")
            print(observation.page.url)
            self.browser.go_back()
            return {"in_scope": False}

        return {"in_scope": True}

    def _scope_route(
        self, state: ExplorationState
    ) -> Literal["prepare_candidates", "observe", "complete"]:
        if state["in_scope"]:
            return "prepare_candidates"
        if state["iterations"] >= self.config.max_steps:
            logger.info("Stopping after configured maximum of %s observations", self.config.max_steps)
            return "complete"
        return "observe"

    @traced_node("Discovery Agent.select_actions")
    def _select_actions(self, state: ExplorationState) -> ExplorationState:
        observation = state["observation"]
        self.memory.mark_page(observation.page.url)

        available_actions = [
            action
            for action in observation.actions
            if not self.memory.has_executed_action(
                observation.page.url,
                self._action_memory_key(action),
            )
        ]
        logger.info(
            "Selected %s unexplored actions on %s",
            len(available_actions),
            observation.page.url,
        )

        print("\n==========================")
        print(observation.page.title)
        print(observation.page.url)
        print("==========================")
        return {"available_actions": available_actions}

    @staticmethod
    def _action_memory_key(action) -> str:
        """Keep exploration memory stable when a page reorders DOM controls.

        ``ActionRegistry`` ids intentionally reset for every observation.
        Persisting those transient ids made an action at a new DOM position look
        unexplored, and could also hide a different action that inherited the
        old id. A visible label plus action type is stable across observations.
        """
        label = " ".join(str(getattr(action, "text", "")).casefold().split())
        return f"{getattr(action, 'type', 'unknown')}::{label}"

    def _action_route(self, state: ExplorationState) -> Literal["plan", "complete"]:
        if state["iterations"] >= self.config.max_steps:
            logger.info("Stopping after configured maximum of %s observations", self.config.max_steps)
            print(f"\n[OK] Exploration stopped after {self.config.max_steps} observations.")
            return "complete"
        if not state["planner_candidates"]:
            logger.info("Exploration complete; no unexplored actions remain")
            print("\n[OK] Exploration complete.")
            return "complete"
        return "plan"

    def _plan_route(self, state: ExplorationState) -> Literal["execute", "complete"]:
        if state.get("plan", {}).get("steps"):
            return "execute"
        logger.info("Exploration complete; planner has no safe workflow action to execute")
        return "complete"

    def _prepare_candidates(self, state: ExplorationState) -> dict[str, Any]:
        """Prepare planner candidates in one graph superstep.

        Selection, classification, and ranking retain their existing order and
        behavior.  Grouping their state hand-offs prevents LangGraph's
        superstep safeguard from firing before the configured observation limit
        can reach its normal ``END`` route.
        """
        selected = self._select_actions(state)
        classified = self._classify_actions({**state, **selected})
        ranked = self._rank_actions({**state, **selected, **classified})
        candidates = self._select_candidates(
            {**state, **selected, **classified, **ranked}
        )
        return {**selected, **classified, **ranked, **candidates}

    @traced_node("Action Classification")
    def _classify_actions(self, state: ExplorationState) -> dict[str, Any]:
        classified_actions = self.action_classifier.classify_all(state["available_actions"])
        logger.info("Action Classification")
        for action in classified_actions:
            logger.info(
                "%s | category=%s | priority=%s",
                action.text,
                action.category,
                action.priority,
            )
        return {"classified_actions": classified_actions}

    @traced_node("Action Ranking")
    def _rank_actions(self, state: ExplorationState) -> dict[str, Any]:
        ranked_actions = self.action_ranker.rank(state["classified_actions"])
        logger.info("Action Ranking completed for %s actions", len(ranked_actions))
        return {"ranked_actions": ranked_actions}

    @traced_node("Planner Candidate Selection")
    def _select_candidates(self, state: ExplorationState) -> dict[str, Any]:
        ranked_actions = state["ranked_actions"]
        candidates = ranked_actions[:self.config.max_planner_actions]

        # Ranking happens before the planner activates workflow memory.  On a
        # dense dashboard, this can put a requested module such as Candidates
        # or Clients just below the normal candidate limit, even though it is
        # the only valid next step. Keep explicit workflow targets in the
        # candidate set; the planner will still enforce its normal hard gate.
        self.planner._activate_workflow(self.config)
        workflow = self.planner.workflow_memory
        if workflow and not workflow.target_reached:
            targets = [
                action for action in ranked_actions
                if self.planner._is_target_action(action)
            ]
            if targets:
                candidate_ids = {action.id for action in candidates}
                promoted = [action for action in targets if action.id not in candidate_ids]
                if promoted:
                    candidates = [*promoted, *candidates]
                    logger.info(
                        "Promoted explicit workflow target(s) beyond planner limit: %s",
                        ", ".join(action.text for action in promoted),
                    )
        logger.info("Planner Candidates")
        for position, action in enumerate(candidates, start=1):
            logger.info("%s. %s (%s)", position, action.text, action.priority)
        return {"planner_candidates": candidates}

    @traced_node("Planner")
    def _plan(self, state: ExplorationState) -> dict[str, Any]:
        # The planner receives only the highest-ranked, unexplored actions.
        # ``Observation.__post_init__`` keeps ``actions`` aligned with its
        # backwards-compatible ``available_actions`` alias.  Update both when
        # copying the observation so classified metadata is not replaced by the
        # original plain Action objects during ``dataclasses.replace``.
        observation = replace(
            state["observation"],
            actions=state["planner_candidates"],
            available_actions=state["planner_candidates"],
        )
        plan = self.planner.plan(
            observation,
            self.memory.visited_pages,
            self.memory.executed_actions,
            self.config,
        )
        logger.info("Planner returned %s steps", len(plan.get("steps", [])))
        self.on_event({"type": "status", "status": "Executing plan", "url": observation.page.url})
        print(plan)
        return {"plan": plan, "source_url": observation.page.url}

    @traced_node("Executor")
    def _execute(self, state: ExplorationState) -> dict[str, Any]:
        success = self.executor.execute(state["plan"])
        if not success and self._should_retry_login(state["observation"]):
            # Model-written fill targets sometimes miss unlabeled login fields.
            # The deterministic login plan targets the real input attributes,
            # so exploration is never stranded on a login page it can pass.
            logger.info("Plan failed on a login form; retrying with the deterministic login plan")
            fallback = self.planner._fallback_plan(
                state["observation"], self.config.username, self.config.password
            )
            if fallback.get("steps"):
                success = self.executor.execute(fallback)
        self.on_event({"type": "status", "status": "Plan execution succeeded" if success else "Plan execution failed",
                       "url": self.browser.current_url()})
        logger.info("Plan execution %s", "succeeded" if success else "failed")
        if not success:
            print("Execution failed.")
        return {"execution_succeeded": success}

    @traced_node("Discovery Agent.record_result")
    def _record_result(self, state: ExplorationState) -> dict[str, Any]:
        """Record attempted clicks so an unclickable action is not retried forever."""
        page_url = state["source_url"]
        page = state["observation"].page
        action_targets = {
            action.id: action for action in state["planner_candidates"]
        }
        for step in state["plan"].get("steps", []):
            action_type = step.get("type")
            target = step.get("target")
            if action_type == "click":
                action = action_targets.get(step.get("action_id"))
                target = action.text if action else target

            if action_type in {"click", "fill"}:
                event: dict[str, Any] = {
                    "action_type": action_type,
                    "target": target,
                    "success": state["execution_succeeded"],
                    "page_title": page.title,
                    "url": page_url,
                    "timestamp": self._timestamp(),
                }
                if action_type == "fill":
                    event["value"] = step.get("value")
                self.execution_history.append(event)

            if step.get("type") == "click":
                action_id = step.get("action_id")
                if action_id is None:
                    logger.warning("Skipping click result without an action_id: %s", step)
                    continue
                action = action_targets.get(action_id)
                if action is None:
                    logger.warning("Skipping click result for unknown action %s", action_id)
                    continue
                self.memory.mark_action(page_url, self._action_memory_key(action))
                logger.info("Recorded click action %s on %s", action.text, page_url)
        if state["execution_succeeded"]:
            self.planner.record_execution(state["plan"], state["planner_candidates"])
        return {}

    def _should_retry_login(self, observation) -> bool:
        """A failed plan is worth one deterministic retry only on a login form."""
        if not (self.config.username and self.config.password):
            return False
        return any(
            str((item or {}).get("type") or "").casefold() == "password"
            for item in getattr(observation, "inputs", [])
            if isinstance(item, dict)
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def explore(self) -> ExplorationState:
        """Run the graph until no actions remain or ``config.max_steps`` is reached."""
        # LangGraph counts every node traversal, not browser observations.
        recursion_limit = max(self.config.max_steps * 6 + 10, 100)
        metadata = TraceMetadata(**self.trace_metadata())
        try:
            with self.observability.trace(
                "Discovery Agent", metadata=metadata,
                input={"start_url": self.config.start_url, "max_steps": self.config.max_steps},
            ) as trace:
                result = self.graph.invoke({}, config={"recursion_limit": recursion_limit})
                trace.update(
                    output={"iterations": result.get("iterations", 0)},
                    metadata=self.trace_metadata(result),
                )
                result["exploration_history"] = self.execution_history
                return result
        except Exception as exc:
            self.observability.record_exception(
                exc,
                input={"start_url": self.config.start_url},
                context={
                    "current_url": self.browser.current_url() if self.browser.page else self.config.start_url,
                    "workflow_name": self.config.current_goal,
                    "screenshot_path": getattr(self.browser, "latest_screenshot_path", None),
                },
            )
            raise
        finally:
            self.observability.flush()
