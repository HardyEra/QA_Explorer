"""LangGraph workflow for autonomous browser exploration."""

import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from executor import Executor
from exploration_memory import ExplorationMemory
from planner import Planner


logger = logging.getLogger(__name__)


class ExplorationState(TypedDict, total=False):
    """Data passed between the exploration workflow nodes."""

    observation: Any
    plan: dict[str, Any]
    available_actions: list[Any]
    source_url: str
    execution_succeeded: bool
    in_scope: bool
    iterations: int


class Explorer:
    """Explore an application through an explicit LangGraph state machine."""

    def __init__(self, browser, config):
        self.browser = browser
        self.config = config
        self.planner = Planner()
        self.executor = Executor(browser)
        self.memory = ExplorationMemory()
        self.browser.set_new_tab_policy(config.explore_new_tabs)
        self.browser.set_navigation_policy(config.start_url, config.follow_external)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ExplorationState)
        workflow.add_node("observe", self._observe)
        workflow.add_node("check_scope", self._check_scope)
        workflow.add_node("select_actions", self._select_actions)
        workflow.add_node("plan", self._plan)
        workflow.add_node("execute", self._execute)
        workflow.add_node("record_result", self._record_result)

        workflow.add_edge(START, "observe")
        workflow.add_edge("observe", "check_scope")
        workflow.add_conditional_edges(
            "check_scope",
            self._scope_route,
            {"select_actions": "select_actions", "observe": "observe"},
        )
        workflow.add_conditional_edges(
            "select_actions",
            self._action_route,
            {"plan": "plan", "complete": END},
        )
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "record_result")
        workflow.add_edge("record_result", "observe")

        return workflow.compile()

    def _observe(self, state: ExplorationState) -> ExplorationState:
        observation = self.browser.observe()
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

    def _check_scope(self, state: ExplorationState) -> dict[str, Any]:
        observation = state["observation"]
        if not self.config.in_scope(observation.page.url):
            logger.warning("Out-of-scope URL encountered: %s", observation.page.url)
            print("\nOutside exploration scope.")
            print(observation.page.url)
            self.browser.page.go_back()
            return {"in_scope": False}

        return {"in_scope": True}

    def _scope_route(
        self, state: ExplorationState
    ) -> Literal["select_actions", "observe"]:
        return "select_actions" if state["in_scope"] else "observe"

    def _select_actions(self, state: ExplorationState) -> ExplorationState:
        observation = state["observation"]
        self.memory.mark_page(observation.page.url)

        available_actions = [
            action
            for action in observation.actions
            if not self.memory.has_executed_action(observation.page.url, action.id)
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

    def _action_route(self, state: ExplorationState) -> Literal["plan", "complete"]:
        if state["iterations"] >= self.config.max_steps:
            logger.info("Stopping after configured maximum of %s observations", self.config.max_steps)
            print(f"\n[OK] Exploration stopped after {self.config.max_steps} observations.")
            return "complete"
        if not state["available_actions"]:
            logger.info("Exploration complete; no unexplored actions remain")
            print("\n[OK] Exploration complete.")
            return "complete"
        return "plan"

    def _plan(self, state: ExplorationState) -> dict[str, Any]:
        observation = state["observation"]
        # The planner sees only actions that have not yet been executed on this page.
        observation.actions = state["available_actions"]
        plan = self.planner.plan(
            observation,
            self.memory.visited_pages,
            self.memory.executed_actions,
        )
        logger.info("Planner returned %s steps", len(plan.get("steps", [])))
        print(plan)
        return {"plan": plan, "source_url": observation.page.url}

    def _execute(self, state: ExplorationState) -> dict[str, Any]:
        success = self.executor.execute(state["plan"])
        logger.info("Plan execution %s", "succeeded" if success else "failed")
        if not success:
            print("Execution failed.")
        return {"execution_succeeded": success}

    def _record_result(self, state: ExplorationState) -> dict[str, Any]:
        """Record attempted clicks so an unclickable action is not retried forever."""
        page_url = state["source_url"]
        for step in state["plan"].get("steps", []):
            if step.get("type") == "click":
                self.memory.mark_action(page_url, step["action_id"])
                logger.info("Recorded click action %s on %s", step["action_id"], page_url)
        return {}

    def explore(self) -> ExplorationState:
        """Run the graph until no actions remain or ``config.max_steps`` is reached."""
        # LangGraph counts every node traversal, not browser observations.
        recursion_limit = max(self.config.max_steps * 6 + 10, 100)
        return self.graph.invoke({}, config={"recursion_limit": recursion_limit})
