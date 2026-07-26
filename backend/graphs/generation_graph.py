"""LangGraph stage that turns Discovery history into a business workflow."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.flow_generator import FlowGenerator
from .state import QAState


logger = logging.getLogger(__name__)
FLOW_EXPORT_PATH = Path(__file__).resolve().parent.parent / "generated" / "flows" / "flow.json"
FLOW_EXPORT_LOG_PATH = "backend/generated/flows/flow.json"


class GenerationGraph:
    """Build the post-discovery artifact-generation graph.

    New nodes can be added after ``generate_flow`` as script, README, and
    packaging capabilities are introduced.  This graph deliberately produces
    only the normalized business flow today.
    """

    def __init__(self, flow_generator: FlowGenerator | None = None) -> None:
        self.flow_generator = flow_generator or FlowGenerator()
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(QAState)
        workflow.add_node("generate_flow", self._generate_flow)
        workflow.add_edge(START, "generate_flow")
        workflow.add_edge("generate_flow", END)
        return workflow.compile()

    def _generate_flow(self, state: QAState) -> dict[str, Any]:
        logger.info("Generation Graph started")
        logger.info("Flow generation started")
        flow = self.flow_generator.generate_json(state.get("exploration_history", []))
        logger.info("Flow generated successfully")
        FLOW_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FLOW_EXPORT_PATH.write_text(
            json.dumps(flow, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Flow saved to: %s", FLOW_EXPORT_LOG_PATH)
        logger.info("Generation Graph completed")
        # Returning only the updated key lets LangGraph preserve every other
        # field in the shared state.
        return {"flow": flow}

    def invoke(self, state: QAState) -> QAState:
        """Run generation after Discovery has completed successfully."""
        return self.graph.invoke(state)


def build_generation_graph(flow_generator: FlowGenerator | None = None):
    """Return a compiled Generation Graph for direct LangGraph composition."""
    return GenerationGraph(flow_generator).graph
