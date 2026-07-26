"""Shared state passed between Discovery and artifact-generation graphs."""

from __future__ import annotations

from typing import Any, TypedDict


class QAState(TypedDict, total=False):
    """Cross-graph state for the end-to-end QA workflow.

    Fields for later artifact stages are present now so those nodes can be
    appended without changing the Discovery-to-Generation contract.
    """

    task: str
    start_url: str
    discovery_output: dict[str, Any]
    exploration_history: list[dict[str, Any]]
    flow: dict[str, Any] | None
    generated_script: str | None
    readme: str | None
