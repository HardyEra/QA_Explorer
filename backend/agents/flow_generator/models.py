"""Data contracts for normalized business workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FlowStep:
    """One successful, user-visible action in a business workflow."""

    type: str
    target: str
    value: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


@dataclass
class Workflow:
    """Portable output for the Script Generator Agent.

    This model intentionally captures business intent only.  It has no browser
    selectors, assertions, implementation details, or test-case semantics.
    """

    flow_name: str
    description: str
    pages: list[str] = field(default_factory=list)
    steps: list[FlowStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_name": self.flow_name,
            "description": self.description,
            "pages": self.pages,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class HistoryEvent:
    """Internal, normalized representation of one discovery-history item."""

    action_type: str | None
    target: str | None
    value: Any | None
    success: bool
    page_title: str | None
    url: str | None
    timestamp: Any | None
    raw: Mapping[str, Any] = field(repr=False, compare=False)
