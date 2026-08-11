"""Structured requirement extracted from a product document."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_PRIORITIES = ("critical", "high", "medium", "low")


@dataclass
class Requirement:
    """One testable statement of product behaviour, traceable to its source."""

    id: str
    feature: str
    title: str
    description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    priority: str = "medium"
    source_doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "feature": self.feature,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": list(self.acceptance_criteria),
            "priority": self.priority if self.priority in VALID_PRIORITIES else "medium",
            "source_doc": self.source_doc,
        }
