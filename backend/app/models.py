from dataclasses import dataclass, field


@dataclass
class Action:
    id: int
    text: str
    type: str
    # Lightweight, serializable extraction context. Existing consumers can
    # continue using ``id``, ``text``, and ``type`` unchanged.
    role: str | None = None
    parent_role: str | None = None
    container_context: str | None = None
    ancestor_tags: list[str] = field(default_factory=list)
    aria_role: str | None = None
    tag_name: str | None = None


@dataclass
class Page:
    title: str
    url: str

from typing import Any

@dataclass
class Observation:
    """Page observation with legacy and accessibility-first representations.

    ``page`` and ``actions`` remain available for the existing explorer and
    planner.  ``page_title`` and ``available_actions`` provide the explicit
    observation contract used by the accessibility pipeline.
    """

    page: Page
    actions: list[Action]
    inputs: list[Any]
    forms: list[Any]
    accessibility_tree: list[dict[str, Any]] = field(default_factory=list)
    page_summary: str = ""
    page_title: str = ""
    # Optional, screenshot-derived UI context. Kept independent from the DOM
    # and accessibility representations so planner consumers can use either.
    vision_observation: Any | None = None
    available_actions: list[Action] | None = None

    def __post_init__(self) -> None:
        # Keep current consumers of ``page.title`` and ``actions`` working,
        # while accepting the explicit accessibility-observation field names.
        if not self.page_title:
            self.page_title = self.page.title
        if self.available_actions is None:
            self.available_actions = self.actions
        else:
            self.actions = self.available_actions
