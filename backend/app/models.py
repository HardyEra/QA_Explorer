from dataclasses import dataclass


@dataclass
class Action:
    id: int
    text: str
    type: str


@dataclass
class Page:
    title: str
    url: str

from typing import Any

@dataclass
class Observation:
    page: Page
    actions: list[Action]
    inputs: list[Any]
    forms: list[Any]