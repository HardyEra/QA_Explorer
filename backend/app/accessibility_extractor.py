"""Accessibility-tree based page observation.

The browser's accessibility snapshot is the authoritative description of the
page.  DOM locators are used only to turn actionable accessibility nodes into
Playwright locators the executor can click or fill.
"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from typing import Any

from models import Action


logger = logging.getLogger(__name__)

USEFUL_ROLES = {
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "menuitem", "dialog", "heading", "form",
}
ACTIONABLE_ROLES = {"button", "link", "checkbox", "radio", "combobox", "menuitem"}
INPUT_ROLES = {"textbox", "checkbox", "radio", "combobox"}


@dataclass
class AccessibilityExtraction:
    """A normalised accessibility snapshot and adapter data for the explorer."""

    tree: list[dict[str, Any]]
    actions: list[Action]
    inputs: list[dict[str, Any]]
    forms: list[dict[str, Any]]


class AccessibilityExtractor:
    """Extract meaningful, user-perceivable controls from an AX snapshot."""

    def __init__(self, page, action_registry):
        self.page = page
        self.action_registry = action_registry
        self._role_name_counts: dict[tuple[str, str], int] = {}

    def extract(self) -> AccessibilityExtraction | None:
        """Return an AX-based observation, or ``None`` when the API is unavailable."""
        try:
            # ``Page.accessibility`` was removed from Playwright Python. The
            # supported 1.61 API is Locator.aria_snapshot(), which returns a
            # YAML representation of the accessible subtree.
            snapshot = self.page.locator("body").aria_snapshot(timeout=5_000)
        except Exception as exc:
            logger.warning("Accessibility snapshot unavailable; using DOM fallback: %s", exc)
            return None

        if not snapshot:
            logger.warning("Accessibility snapshot was empty; using DOM fallback")
            return None

        tree = self._parse_snapshot(snapshot)
        if not tree:
            logger.warning("Accessibility snapshot contained no parseable nodes; using DOM fallback")
            return None

        self._role_name_counts.clear()
        inputs: list[dict[str, Any]] = []
        forms: list[dict[str, Any]] = []
        actions: list[Action] = []
        useful_tree: list[dict[str, Any]] = []
        for node in tree:
            useful_tree.extend(self._walk(node, inputs, forms, actions))
        if not useful_tree:
            logger.warning("Accessibility snapshot contained no useful nodes; using DOM fallback")
            return None
        return AccessibilityExtraction(tree=useful_tree, actions=actions, inputs=inputs, forms=forms)

    @staticmethod
    def _parse_snapshot(snapshot: str) -> list[dict[str, Any]]:
        """Parse Playwright's small ARIA-snapshot YAML dialect without a new dependency.

        ``aria_snapshot`` emits an indentation-based list of role descriptors,
        e.g. ``- button \"Sign in\" [disabled]``. Static text and other
        decorative entries are retained while parsing, then removed by
        ``_walk`` so meaningful descendants are not lost.
        """
        roots: list[dict[str, Any]] = []
        stack: list[tuple[int, dict[str, Any]]] = []
        entry_pattern = re.compile(r"^(?P<indent>\s*)-\s+(?P<descriptor>.+?)\s*$")

        for line in snapshot.splitlines():
            match = entry_pattern.match(line)
            if not match:
                continue
            node = AccessibilityExtractor._parse_descriptor(match.group("descriptor"))
            if node is None:
                continue
            indent = len(match.group("indent"))
            while stack and indent <= stack[-1][0]:
                stack.pop()
            if stack:
                stack[-1][1].setdefault("children", []).append(node)
            else:
                roots.append(node)
            stack.append((indent, node))
        return roots

    @staticmethod
    def _parse_descriptor(descriptor: str) -> dict[str, Any] | None:
        descriptor = descriptor.rstrip(":").strip()
        attributes = re.findall(r"\[([^\]]+)\]", descriptor)
        descriptor = re.sub(r"\s*\[[^\]]+\]", "", descriptor).strip()
        match = re.match(
            r'^(?P<role>[A-Za-z][\w-]*)(?:\s+"(?P<name>(?:\\.|[^"\\])*)")?$',
            descriptor,
        )
        if not match:
            return None

        node: dict[str, Any] = {
            "role": match.group("role"),
            "name": AccessibilityExtractor._unescape_name(match.group("name") or ""),
        }
        for attribute in attributes:
            for token in attribute.split():
                key, separator, value = token.partition("=")
                if key == "disabled":
                    node["disabled"] = AccessibilityExtractor._state_value(value, separator)
                elif key in {"checked", "expanded"}:
                    node[key] = AccessibilityExtractor._state_value(value, separator)
                elif key == "level" and separator:
                    try:
                        node["level"] = int(value)
                    except ValueError:
                        pass
        return node

    @staticmethod
    def _unescape_name(name: str) -> str:
        # ARIA snapshots quote names using JSON-compatible escapes.
        try:
            return json.loads(f'"{name}"')
        except json.JSONDecodeError:
            return name

    @staticmethod
    def _state_value(value: str, has_value: str) -> bool | str:
        if not has_value:
            return True
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return value

    def _walk(
        self,
        node: dict[str, Any],
        inputs: list[dict[str, Any]],
        forms: list[dict[str, Any]],
        actions: list[Action],
    ) -> list[dict[str, Any]]:
        children: list[dict[str, Any]] = []
        for child in node.get("children") or []:
            children.extend(self._walk(child, inputs, forms, actions))

        role = node.get("role")
        if role not in USEFUL_ROLES:
            return children

        cleaned = self._clean_node(node, children)
        name = cleaned["name"]
        if role in INPUT_ROLES:
            inputs.append(self._input_data(cleaned))
        if role == "form":
            forms.append(cleaned)
        if role in ACTIONABLE_ROLES and name and not cleaned.get("disabled", False):
            action = self._action_for(cleaned)
            if action:
                actions.append(action)
        return [cleaned]

    @staticmethod
    def _clean_node(node: dict[str, Any], children: list[dict[str, Any]]) -> dict[str, Any]:
        """Retain only state that conveys useful semantics to a test agent."""
        result: dict[str, Any] = {
            "role": node["role"],
            "name": node.get("name") or "",
            "disabled": bool(node.get("disabled", False)),
        }
        for key in ("checked", "expanded"):
            if key in node:
                result[key] = node[key]
        if node.get("role") == "heading" and "level" in node:
            result["level"] = node["level"]
        if children:
            result["children"] = children
        return result

    @staticmethod
    def _input_data(node: dict[str, Any]) -> dict[str, Any]:
        role = node["role"]
        input_type = {"textbox": "text", "checkbox": "checkbox", "radio": "radio", "combobox": "select"}[role]
        # Existing executor lookup accepts a name, placeholder, id, or aria label.
        return {
            "type": input_type,
            "name": node["name"],
            "placeholder": node["name"],
            "id": None,
            "role": role,
            "disabled": node.get("disabled", False),
            **({"checked": node["checked"]} if "checked" in node else {}),
            **({"expanded": node["expanded"]} if "expanded" in node else {}),
        }

    def _action_for(self, node: dict[str, Any]) -> Action | None:
        role, name = node["role"], node["name"]
        try:
            locator = self._role_locator(role, name)
            action_id = self.action_registry.register(locator=locator, text=name, action_type=role)
            return Action(id=action_id, text=name, type=role)
        except Exception as exc:
            logger.debug("Skipping accessibility action %s %r: %s", role, name, exc)
            return None

    def _role_locator(self, role: str, name: str):
        """Create a stable locator for duplicate role/name controls in tree order."""
        key = (role, name)
        position = self._role_name_counts.get(key, 0)
        self._role_name_counts[key] = position + 1
        return self.page.get_by_role(role, name=name, exact=True).nth(position)
