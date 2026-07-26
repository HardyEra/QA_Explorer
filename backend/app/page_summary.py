"""Deterministic, LLM-free summaries of accessibility-tree observations."""

from __future__ import annotations

from typing import Any, Iterable


PRIMARY_TERMS = (
    "login", "log in", "sign in", "submit", "continue", "save", "checkout",
    "pay", "purchase", "book", "confirm", "create", "register", "start",
)
SECONDARY_TERMS = (
    "forgot", "reset", "cancel", "back", "help", "learn more", "details",
    "privacy", "terms", "contact",
)
NAVIGATION_TERMS = (
    "home", "products", "services", "dashboard", "profile", "account", "settings",
    "menu", "catalog", "about", "search", "cart",
)
CONTROL_ROLES = {"textbox", "checkbox", "radio", "combobox"}
ACTION_ROLES = {"button", "link", "checkbox", "radio", "combobox", "menuitem"}


def _descendants(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for child in node.get("children", []):
        yield child
        yield from _descendants(child)


def _first_named(nodes: Iterable[dict[str, Any]], role: str | None = None) -> str | None:
    for node in nodes:
        if (role is None or node.get("role") == role) and node.get("name"):
            return str(node["name"])
    return None


class PageSummaryGenerator:
    """Produce stable, concise page summaries from useful AX nodes only."""

    def generate(self, page_title: str, tree: list[dict[str, Any]]) -> str:
        all_nodes = list(self._flatten(tree))
        forms = [node for node in all_nodes if node.get("role") == "form"]
        form_controls = {id(node) for form in forms for node in _descendants(form)}

        primary: list[str] = []
        secondary: list[str] = []
        navigation: list[str] = []
        for node in all_nodes:
            if node.get("role") not in ACTION_ROLES or node.get("disabled") or not node.get("name"):
                continue
            name = str(node["name"])
            destination = self._action_destination(node, name, id(node) in form_controls)
            self._append_unique({"primary": primary, "secondary": secondary, "navigation": navigation}[destination], name)

        lines = [page_title or "Untitled Page"]
        if forms:
            lines.extend(["", "Forms:"])
            for index, form in enumerate(forms, start=1):
                form_name = form.get("name") or _first_named(_descendants(form), "heading") or f"Form {index}"
                lines.append(f"- {form_name}")
                for control in _descendants(form):
                    if control.get("role") in CONTROL_ROLES and control.get("name"):
                        lines.append(f"  - {control['name']}")
        self._add_section(lines, "Primary Actions", primary)
        self._add_section(lines, "Secondary Actions", secondary)
        self._add_section(lines, "Navigation", navigation)
        return "\n".join(lines)

    @staticmethod
    def _flatten(nodes: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        for node in nodes:
            yield node
            yield from _descendants(node)

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value not in items:
            items.append(value)

    @staticmethod
    def _add_section(lines: list[str], title: str, items: list[str]) -> None:
        if items:
            lines.extend(["", f"{title}:", *[f"- {item}" for item in items]])

    @staticmethod
    def _action_destination(node: dict[str, Any], name: str, is_in_form: bool) -> str:
        lower_name = name.lower()
        if node.get("role") == "link" and any(term in lower_name for term in NAVIGATION_TERMS):
            return "navigation"
        if any(term in lower_name for term in SECONDARY_TERMS):
            return "secondary"
        if is_in_form or node.get("role") == "button" or any(term in lower_name for term in PRIMARY_TERMS):
            return "primary"
        if node.get("role") in {"link", "menuitem"}:
            return "navigation"
        return "secondary"
