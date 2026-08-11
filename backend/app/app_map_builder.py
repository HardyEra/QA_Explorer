"""Deterministically condense a Discovery run into a compact App Map.

The App Map is the only exploration artifact later agents ever see.  Raw
observations, accessibility trees, and planner scratch state stay inside the
Discovery graph and are discarded; this keeps downstream prompts small.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


MAX_ACTIONS_PER_PAGE = 30


class AppMapBuilder:
    """Fold Discovery history events into pages with their observed actions."""

    def build(self, start_url: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        pages: dict[str, dict[str, Any]] = {}
        transitions: list[dict[str, str]] = []
        previous_url = None

        for event in history:
            if not isinstance(event, dict):
                continue
            url = self._normalise_url(str(event.get("url") or ""))
            if not url:
                continue
            page = pages.setdefault(
                url,
                {"url": url, "title": str(event.get("page_title") or ""), "actions": [],
                 "fills": [], "controls": [], "fields": []},
            )
            if event.get("page_title") and not page["title"]:
                page["title"] = str(event["page_title"])

            action_type = str(event.get("action_type") or "")
            target = str(event.get("target") or "").strip()
            if action_type == "navigation":
                for control in event.get("controls") or []:
                    label = str(control).strip()
                    if label and label not in page["controls"] and len(page["controls"]) < MAX_ACTIONS_PER_PAGE:
                        page["controls"].append(label)
                for item in event.get("inputs") or []:
                    if isinstance(item, dict):
                        field = {
                            "type": str(item.get("type") or ""),
                            "name": str(item.get("name") or ""),
                            "placeholder": str(item.get("placeholder") or ""),
                        }
                        if field not in page["fields"]:
                            page["fields"].append(field)
                if previous_url and previous_url != url:
                    transition = {"from": previous_url, "to": url}
                    if transition not in transitions:
                        transitions.append(transition)
                previous_url = url
            elif action_type == "click" and target:
                entry = {"label": target, "succeeded": bool(event.get("success"))}
                if entry not in page["actions"] and len(page["actions"]) < MAX_ACTIONS_PER_PAGE:
                    page["actions"].append(entry)
            elif action_type == "fill" and target:
                entry = {"field": target}
                if entry not in page["fills"]:
                    page["fills"].append(entry)

        return {
            "start_url": start_url,
            "pages": list(pages.values()),
            "transitions": transitions,
        }

    @staticmethod
    def _normalise_url(url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        # Query strings usually encode transient state (filters, tokens) that
        # would fragment one logical page into many map entries.
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/") or url

    @staticmethod
    def compact_text(app_map: dict[str, Any], max_pages: int = 15) -> str:
        """Serialize the map for a prompt without flooding the context."""
        lines = [f"Start URL: {app_map.get('start_url', '')}"]
        for page in app_map.get("pages", [])[:max_pages]:
            lines.append(f"- Page: {page.get('title') or '(untitled)'} | {page.get('url')}")
            labels = [action["label"] for action in page.get("actions", [])]
            for control in page.get("controls", []):
                if control not in labels:
                    labels.append(control)
            if labels:
                lines.append(f"  Clickable: {', '.join(labels[:MAX_ACTIONS_PER_PAGE])}")
            fields = [fill["field"] for fill in page.get("fills", [])]
            for item in page.get("fields", []):
                descriptor = item.get("placeholder") or item.get("name")
                if descriptor and item.get("type"):
                    descriptor = f"{descriptor} (type={item['type']})"
                elif not descriptor:
                    descriptor = f"unlabeled input (type={item.get('type') or 'text'})"
                if descriptor not in fields:
                    fields.append(descriptor)
            if fields:
                lines.append(f"  Form fields: {', '.join(fields)}")
        for transition in app_map.get("transitions", [])[:20]:
            lines.append(f"  Nav: {transition['from']} -> {transition['to']}")
        return "\n".join(lines)
