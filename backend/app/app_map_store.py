"""Persistent, cumulative App Map storage.

One JSON file per application domain.  Every exploration run merges its fresh
observations into the stored map instead of replacing it, so the agent's
knowledge of an application grows run over run and survives crashes, restarts,
and docs-only runs (which load the stored map for grounding even though they
never open a browser).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

APP_MAPS_ROOT = Path(__file__).resolve().parent.parent / "generated" / "app_maps"
MAX_LIST_ITEMS = 60


class AppMapStore:
    """Load, merge, and save one cumulative App Map per domain."""

    def __init__(self, root: Path | None = None):
        self.root = root or APP_MAPS_ROOT

    def _path(self, start_url: str) -> Path:
        domain = urlparse(start_url).netloc or "unknown"
        return self.root / f"{domain.replace(':', '_')}.json"

    def load(self, start_url: str) -> dict | None:
        path = self._path(start_url)
        if not path.is_file():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            logger.info(
                "Loaded stored App Map for %s: %s page(s) from %s run(s)",
                urlparse(start_url).netloc, len(stored.get("pages", [])),
                stored.get("run_count", "?"),
            )
            return stored
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load stored App Map %s: %s", path, exc)
            return None

    def merge_and_save(self, start_url: str, fresh_map: dict) -> dict:
        """Fold a run's observations into the stored map and persist it."""
        merged = self.merge(self.load(start_url) or {}, fresh_map)
        merged["start_url"] = fresh_map.get("start_url") or merged.get("start_url") or start_url
        merged["updated_at"] = datetime.now(timezone.utc).isoformat()
        merged["run_count"] = int(merged.get("run_count") or 0) + 1
        path = self._path(start_url)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            logger.info(
                "App Map for %s now covers %s page(s) across %s run(s)",
                urlparse(start_url).netloc, len(merged["pages"]), merged["run_count"],
            )
        except OSError as exc:
            logger.warning("Could not persist App Map %s: %s", path, exc)
        return merged

    @staticmethod
    def merge(old: dict, new: dict) -> dict:
        """Union two maps; page identity is the normalised URL."""

        def merge_list(existing: list, incoming: list) -> list:
            for item in incoming:
                if item not in existing and len(existing) < MAX_LIST_ITEMS:
                    existing.append(item)
            return existing

        pages: dict[str, dict] = {}
        for source in (old.get("pages", []), new.get("pages", [])):
            for page in source:
                url = page.get("url")
                if not url:
                    continue
                existing = pages.get(url)
                if existing is None:
                    # Deep-copy the lists so merging never mutates the caller's map.
                    pages[url] = {
                        "url": url,
                        "title": page.get("title", ""),
                        "actions": list(page.get("actions", [])),
                        "fills": list(page.get("fills", [])),
                        "controls": list(page.get("controls", [])),
                        "fields": list(page.get("fields", [])),
                    }
                    continue
                if page.get("title") and not existing.get("title"):
                    existing["title"] = page["title"]
                for key in ("actions", "fills", "controls", "fields"):
                    merge_list(existing.setdefault(key, []), page.get(key, []))

        transitions: list[dict] = []
        for source in (old.get("transitions", []), new.get("transitions", [])):
            for transition in source:
                if transition not in transitions:
                    transitions.append(transition)

        return {
            "start_url": new.get("start_url") or old.get("start_url", ""),
            "pages": list(pages.values()),
            "transitions": transitions,
            "run_count": old.get("run_count", 0),
        }
