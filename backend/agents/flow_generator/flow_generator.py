"""Generate clean business workflows from Discovery Agent execution history."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import FlowStep, HistoryEvent, Workflow


class FlowGenerator:
    """Deterministically reconstruct a human journey from discovery history.

    The generator accepts either a list of history events or a discovery result
    containing one of ``execution_history``, ``history``, or ``events``.  This
    keeps the consumer decoupled from how the Discovery Agent stores its state.
    """

    _ACTION_KEYS = ("action_type", "action", "type", "operation", "event")
    _TARGET_KEYS = ("target", "label", "text", "name", "element", "selector")
    _VALUE_KEYS = ("value", "input_value", "input", "text_value")
    _PAGE_TITLE_KEYS = ("page_title", "title", "page_name")
    _URL_KEYS = ("url", "page_url", "destination_url", "current_url")
    _NOISE_TERMS = (
        "telemetry", "observability", "trace", "span", "metric", "log",
        "framework", "internal", "network", "console", "screenshot",
        "observation", "planner", "classifier", "ranking",
    )
    _NAVIGATION_TYPES = {"navigate", "navigation", "page_view", "page_transition"}

    def generate(
        self,
        discovery_history: Any,
        *,
        flow_name: str | None = None,
        description: str | None = None,
    ) -> Workflow:
        """Return a normalized workflow without generating code or test cases."""
        events = [self._normalise(item) for item in self._history_items(discovery_history)]
        events = [event for event in events if event and self._is_business_event(event)]

        pages = self._pages(events)
        steps = self._steps(events)
        name = flow_name or self._infer_name(steps, pages)
        return Workflow(
            flow_name=name,
            description=description or self._describe(steps, pages),
            pages=pages,
            steps=steps,
        )

    def generate_json(self, discovery_history: Any, **kwargs: Any) -> dict[str, Any]:
        """Convenience API for consumers that require a JSON-serializable dict."""
        return self.generate(discovery_history, **kwargs).to_dict()

    def _history_items(self, history: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(history, Mapping):
            for key in ("execution_history", "history", "events"):
                candidate = history.get(key)
                if isinstance(candidate, list):
                    return [item for item in candidate if isinstance(item, Mapping)]
            # A single event is useful for streaming callers.
            return [history]
        if isinstance(history, list):
            return [item for item in history if isinstance(item, Mapping)]
        return []

    def _normalise(self, item: Mapping[str, Any]) -> HistoryEvent | None:
        action = item.get("action")
        nested = action if isinstance(action, Mapping) else {}
        merged = {**item, **nested}
        action_type = self._value(merged, self._ACTION_KEYS)
        target = self._value(merged, self._TARGET_KEYS)
        value = self._value(merged, self._VALUE_KEYS)
        page = merged.get("page")
        if isinstance(page, Mapping):
            merged = {**page, **merged}
        success = merged.get("success", merged.get("execution_succeeded", merged.get("status", "success")))
        return HistoryEvent(
            action_type=self._clean_text(action_type),
            target=self._clean_text(target),
            value=value,
            success=self._successful(success),
            page_title=self._clean_text(self._value(merged, self._PAGE_TITLE_KEYS)),
            url=self._clean_text(self._value(merged, self._URL_KEYS)),
            timestamp=merged.get("timestamp", merged.get("created_at")),
            raw=item,
        )

    @staticmethod
    def _value(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
        return next((data[key] for key in keys if data.get(key) not in (None, "")), None)

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        return " ".join(str(value).split()) if value not in (None, "") else None

    @staticmethod
    def _successful(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"success", "succeeded", "completed", "ok", "passed", "true", "1"}

    def _is_business_event(self, event: HistoryEvent) -> bool:
        action = (event.action_type or "").lower()
        if any(term in action for term in self._NOISE_TERMS) or not event.success:
            return False
        return bool(event.action_type or event.page_title or event.url)

    def _pages(self, events: list[HistoryEvent]) -> list[str]:
        pages: list[str] = []
        seen: set[str] = set()
        for event in events:
            label = event.page_title or self._page_label(event.url)
            if label and label.casefold() not in seen:
                pages.append(label)
                seen.add(label.casefold())
        return pages

    def _steps(self, events: list[HistoryEvent]) -> list[FlowStep]:
        steps: list[FlowStep] = []
        for event in events:
            action_type = (event.action_type or "").lower()
            if action_type in self._NAVIGATION_TYPES or not event.target:
                continue
            step = FlowStep(type=self._canonical_type(action_type), target=event.target, value=event.value)
            # Consecutive equal actions are explorer retries/duplicates.  This
            # also merges repeated clicks where no meaningful action intervened.
            if steps and steps[-1] == step:
                continue
            steps.append(step)
        return steps

    @staticmethod
    def _canonical_type(action_type: str) -> str:
        aliases = {"type": "fill", "input": "fill", "enter": "fill", "tap": "click", "press": "click"}
        return aliases.get(action_type, action_type)

    @staticmethod
    def _page_label(url: str | None) -> str | None:
        if not url:
            return None
        path = url.rstrip("/").rsplit("/", 1)[-1]
        return path.replace("-", " ").replace("_", " ").title() or "Home"

    @staticmethod
    def _infer_name(steps: list[FlowStep], pages: list[str]) -> str:
        if steps:
            return f"{steps[0].type.title()} {steps[-1].target}"
        if pages:
            return f"{pages[0]} Workflow"
        return "Discovered Workflow"

    @staticmethod
    def _describe(steps: list[FlowStep], pages: list[str]) -> str:
        if steps:
            actions = ", ".join(f"{step.type} {step.target}" for step in steps)
            return f"User journey: {actions}."
        if pages:
            return f"User journey through {' -> '.join(pages)}."
        return "No successful business actions were found in the discovery history."
