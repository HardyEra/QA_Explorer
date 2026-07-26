"""Reusable instrumentation decorators; no Langfuse SDK is imported here."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable


def _safe(value: Any) -> Any:
    """Return telemetry-friendly state without requiring application DTOs."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {key: _safe(item) for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    return value


def traced_node(name: str) -> Callable:
    """Create a span around a LangGraph node and attach its state transition."""
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(self, state, *args, **kwargs):
            metadata = self.trace_metadata(state) if hasattr(self, "trace_metadata") else None
            with self.observability.span(
                f"langgraph.{name}", input=_safe(state), metadata=metadata
            ) as span:
                try:
                    result = function(self, state, *args, **kwargs)
                    span.update(output=_safe(result))
                    return result
                except Exception as exc:
                    self.observability.record_exception(exc, input=_safe(state))
                    raise
        return wrapped
    return decorator
