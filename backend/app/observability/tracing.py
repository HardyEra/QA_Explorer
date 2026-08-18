"""Observability contracts shared by the application.

Business services depend on these small contracts, rather than a vendor SDK.  A
different backend can implement :class:`ObservabilityBackend` without changing
the discovery workflow.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

from pydantic import BaseModel, Field


class TraceMetadata(BaseModel):
    project_name: str
    session_id: str
    exploration_id: str
    current_page: str | None = None
    visited_pages: list[str] = Field(default_factory=list)
    visited_actions: list[str] = Field(default_factory=list)
    current_goal: str = "Autonomously discover application pages and actions"


class Observation(Protocol):
    """Mutable handle exposed while a span or generation is active."""

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None,
               usage_details: dict[str, int] | None = None,
               status_message: str | None = None) -> None: ...


class ObservabilityBackend(Protocol):
    """The only telemetry dependency application services need."""

    def trace(self, name: str, *, metadata: TraceMetadata,
              input: Any = None) -> AbstractContextManager[Observation]: ...

    def span(self, name: str, *, input: Any = None,
             metadata: dict[str, Any] | None = None) -> AbstractContextManager[Observation]: ...

    def generation(self, name: str, *, model: str, temperature: float | None,
                   input: Any = None, metadata: dict[str, Any] | None = None) -> AbstractContextManager[Observation]: ...

    def record_exception(self, exception: BaseException, *, input: Any = None,
                         output: Any = None, retry_count: int = 0,
                         context: dict[str, Any] | None = None) -> None: ...

    def score_current_trace(self, name: str, value: float | str, *,
                            data_type: str = "NUMERIC", comment: str | None = None,
                            metadata: dict[str, Any] | None = None) -> None: ...

    def flush(self) -> None: ...


class NoopObservation:
    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None,
               usage_details: dict[str, int] | None = None,
               status_message: str | None = None) -> None:
        return None


class NoopObservability:
    """Safe local/default backend when telemetry credentials are not configured."""

    def trace(self, name: str, *, metadata: TraceMetadata, input: Any = None) -> AbstractContextManager[Observation]:
        return nullcontext(NoopObservation())

    def span(self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None) -> AbstractContextManager[Observation]:
        return nullcontext(NoopObservation())

    def generation(self, name: str, *, model: str, temperature: float | None,
                   input: Any = None, metadata: dict[str, Any] | None = None) -> AbstractContextManager[Observation]:
        return nullcontext(NoopObservation())

    def record_exception(self, exception: BaseException, *, input: Any = None,
                         output: Any = None, retry_count: int = 0,
                         context: dict[str, Any] | None = None) -> None:
        return None

    def score_current_trace(self, name: str, value: float | str, *,
                            data_type: str = "NUMERIC", comment: str | None = None,
                            metadata: dict[str, Any] | None = None) -> None:
        return None

    def flush(self) -> None:
        return None
