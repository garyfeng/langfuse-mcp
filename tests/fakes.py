"""Fake classes for testing langfuse-mcp against Langfuse v3 semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class FakeTrace:
    """Simple trace model inspired by Langfuse v3 responses."""

    id: str
    name: str
    user_id: Optional[str]
    session_id: Optional[str]
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    observations: list["FakeObservation"] = field(default_factory=list)


@dataclass
class FakeObservation:
    """Observation model with nested structure similar to Langfuse v3."""

    id: str
    type: str
    name: str
    status: str
    start_time: datetime
    end_time: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    level: str | None = None


@dataclass
class FakeSession:
    """Session model mimicking the Langfuse v3 SDK."""

    id: str
    user_id: str
    created_at: datetime
    trace_ids: list[str] = field(default_factory=list)


@dataclass
class FakePaginatedResponse:
    """Generic paginated container returned by resource list methods."""

    items: list[Any]
    next_page: str | None = None
    total: int | None = None


class FakeTracesResource:
    """Resource facade that tracks calls to trace listing/getting."""

    def __init__(self, store: "FakeDataStore") -> None:
        self._store = store
        self.last_list_kwargs: dict[str, Any] | None = None
        self.last_get_kwargs: dict[str, Any] | None = None

    def list(
        self,
        *,
        limit: int | None = None,
        order: str | None = None,
        include_observations: bool = False,
        **filters: Any,
    ) -> FakePaginatedResponse:
        """Return traces matching filters; mirrors Langfuse v3 semantics."""

        normalized_filters = filters.get("filters", filters)
        self.last_list_kwargs = {
            "limit": limit,
            "order": order,
            "include_observations": include_observations,
            "filters": normalized_filters,
        }
        traces = list(self._store.traces.values())
        if include_observations:
            for trace in traces:
                trace.observations = [self._store.observations[o_id] for o_id in trace.metadata.get("observation_ids", [])]
        return FakePaginatedResponse(items=traces, next_page=None, total=len(traces))

    def get(self, trace_id: str, *, include_observations: bool = False) -> FakeTrace | None:
        """Return a single trace model."""

        self.last_get_kwargs = {"trace_id": trace_id, "include_observations": include_observations}
        trace = self._store.traces.get(trace_id)
        if trace and include_observations:
            trace.observations = [self._store.observations[o_id] for o_id in trace.metadata.get("observation_ids", [])]
        return trace


class FakeObservationsResource:
    """Resource for observations supporting list/get operations."""

    def __init__(self, store: "FakeDataStore") -> None:
        self._store = store
        self.last_list_kwargs: dict[str, Any] | None = None
        self.last_get_kwargs: dict[str, Any] | None = None

    def list(self, *, limit: int | None = None, **filters: Any) -> FakePaginatedResponse:
        self.last_list_kwargs = {"limit": limit, "filters": filters}
        observations = list(self._store.observations.values())
        return FakePaginatedResponse(items=observations, next_page=None, total=len(observations))

    def get(self, observation_id: str) -> FakeObservation | None:
        self.last_get_kwargs = {"observation_id": observation_id}
        return self._store.observations.get(observation_id)


class FakeSessionsResource:
    """Resource for sessions supporting list/get operations."""

    def __init__(self, store: "FakeDataStore") -> None:
        self._store = store
        self.last_list_kwargs: dict[str, Any] | None = None
        self.last_get_kwargs: dict[str, Any] | None = None

    def list(self, *, limit: int | None = None, **filters: Any) -> FakePaginatedResponse:
        self.last_list_kwargs = {"limit": limit, "filters": filters}
        sessions = list(self._store.sessions.values())
        return FakePaginatedResponse(items=sessions, next_page=None, total=len(sessions))

    def get(self, session_id: str) -> FakeSession | None:
        self.last_get_kwargs = {"session_id": session_id}
        return self._store.sessions.get(session_id)


class FakeDataStore:
    """In-memory backing store shared across resource facades."""

    def __init__(self) -> None:
        now = datetime(2023, 1, 1, tzinfo=timezone.utc)
        self.observations: dict[str, FakeObservation] = {
            "obs_1": FakeObservation(
                id="obs_1",
                type="SPAN",
                name="root_span",
                status="SUCCEEDED",
                start_time=now,
                end_time=now,
                metadata={"code.filepath": "app.py"},
            )
        }
        self.traces: dict[str, FakeTrace] = {
            "trace_1": FakeTrace(
                id="trace_1",
                name="test-trace",
                user_id="user_1",
                session_id="session_1",
                created_at=now,
                metadata={"observation_ids": ["obs_1"]},
                tags=["unit-test"],
            )
        }
        self.sessions: dict[str, FakeSession] = {
            "session_1": FakeSession(
                id="session_1",
                user_id="user_1",
                created_at=now,
                trace_ids=["trace_1"],
            )
        }


class FakeLangfuse:
    """Langfuse client double exposing v3-style resources only."""

    def __init__(self) -> None:
        self._store = FakeDataStore()
        self.traces = FakeTracesResource(self._store)
        self.observations = FakeObservationsResource(self._store)
        self.sessions = FakeSessionsResource(self._store)
        self.closed = False

    # v2 helpers are intentionally unsupported to catch accidental use during migration
    def fetch_traces(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
        raise AssertionError("Langfuse v3 SDK does not expose fetch_traces")

    def fetch_trace(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
        raise AssertionError("Langfuse v3 SDK does not expose fetch_trace")

    def fetch_observations(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
        raise AssertionError("Langfuse v3 SDK does not expose fetch_observations")

    def fetch_observation(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
        raise AssertionError("Langfuse v3 SDK does not expose fetch_observation")

    def fetch_sessions(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
        raise AssertionError("Langfuse v3 SDK does not expose fetch_sessions")

    def close(self) -> None:
        """mimic the shutdown hook exposed in Langfuse v3."""

        self.closed = True

    # For compatibility with cleanup logic that still calls flush/shutdown, keep no-op stubs
    def flush(self) -> None:  # pragma: no cover - backwards compatibility during migration
        return None

    def shutdown(self) -> None:  # pragma: no cover - backwards compatibility during migration
        self.close()


class FakeContext:
    """Mimic `mcp.server.fastmcp.Context` used by the tools."""

    def __init__(self, state: Any) -> None:
        self.request_context = type("_RC", (), {"lifespan_context": state})
