"""Unit tests for langfuse-mcp tool functions."""

from __future__ import annotations

import asyncio

import pytest

from tests.fakes import FakeContext, FakeLangfuse


@pytest.fixture()
def state(tmp_path):
    """Return an MCPState instance using the fake client."""
    from langfuse_mcp.__main__ import MCPState

    return MCPState(langfuse_client=FakeLangfuse(), dump_dir=str(tmp_path))


def test_fetch_traces_with_observations(state):
    """fetch_traces should use the v3 traces resource and embed observations."""
    from langfuse_mcp.__main__ import fetch_traces

    ctx = FakeContext(state)
    result = asyncio.run(
        fetch_traces(
            ctx,
            age=10,
            name=None,
            user_id=None,
            session_id=None,
            metadata=None,
            page=1,
            limit=50,
            tags=None,
            include_observations=True,
            output_mode="compact",
        )
    )
    assert result["metadata"]["item_count"] == 1
    assert result["data"][0]["id"] == "trace_1"
    assert isinstance(result["data"][0]["observations"], list)
    assert result["data"][0]["observations"][0]["id"] == "obs_1"
    assert state.langfuse_client.traces.last_list_kwargs is not None
    assert state.langfuse_client.traces.last_list_kwargs["include_observations"] is True
    assert state.langfuse_client.traces.last_list_kwargs["limit"] == 50


def test_fetch_trace(state):
    """fetch_trace should pull from the v3 traces resource."""
    from langfuse_mcp.__main__ import fetch_trace

    ctx = FakeContext(state)
    result = asyncio.run(fetch_trace(ctx, trace_id="trace_1", include_observations=True, output_mode="compact"))
    assert result["data"]["id"] == "trace_1"
    assert result["data"]["observations"][0]["id"] == "obs_1"
    assert state.langfuse_client.traces.last_get_kwargs == {"trace_id": "trace_1", "include_observations": True}


def test_fetch_observations(state):
    """fetch_observations should call the v3 observations resource."""
    from langfuse_mcp.__main__ import fetch_observations

    ctx = FakeContext(state)
    result = asyncio.run(
        fetch_observations(
            ctx,
            type=None,
            age=10,
            name=None,
            user_id=None,
            trace_id=None,
            parent_observation_id=None,
            page=1,
            limit=50,
            output_mode="compact",
        )
    )
    assert result["metadata"]["item_count"] == 1
    assert result["data"][0]["id"] == "obs_1"
    assert state.langfuse_client.observations.last_list_kwargs is not None
    assert state.langfuse_client.observations.last_list_kwargs["limit"] == 50


def test_fetch_observation(state):
    """fetch_observation should hit the observations resource."""
    from langfuse_mcp.__main__ import fetch_observation

    ctx = FakeContext(state)
    result = asyncio.run(fetch_observation(ctx, observation_id="obs_1", output_mode="compact"))
    assert result["data"]["id"] == "obs_1"
    assert state.langfuse_client.observations.last_get_kwargs == {"observation_id": "obs_1"}


def test_fetch_sessions(state):
    """fetch_sessions should rely on the v3 sessions resource."""
    from langfuse_mcp.__main__ import fetch_sessions

    ctx = FakeContext(state)
    result = asyncio.run(fetch_sessions(ctx, age=10, page=1, limit=50, output_mode="compact"))
    assert result["metadata"]["item_count"] == 1
    assert result["data"][0]["id"] == "session_1"
    assert state.langfuse_client.sessions.last_list_kwargs is not None
    assert state.langfuse_client.sessions.last_list_kwargs["limit"] == 50


def test_get_session_details(state):
    """get_session_details should reuse the v3 traces resource."""
    from langfuse_mcp.__main__ import get_session_details

    ctx = FakeContext(state)
    result = asyncio.run(get_session_details(ctx, session_id="session_1", include_observations=True, output_mode="compact"))
    assert result["data"]["found"] is True
    assert result["data"]["trace_count"] == 1
    assert state.langfuse_client.traces.last_list_kwargs is not None
    assert state.langfuse_client.traces.last_list_kwargs["filters"]["session_id"] == "session_1"
