"""Unit tests for the top-level :mod:`memmachine.main.memmachine` module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memmachine.main.memmachine import MemMachine, MemoryType


@pytest.fixture
def minimal_conf():
    """Provide a minimal configuration object for MemMachine defaults."""

    return SimpleNamespace(
        episodic_memory=SimpleNamespace(
            short_term_memory=None,
            long_term_memory=None,
        )
    )


@pytest.fixture
def patched_resource_manager(monkeypatch):
    """Patch :class:`ResourceManagerImpl` so real resources are not created."""

    fake_manager = MagicMock()
    monkeypatch.setattr(
        "memmachine.main.memmachine.ResourceManagerImpl",
        MagicMock(return_value=fake_manager),
    )
    return fake_manager


class DummySessionData:
    """Simple SessionData implementation for tests."""

    def __init__(self, session_key: str):
        self._session_key = session_key

    @property
    def session_key(self) -> str:  # pragma: no cover - trivial accessor
        return self._session_key


def test_with_default_episodic_memory_conf_uses_fallbacks(
    minimal_conf, patched_resource_manager
):
    """Verify default episodic memory configuration uses sensible fallbacks."""

    memmachine = MemMachine(minimal_conf)

    conf = memmachine._with_default_episodic_memory_conf(session_key="session-1")

    assert conf.session_key == "session-1"
    assert conf.long_term_memory is not None
    assert conf.short_term_memory is not None
    assert conf.long_term_memory.embedder == "default"
    assert conf.long_term_memory.reranker == "default"
    assert conf.long_term_memory.vector_graph_store == "default_store"
    assert conf.short_term_memory.llm_model == "gpt-4.1"
    assert (
        "You are a helpful assistant." in conf.short_term_memory.summary_prompt_system
    )
    assert (
        "Based on the following episodes" in conf.short_term_memory.summary_prompt_user
    )


@pytest.mark.asyncio
async def test_create_session_passes_generated_config(
    monkeypatch, minimal_conf, patched_resource_manager
):
    """Ensure ``create_session`` builds an episodic config and sends it to the manager."""

    session_manager = AsyncMock()
    patched_resource_manager.get_session_data_manager = AsyncMock(
        return_value=session_manager
    )

    memmachine = MemMachine(minimal_conf)

    await memmachine.create_session(
        "alpha",
        description="demo",
        embedder_name="custom-embed",
        reranker_name="custom-reranker",
    )

    session_manager.create_new_session.assert_awaited_once()
    _, kwargs = session_manager.create_new_session.await_args
    episodic_conf = kwargs["param"]

    assert episodic_conf.long_term_memory.embedder == "custom-embed"
    assert episodic_conf.long_term_memory.reranker == "custom-reranker"
    assert episodic_conf.short_term_memory.session_key == "alpha"
    assert kwargs["description"] == "demo"


@pytest.mark.asyncio
async def test_query_search_runs_targeted_memory_tasks(
    monkeypatch, minimal_conf, patched_resource_manager
):
    """``query_search`` should dispatch queries to both episodic and semantic memory."""

    dummy_session = DummySessionData("s1")

    async_episodic = AsyncMock(return_value="episodic-response")
    monkeypatch.setattr(MemMachine, "_search_episodic_memory", async_episodic)

    semantic_manager = MagicMock()
    semantic_manager.search = AsyncMock(return_value=["semantic-response"])
    patched_resource_manager.get_semantic_session_manager = AsyncMock(
        return_value=semantic_manager
    )

    memmachine = MemMachine(minimal_conf)

    result = await memmachine.query_search(
        dummy_session,
        target_memories=[MemoryType.Episodic, MemoryType.Semantic],
        query="hello world",
    )

    async_episodic.assert_awaited_once()
    semantic_manager.search.assert_awaited_once()

    assert result.episodic_memory == "episodic-response"
    assert result.semantic_memory == ["semantic-response"]
