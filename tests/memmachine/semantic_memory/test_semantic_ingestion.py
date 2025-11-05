"""Tests for the ingestion service using the in-memory semantic storage."""

from typing import Iterable
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from memmachine.semantic_memory.semantic_ingestion import IngestionService
from memmachine.semantic_memory.semantic_llm import (
    LLMReducedFeature,
    SemanticConsolidateMemoryRes,
)
from memmachine.semantic_memory.semantic_model import (
    HistoryMessage,
    Resources,
    SemanticCommand,
    SemanticFeature,
    SemanticPrompt,
    SemanticType,
)
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    MockEmbedder,
    MockResourceRetriever,
)
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    InMemorySemanticStorage,
)


@pytest.fixture
def semantic_prompt() -> SemanticPrompt:
    return SemanticPrompt(
        update_prompt="update-prompt",
        consolidation_prompt="consolidation-prompt",
    )


@pytest.fixture
def semantic_type(semantic_prompt: SemanticPrompt) -> SemanticType:
    return SemanticType(
        name="Profile",
        tags={"food"},
        prompt=semantic_prompt,
    )


@pytest.fixture
def embedder_double() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def llm_model(mock_llm_model):
    return mock_llm_model


@pytest.fixture
def resources(
    embedder_double: MockEmbedder,
    llm_model,
    semantic_type: SemanticType,
) -> Resources:
    return Resources(
        embedder=embedder_double,
        language_model=llm_model,
        semantic_types=[semantic_type],
    )


@pytest.fixture
def resource_retriever(resources: Resources) -> MockResourceRetriever:
    return MockResourceRetriever(resources)


@pytest_asyncio.fixture
async def storage() -> InMemorySemanticStorage:
    store = InMemorySemanticStorage()
    await store.startup()
    yield store
    await store.delete_all()
    await store.cleanup()


@pytest_asyncio.fixture
async def ingestion_service(
    storage: InMemorySemanticStorage,
    resource_retriever: MockResourceRetriever,
) -> IngestionService:
    params = IngestionService.Params(
        semantic_storage=storage,
        resource_retriever=resource_retriever,
        consolidated_threshold=2,
    )
    return IngestionService(params)


@pytest.mark.asyncio
async def test_process_single_set_returns_when_no_messages(
    ingestion_service: IngestionService,
    storage: InMemorySemanticStorage,
    resource_retriever: MockResourceRetriever,
):
    await ingestion_service._process_single_set("user-123")

    assert resource_retriever.seen_ids == ["user-123"]
    assert await storage.get_feature_set(set_ids=["user-123"]) == []
    assert (
        await storage.get_history_messages(
            set_ids=["user-123"],
            is_ingested=False,
        )
        == []
    )


@pytest.mark.asyncio
async def test_process_single_set_applies_commands(
    ingestion_service: IngestionService,
    storage: InMemorySemanticStorage,
    embedder_double: MockEmbedder,
    semantic_type: SemanticType,
    monkeypatch,
):
    message_id = await storage.add_history(content="I love blue cars")
    await storage.add_history_to_set(set_id="user-123", history_id=message_id)

    await storage.add_feature(
        set_id="user-123",
        type_name=semantic_type.name,
        feature="favorite_motorcycle",
        value="old bike",
        tag="bike",
        embedding=np.array([1.0, 1.0]),
    )

    commands = [
        SemanticCommand(
            command="add",
            feature="favorite_car",
            tag="car",
            value="blue",
        ),
        SemanticCommand(
            command="delete",
            feature="favorite_motorcycle",
            tag="bike",
            value="",
        ),
    ]
    llm_feature_update_mock = AsyncMock(return_value=commands)
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_feature_update",
        llm_feature_update_mock,
    )

    await ingestion_service._process_single_set("user-123")

    llm_feature_update_mock.assert_awaited_once()
    features = await storage.get_feature_set(
        set_ids=["user-123"],
        type_names=[semantic_type.name],
        load_citations=True,
    )
    assert len(features) == 1
    feature = features[0]
    assert feature.feature == "favorite_car"
    assert feature.value == "blue"
    assert feature.tag == "car"
    assert feature.metadata.citations is not None
    assert [c.metadata.id for c in feature.metadata.citations] == [message_id]

    remaining = await storage.get_feature_set(
        set_ids=["user-123"],
        feature_names=["favorite_motorcycle"],
    )
    assert remaining == []

    assert (
        await storage.get_history_messages(
            set_ids=["user-123"],
            is_ingested=False,
        )
        == []
    )
    ingested = await storage.get_history_messages(
        set_ids=["user-123"],
        is_ingested=True,
    )
    assert [msg.metadata.id for msg in ingested] == [message_id]
    assert embedder_double.ingest_calls == [["blue"]]


@pytest.mark.asyncio
async def test_consolidation_groups_by_tag(
    ingestion_service: IngestionService,
    storage: InMemorySemanticStorage,
    resources: Resources,
    semantic_type: SemanticType,
    monkeypatch,
):
    first_history = await storage.add_history(content="thin crust")
    second_history = await storage.add_history(content="deep dish")

    first_feature = await storage.add_feature(
        set_id="user-456",
        type_name=semantic_type.name,
        feature="pizza",
        value="thin crust",
        tag="food",
        embedding=np.array([1.0, -1.0]),
    )
    second_feature = await storage.add_feature(
        set_id="user-456",
        type_name=semantic_type.name,
        feature="pizza",
        value="deep dish",
        tag="food",
        embedding=np.array([2.0, -2.0]),
    )
    await storage.add_citations(first_feature, [first_history])
    await storage.add_citations(second_feature, [second_history])

    dedupe_mock = AsyncMock()
    monkeypatch.setattr(ingestion_service, "_deduplicate_features", dedupe_mock)

    await ingestion_service._consolidate_set_memories_if_applicable(
        set_id="user-456",
        resources=resources,
    )

    assert dedupe_mock.await_count == 1
    call = dedupe_mock.await_args_list[0]
    memories: list[SemanticFeature] = call.kwargs["memories"]
    assert {m.metadata.id for m in memories} == {first_feature, second_feature}
    assert call.kwargs["set_id"] == "user-456"
    assert call.kwargs["semantic_type"] == semantic_type
    assert call.kwargs["resources"] == resources


@pytest.mark.asyncio
async def test_deduplicate_features_merges_and_relabels(
    ingestion_service: IngestionService,
    storage: InMemorySemanticStorage,
    resources: Resources,
    semantic_type: SemanticType,
    monkeypatch,
):
    keep_history = await storage.add_history(content="keep")
    drop_history = await storage.add_history(content="drop")

    keep_feature_id = await storage.add_feature(
        set_id="user-789",
        type_name=semantic_type.name,
        feature="pizza",
        value="original pizza",
        tag="food",
        embedding=np.array([1.0, 0.5]),
    )
    drop_feature_id = await storage.add_feature(
        set_id="user-789",
        type_name=semantic_type.name,
        feature="pizza",
        value="duplicate pizza",
        tag="food",
        embedding=np.array([2.0, 1.0]),
    )

    await storage.add_citations(keep_feature_id, [keep_history])
    await storage.add_citations(drop_feature_id, [drop_history])

    memories = await storage.get_feature_set(
        set_ids=["user-789"],
        type_names=[semantic_type.name],
        load_citations=True,
    )

    consolidated_feature = LLMReducedFeature(
        tag="food",
        feature="pizza",
        value="consolidated pizza",
    )
    llm_consolidate_mock = AsyncMock(
        return_value=SemanticConsolidateMemoryRes(
            consolidated_memories=[consolidated_feature],
            keep_memories=[keep_feature_id],
        )
    )
    monkeypatch.setattr(
        "memmachine.semantic_memory.semantic_ingestion.llm_consolidate_features",
        llm_consolidate_mock,
    )

    original_add_citations = storage.add_citations

    async def _normalized_add_citations(
        feature_id: int,
        history_items: Iterable[HistoryMessage | int],
    ):
        normalized: list[int] = []
        for item in history_items:
            if isinstance(item, HistoryMessage):
                if item.metadata.id is not None:
                    normalized.append(item.metadata.id)
            else:
                normalized.append(int(item))
        await original_add_citations(feature_id, normalized)

    monkeypatch.setattr(
        ingestion_service._semantic_storage,
        "add_citations",
        _normalized_add_citations,
    )

    await ingestion_service._deduplicate_features(
        set_id="user-789",
        memories=memories,
        semantic_type=semantic_type,
        resources=resources,
    )

    llm_consolidate_mock.assert_awaited_once()
    assert await storage.get_feature(drop_feature_id, load_citations=True) is None
    kept_feature = await storage.get_feature(keep_feature_id, load_citations=True)
    assert kept_feature is not None
    assert kept_feature.value == "original pizza"

    all_features = await storage.get_feature_set(
        set_ids=["user-789"],
        type_names=[semantic_type.name],
        load_citations=True,
    )
    consolidated = next(
        (f for f in all_features if f.value == "consolidated pizza"),
        None,
    )
    assert consolidated is not None
    assert consolidated.tag == "food"
    assert consolidated.feature == "pizza"
    assert consolidated.metadata.citations is not None
    assert [c.metadata.id for c in consolidated.metadata.citations] == [drop_history]
    assert resources.embedder.ingest_calls == [["consolidated pizza"]]
