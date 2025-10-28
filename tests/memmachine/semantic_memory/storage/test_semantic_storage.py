from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pytest
import pytest_asyncio

from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


@pytest.fixture(
    params=[pytest.param("postgres", marks=pytest.mark.integration), "inmemory"]
)
def storage(request):
    match request.param:
        case "postgres":
            return request.getfixturevalue("sqlalchemy_profile_storage")
        case "inmemory":
            return request.getfixturevalue("in_memory_profile_storage")
        case _:
            raise ValueError(f"Unknown storage type: {request.param}")


@pytest.mark.asyncio
async def test_empty_storage(storage: SemanticStorageBase):
    assert await storage.get_feature_set(set_id="user") == []


@pytest.mark.asyncio
async def test_multiple_features(
    storage: SemanticStorageBase,
    with_multiple_features,
):
    # Given a storage with two features
    # When we retrieve the profile
    profile_result = await storage.get_feature_set(set_id="user")
    profile_result = SemanticFeature.group_features(profile_result)

    assert len(profile_result) == 1

    key, expected_profile = with_multiple_features

    test_user_profile = profile_result[key]
    expected_test_user_profile = expected_profile[key]

    # Then the profile should contain both features
    assert len(test_user_profile) == 2
    for i in range(len(test_user_profile)):
        assert test_user_profile[i].value == expected_test_user_profile[i]["value"]


@pytest.mark.asyncio
async def test_delete_feature(storage: SemanticStorageBase):
    idx_a = await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=np.array([1.0] * 1536, dtype=float),
    )

    # Given a storage with a single feature
    features = await storage.get_feature_set(set_id="user")
    assert len(features) == 1
    assert features[0].value == "pizza"

    # When we delete the feature
    await storage.delete_features([idx_a])

    features = await storage.get_feature_set(set_id="user")

    # Then the feature should no longer exist
    assert features == []


@pytest.mark.asyncio
async def test_delete_feature_set_by_set_id(
    storage: SemanticStorageBase,
    with_multiple_sets,
):
    # Given a storage with two sets
    res_a = await storage.get_feature_set(set_id="user1")
    res_a = SemanticFeature.group_features(res_a)

    res_b = await storage.get_feature_set(set_id="user2")
    res_b = SemanticFeature.group_features(res_b)

    key, expected = with_multiple_sets

    set_a = [{"value": f.value} for f in res_a[key]]
    set_b = [{"value": f.value} for f in res_b[key]]

    assert set_a == expected["user1"]
    assert set_b == expected["user2"]

    # When we delete the first set
    await storage.delete_feature_set(set_id="user1")

    # Then the first set should be empty
    res_delete_a = await storage.get_feature_set(set_id="user1")
    assert res_delete_a == []

    # And the second set should still exist
    res_delete_b = await storage.get_feature_set(set_id="user2")
    res_delete_b = SemanticFeature.group_features(res_delete_b)
    set_delete_b = [{"value": f.value} for f in res_delete_b[key]]
    assert set_delete_b == expected["user2"]


@pytest_asyncio.fixture
async def oposite_vector_features(storage: SemanticStorageBase):
    embed_a = np.array([1.0], dtype=float)
    value_a = "pizza"

    embed_b = np.array([0.0], dtype=float)
    value_b = "sushi"

    id_a = await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        tag="food",
        feature="likes",
        value=value_a,
        embedding=embed_a,
    )
    id_b = await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        tag="food",
        feature="likes",
        value=value_b,
        embedding=embed_b,
    )

    yield [
        (embed_a, value_a),
        (embed_b, value_b),
    ]

    await storage.delete_features([id_a, id_b])

@pytest.mark.asyncio
async def test_get_feature_set_basic_vector_search(
        storage: SemanticStorageBase,
        oposite_vector_features,
):
    # Given a storage with fully distinct features
    embed_a, value_a = oposite_vector_features[0]
    embed_b, value_b = oposite_vector_features[1]

    # When doing a vector search
    results = await storage.get_feature_set(
        set_id="user",
        k=10,
        vector_search_opts=SemanticStorageBase.VectorSearchOpts(
            query_embedding=embed_a,
            min_cos=0.0,
        ),
    )

    # Then the results should be the two distinct features
    # With value_a being the first and value_b being the second
    results = [f.value for f in results]
    assert results == [ value_a, value_b ]

@pytest.mark.asyncio
async def test_get_feature_set_min_cos_vector_search(storage: SemanticStorageBase):
    pass

@pytest.mark.asyncio
async def test_complex_feature_lifecycle(storage: SemanticStorageBase):
    embed = np.array([1.0] * 1536, dtype=float)

    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="pizza",
        tag="food",
        embedding=embed,
    )
    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="likes",
        value="sushi",
        tag="food",
        embedding=embed,
    )
    await storage.add_feature(
        set_id="user",
        semantic_type_id="tenant_A",
        feature="color",
        value="blue",
        tag="prefs",
        embedding=embed,
    )

    profile_default = await storage.get_feature_set(set_id="user")
    profile_default = SemanticFeature.group_features(profile_default)
    assert ("default", "food", "likes") in profile_default

    likes_entries = profile_default[("default", "food", "likes")]
    if not isinstance(likes_entries, list):
        likes_entries = [likes_entries]
    assert {item.value for item in likes_entries} == {"pizza", "sushi"}

    tenant_profile = await storage.get_feature_set(
        set_id="user",
        semantic_type_id="tenant_A",
    )
    tenant_profile = SemanticFeature.group_features(tenant_profile)
    assert tenant_profile[("tenant_A", "prefs", "color")][0].value == "blue"

    await storage.delete_feature_set(
        set_id="user",
        semantic_type_id="default",
        feature_name="likes",
        tag="food",
    )

    after_delete = await storage.get_feature_set(set_id="user")
    after_delete = SemanticFeature.group_features(after_delete)
    assert ("default", "food", "likes") not in after_delete

    await storage.delete_feature_set(set_id="user", semantic_type_id="tenant_A")
    tenant_only = await storage.get_feature_set(
        set_id="user",
        semantic_type_id="tenant_A",
    )
    assert tenant_only == []

    await storage.delete_feature_set(set_id="user")
    assert await storage.get_feature_set(set_id="user") == []


@pytest.mark.asyncio
async def test_complex_semantic_search_and_citations(storage: SemanticStorageBase):
    history_id = await storage.add_history(
        set_id="user",
        content="context note",
        metadata={"source": "chat"},
    )

    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="topic",
        value="ai",
        tag="facts",
        embedding=np.array([1.0, 0.0]),
        citations=[history_id],
    )
    await storage.add_feature(
        set_id="user",
        semantic_type_id="default",
        feature="topic",
        value="music",
        tag="facts",
        embedding=np.array([0.0, 1.0]),
    )

    results = await storage.get_feature_set(
        set_id="user",
        k=10,
        vector_search_opts=SemanticStorageBase.VectorSearchOpts(
            query_embedding=np.array([1.0, 0.0]),
            min_cos=0.0,
        ),
        # include_citations=True,
    )

    assert results is not None
    assert [entry.value for entry in results] == ["ai", "music"]

    # assert (
    #     results[0]["metadata"]["similarity_score"]
    #     > results[1]["metadata"]["similarity_score"]
    # )
    # assert results[0]["metadata"]["citations"] == ["context note"]

    filtered = await storage.get_feature_set(
        set_id="user",
        k=1,
        vector_search_opts=SemanticStorageBase.VectorSearchOpts(
            query_embedding=np.array([1.0, 0.0]),
            min_cos=0.5,
        ),
        # include_citations=False,
    )
    assert len(filtered) == 1
    assert filtered[0]["value"] == "ai"

    feature_ids = [entry["metadata"]["id"] for entry in results]
    citation_map = [cid for cid in await storage.get_all_citations_for_ids(feature_ids)]
    assert citation_map == [history_id["id"]]

    await storage.delete_features(feature_ids[:1])
    remaining = await storage.get_feature_set(
        set_id="user",
        semantic_type_id="default",
        tag="facts",
        feature_name="topic",
    )
    assert len(remaining) == 1
    assert remaining[0].value == "music"


@pytest.mark.asyncio
async def test_complex_history_workflow(storage: SemanticStorageBase):
    h1_id = await storage.add_history(
        set_id="user",
        content="first",
        metadata={},
    )
    await asyncio.sleep(0.01)
    h2_id = await storage.add_history(
        set_id="user",
        content="second",
        metadata={},
    )
    await asyncio.sleep(0.01)
    cutoff = datetime.now()
    await asyncio.sleep(0.01)
    h3_id = await storage.add_history(
        set_id="user",
        content="third",
        metadata={},
    )

    all_messages = await storage.get_history_messages(set_id="user")
    all_messages = [m.content for m in all_messages]
    assert all_messages == ["first", "second", "third"]

    latest_uningested = await storage.get_history_messages(
        set_id="user",
        k=2,
        is_ingested=False,
    )
    assert [entry.content for entry in latest_uningested] == ["first", "second"]

    assert await storage.get_history_messages_count(is_ingested=False) == 3
    await storage.mark_messages_ingested(ids=[h1_id, h2_id])
    assert await storage.get_history_messages_count(is_ingested=False) == 1
    ingested = await storage.get_history_messages(
        set_id="user",
        is_ingested=True,
    )
    assert {entry.metadata.id for entry in ingested} == {h1_id, h2_id}

    uningested = await storage.get_history_messages(
        set_id="user",
        is_ingested=False,
    )
    assert {entry.metadata.id for entry in uningested} == {h3_id}

    window = await storage.get_history_messages(
        set_id="user",
        end_time=cutoff,
    )
    assert window == ["first", "second"]

    await storage.delete_history_messages(
        set_id="user",
        end_time=cutoff,
    )
    remaining = await storage.get_history_messages(set_id="user")
    assert remaining == ["third"]

    await asyncio.sleep(0.01)
    await storage.delete_history_messages(
        set_id="user",
    )
    assert await storage.get_history_messages(set_id="user") == []
