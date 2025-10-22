import json
import time

import numpy as np
import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer

from memmachine.profile_memory.storage.neo4j_profile import Neo4jProfileStorage

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def neo4j_connection_info():
    username = "neo4j"
    password = "password"

    with Neo4jContainer(
        image="neo4j:latest",
        username=username,
        password=password,
    ) as neo4j:
        yield {
            "uri": neo4j.get_connection_url(),
            "username": username,
            "password": password,
        }


@pytest_asyncio.fixture
async def neo4j_driver(neo4j_connection_info):
    driver = AsyncGraphDatabase.driver(
        neo4j_connection_info["uri"],
        auth=(
            neo4j_connection_info["username"],
            neo4j_connection_info["password"],
        ),
    )
    yield driver
    await driver.close()


@pytest_asyncio.fixture
async def profile_storage(neo4j_driver):
    storage = Neo4jProfileStorage(
        Neo4jProfileStorage.Params(
            driver=neo4j_driver,
        )
    )
    await storage.startup()
    yield storage
    await storage.cleanup()


@pytest_asyncio.fixture(autouse=True)
async def cleanup_database(profile_storage, neo4j_driver):
    await profile_storage.delete_all()
    await neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")
    yield


@pytest.mark.asyncio
async def test_add_get_and_delete_profile_entries(profile_storage):
    history = await profile_storage.add_history(
        user_id="user-1",
        content="User enjoys pizza nights.",
        metadata={"speaker": "tester"},
        isolations={"region": "us"},
    )

    await profile_storage.add_profile_feature(
        user_id="user-1",
        feature="favorite_food",
        value="pizza",
        tag="preferences",
        embedding=np.array([1.0, 0.0], dtype=float),
        metadata={"source": "unit-test"},
        isolations={"region": "us"},
        citations=[history["id"]],
    )

    profile = await profile_storage.get_profile("user-1", {"region": "us"})
    assert "preferences" in profile
    assert "favorite_food" in profile["preferences"]
    assert profile["preferences"]["favorite_food"]["value"] == "pizza"

    citations = await profile_storage.get_citation_list(
        "user-1",
        feature="favorite_food",
        value="pizza",
        tag="preferences",
        isolations={"region": "us"},
    )
    assert citations == [history["id"]]

    await profile_storage.delete_profile_feature(
        "user-1",
        feature="favorite_food",
        tag="preferences",
        isolations={"region": "us"},
    )
    profile_after_delete = await profile_storage.get_profile("user-1", {"region": "us"})
    assert profile_after_delete == {}


@pytest.mark.asyncio
async def test_semantic_search_with_citations(profile_storage):
    history = await profile_storage.add_history(
        "user-2",
        "User cooks pasta every weekend.",
        metadata={"speaker": "tester"},
        isolations={},
    )

    await profile_storage.add_profile_feature(
        "user-2",
        feature="favorite_meal",
        value="pasta",
        tag="habits",
        embedding=np.array([1.0, 0.0], dtype=float),
        isolations={},
        citations=[history["id"]],
    )

    await profile_storage.add_profile_feature(
        "user-2",
        feature="favorite_sport",
        value="cycling",
        tag="habits",
        embedding=np.array([0.0, 1.0], dtype=float),
        isolations={},
    )

    results = await profile_storage.semantic_search(
        "user-2",
        np.array([1.0, 0.0]),
        k=5,
        min_cos=0.1,
        isolations={},
        include_citations=False,
    )
    assert len(results) == 1
    assert results[0]["metadata"]["similarity_score"] > 0.99
    assert "citations" not in results[0]["metadata"]

    results_with_citations = await profile_storage.semantic_search(
        "user-2",
        np.array([1.0, 0.0]),
        k=5,
        min_cos=0.1,
        isolations={},
        include_citations=True,
    )
    assert len(results_with_citations) == 1
    assert results_with_citations[0]["metadata"]["citations"] == [
        "User cooks pasta every weekend."
    ]


@pytest.mark.asyncio
async def test_history_management(profile_storage):
    first = await profile_storage.add_history(
        "user-3",
        "First entry",
        metadata={"speaker": "tester"},
        isolations={"tenant": "A"},
    )
    second = await profile_storage.add_history(
        "user-3",
        "Second entry",
        metadata={"speaker": "tester"},
        isolations={"tenant": "B"},
    )

    rows = await profile_storage.get_history_messages_by_ingestion_status(
        "user-3", k=0, is_ingested=False
    )
    assert {row["id"] for row in rows} == {first["id"], second["id"]}
    assert all(
        json.loads(row["isolations"]) in ({"tenant": "A"}, {"tenant": "B"})
        for row in rows
    )

    count = await profile_storage.get_uningested_history_messages_count()
    assert count == 2

    await profile_storage.mark_messages_ingested([first["id"]])
    count_after_mark = await profile_storage.get_uningested_history_messages_count()
    assert count_after_mark == 1

    ingested_rows = await profile_storage.get_history_messages_by_ingestion_status(
        "user-3", k=10, is_ingested=True
    )
    assert [row["id"] for row in ingested_rows] == [first["id"]]

    messages = await profile_storage.get_history_message(
        "user-3", isolations={"tenant": "B"}
    )
    assert messages == ["Second entry"]

    await profile_storage.delete_history("user-3", isolations={"tenant": "A"})
    remaining = await profile_storage.get_history_messages_by_ingestion_status(
        "user-3", k=10, is_ingested=False
    )
    assert [row["id"] for row in remaining] == [second["id"]]

    future_time = int(time.time()) + 10
    await profile_storage.purge_history(
        "user-3", start_time=future_time, isolations={"tenant": "B"}
    )
    final_rows = await profile_storage.get_history_messages_by_ingestion_status(
        "user-3", k=10, is_ingested=False
    )
    assert final_rows == []
