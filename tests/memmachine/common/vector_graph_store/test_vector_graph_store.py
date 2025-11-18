"""
General tests for all VectorGraphStore implementations.

Tests are parametrized to run against all implementations:
- Neo4j
- Apache AGE with PostgreSQL
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from memmachine.common.data_types import SimilarityMetric
from memmachine.common.vector_graph_store.data_types import Edge, Node
from memmachine.common.vector_graph_store.vector_graph_store import VectorGraphStore

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_add_nodes(vector_graph_store: VectorGraphStore):
    """Test adding nodes to a collection."""
    # Empty list should work
    nodes = []
    await vector_graph_store.add_nodes("Entity", nodes)

    # Add multiple nodes with various properties
    nodes = [
        Node(
            uuid=uuid4(),
            properties={"name": "Node1"},
        ),
        Node(
            uuid=uuid4(),
            properties={"name": "Node2"},
        ),
        Node(
            uuid=uuid4(),
            properties={"name": "Node3", "time": datetime.now(tz=UTC)},
            embeddings={
                "embedding_name": (
                    [0.1, 0.2, 0.3],
                    SimilarityMetric.COSINE,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes("Entity", nodes)

    # Verify nodes were added
    fetched_nodes = await vector_graph_store.get_nodes(
        "Entity",
        [node.uuid for node in nodes],
    )
    assert len(fetched_nodes) == len(nodes)


@pytest.mark.asyncio
async def test_add_edges(vector_graph_store: VectorGraphStore):
    """Test adding edges between nodes."""
    node1_uuid = uuid4()
    node2_uuid = uuid4()
    node3_uuid = uuid4()

    nodes = [
        Node(
            uuid=node1_uuid,
            properties={"name": "Node1"},
        ),
        Node(
            uuid=node2_uuid,
            properties={"name": "Node2"},
        ),
        Node(
            uuid=node3_uuid,
            properties={"name": "Node3", "time": datetime.now(tz=UTC)},
            embeddings={
                "embedding_name": (
                    [0.1, 0.2, 0.3],
                    SimilarityMetric.COSINE,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes("Entity", nodes)

    # Empty edges list should work
    edges = []
    await vector_graph_store.add_edges("RELATED_TO", "Entity", "Entity", edges)

    # Add edges with various properties
    related_to_edges = [
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node2_uuid,
            properties={"description": "Node1 to Node2", "time": datetime.now(tz=UTC)},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node1_uuid,
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node3_uuid,
            properties={"description": "Node1 to Node3"},
            embeddings={
                "embedding_name": (
                    [0.4, 0.5, 0.6],
                    SimilarityMetric.DOT,
                ),
            },
        ),
    ]

    is_edges = [
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node1_uuid,
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node2_uuid,
            properties={"description": "Node2 loop"},
        ),
    ]

    await vector_graph_store.add_edges(
        "RELATED_TO",
        "Entity",
        "Entity",
        related_to_edges,
    )
    await vector_graph_store.add_edges("IS", "Entity", "Entity", is_edges)

    # Verify edges were created by searching for related nodes
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node1_uuid,
    )
    assert len(results) >= 2


@pytest.mark.asyncio
async def test_search_similar_nodes(
    vector_graph_store: VectorGraphStore,
    vector_graph_store_ann: VectorGraphStore,
):
    """Test vector similarity search for nodes."""
    nodes = [
        Node(
            uuid=uuid4(),
            properties={
                "name": "Node1",
            },
            embeddings={
                "embedding1": (
                    [1000.0, 0.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [1000.0, 0.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Node2",
                "include?": "yes",
            },
            embeddings={
                "embedding1": (
                    [10.0, 10.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [10.0, 10.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Node3",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, 0.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, 0.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Node4",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -1.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -1.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Node5",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -2.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -2.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Node6",
                "include?": "no",
            },
            embeddings={
                "embedding1": (
                    [-100.0, -3.0],
                    SimilarityMetric.COSINE,
                ),
                "embedding2": (
                    [-100.0, -3.0],
                    SimilarityMetric.EUCLIDEAN,
                ),
            },
        ),
    ]

    await vector_graph_store.add_nodes("Entity", nodes)

    # Test ANN search
    results = await vector_graph_store_ann.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert 0 < len(results) <= 5

    # Test exact search
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node1"

    # Test with property filtering
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        required_properties={"include?": "yes"},
        include_missing_properties=False,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    # Test with include_missing_properties
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding1",
        similarity_metric=SimilarityMetric.COSINE,
        limit=5,
        required_properties={"include?": "yes"},
        include_missing_properties=True,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Node1"

    # Test Euclidean distance
    results = await vector_graph_store.search_similar_nodes(
        collection="Entity",
        query_embedding=[1.0, 0.0],
        embedding_name="embedding2",
        similarity_metric=SimilarityMetric.EUCLIDEAN,
        limit=5,
    )
    assert len(results) == 5
    assert results[0].properties["name"] == "Node2"


@pytest.mark.asyncio
async def test_search_related_nodes(vector_graph_store: VectorGraphStore):
    """Test searching for nodes related via edges."""
    node1_uuid = uuid4()
    node2_uuid = uuid4()
    node3_uuid = uuid4()
    node4_uuid = uuid4()

    nodes = [
        Node(
            uuid=node1_uuid,
            properties={"name": "Node1"},
        ),
        Node(
            uuid=node2_uuid,
            properties={"name": "Node2", "extra!": "something"},
        ),
        Node(
            uuid=node3_uuid,
            properties={"name": "Node3", "marker?": "A"},
        ),
        Node(
            uuid=node4_uuid,
            properties={"name": "Node4", "marker?": "B"},
        ),
    ]

    related_to_edges = [
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node2_uuid,
            properties={"description": "Node1 to Node2"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node1_uuid,
            properties={"description": "Node2 to Node1"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node3_uuid,
            target_uuid=node2_uuid,
            properties={
                "description": "Node3 to Node2",
                "extra": 1,
            },
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node3_uuid,
            target_uuid=node4_uuid,
            properties={
                "description": "Node3 to Node4",
                "extra": 2,
            },
        ),
    ]

    is_edges = [
        Edge(
            uuid=uuid4(),
            source_uuid=node1_uuid,
            target_uuid=node1_uuid,
            properties={"description": "Node1 loop"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node2_uuid,
            target_uuid=node2_uuid,
            properties={"description": "Node2 loop"},
        ),
        Edge(
            uuid=uuid4(),
            source_uuid=node3_uuid,
            target_uuid=node3_uuid,
            properties={"description": "Node3 loop"},
        ),
    ]

    await vector_graph_store.add_nodes("Entity", nodes)
    await vector_graph_store.add_edges(
        "RELATED_TO",
        "Entity",
        "Entity",
        related_to_edges,
    )
    await vector_graph_store.add_edges("RELATED_TO", "Entity", "Entity", is_edges)

    # Test bidirectional search
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node1_uuid,
    )
    assert len(results) == 2
    assert results[0].properties["name"] != results[1].properties["name"]
    assert results[0].properties["name"] in ("Node1", "Node2")
    assert results[1].properties["name"] in ("Node1", "Node2")

    # Test with node property filtering
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node1_uuid,
        required_node_properties={"extra!": "something"},
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node2"

    # Test find_sources=False
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node2_uuid,
        find_sources=False,
    )
    assert len(results) == 2

    # Test find_targets=False
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node3_uuid,
        find_targets=False,
    )
    assert len(results) == 1
    assert results[0].properties["name"] == "Node3"

    # Test with include_missing_node_properties
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node3_uuid,
        required_node_properties={"marker?": "A"},
        include_missing_node_properties=True,
    )
    assert len(results) == 2

    # Test with edge property filtering
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node3_uuid,
        required_edge_properties={"extra": 1},
    )
    assert len(results) == 1

    # Test with include_missing_edge_properties
    results = await vector_graph_store.search_related_nodes(
        relation="RELATED_TO",
        other_collection="Entity",
        this_collection="Entity",
        this_node_uuid=node3_uuid,
        required_edge_properties={"extra": 1},
        include_missing_edge_properties=True,
    )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_directional_nodes(vector_graph_store: VectorGraphStore):
    """Test searching nodes ordered by property values."""
    time = datetime.now(tz=UTC)
    delta = timedelta(days=1)

    nodes = [
        Node(
            uuid=uuid4(),
            properties={
                "name": "Event1",
                "timestamp": time,
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Event2",
                "timestamp": time + delta,
                "include?": "yes",
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Event3",
                "timestamp": time + 2 * delta,
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Event4",
                "timestamp": time + 3 * delta,
                "include?": "yes",
            },
        ),
    ]

    await vector_graph_store.add_nodes("Event", nodes)

    # Test ascending with include_equal_start
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event3"

    # Test with required properties
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=True,
        limit=2,
        required_properties={"include?": "yes"},
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event4"

    # Test descending
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[False],
        include_equal_start=True,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event2"
    assert results[1].properties["name"] == "Event1"

    # Test without include_equal_start
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[time + delta],
        order_ascending=[True],
        include_equal_start=False,
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event3"
    assert results[1].properties["name"] == "Event4"

    # Test with None starting_at
    results = await vector_graph_store.search_directional_nodes(
        collection="Event",
        by_properties=["timestamp"],
        starting_at=[None],
        order_ascending=[False],
        limit=2,
    )
    assert len(results) == 2
    assert results[0].properties["name"] == "Event4"
    assert results[1].properties["name"] == "Event3"


@pytest.mark.asyncio
async def test_search_matching_nodes(vector_graph_store: VectorGraphStore):
    """Test searching nodes by property values."""
    person_nodes = [
        Node(
            uuid=uuid4(),
            properties={
                "name": "Alice",
                "age!with$pecialchars": 30,
                "city": "San Francisco",
                "title": "Engineer",
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Bob",
                "age!with$pecialchars": 25,
                "city": "Los Angeles",
                "title": "Designer",
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "Charlie",
                "city": "New York",
            },
        ),
        Node(
            uuid=uuid4(),
            properties={
                "name": "David",
                "age!with$pecialchars": 30,
                "city": "New York",
            },
        ),
    ]

    robot_nodes = [
        Node(
            uuid=uuid4(),
            properties={"name": "Eve", "city": "Axiom"},
        ),
    ]

    await vector_graph_store.add_nodes("Person", person_nodes)
    await vector_graph_store.add_nodes("Robot", robot_nodes)

    # Test without filtering
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
    )
    assert len(results) == 4

    results = await vector_graph_store.search_matching_nodes(
        collection="Robot",
    )
    assert len(results) == 1

    # Test with single property filter
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        required_properties={
            "city": "New York",
        },
    )
    assert len(results) == 2

    # Test with multiple property filters (no match)
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        required_properties={
            "city": "San Francisco",
            "age!with$pecialchars": 20,
        },
    )
    assert len(results) == 0

    # Test with multiple property filters (with match)
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        required_properties={
            "city": "New York",
            "age!with$pecialchars": 30,
        },
    )
    assert len(results) == 1

    # Test with special characters in property name
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        required_properties={
            "age!with$pecialchars": 30,
        },
    )
    assert len(results) == 2

    # Test with include_missing_properties
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        required_properties={
            "age!with$pecialchars": 30,
        },
        include_missing_properties=True,
    )
    assert len(results) == 3

    # Test with specific title
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        required_properties={
            "title": "Engineer",
        },
    )
    assert len(results) == 1

    # Test with include_missing_properties for title
    results = await vector_graph_store.search_matching_nodes(
        collection="Person",
        required_properties={
            "title": "Engineer",
        },
        include_missing_properties=True,
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_nodes(vector_graph_store: VectorGraphStore):
    """Test retrieving nodes by UUID."""
    nodes = [
        Node(
            uuid=uuid4(),
            properties={"name": "Node1", "time": datetime.now(tz=UTC)},
        ),
        Node(
            uuid=uuid4(),
            properties={"name": "Node2"},
        ),
        Node(
            uuid=uuid4(),
            properties={"name": "Node3"},
        ),
    ]

    await vector_graph_store.add_nodes("Entity", nodes)

    # Get all nodes
    fetched_nodes = await vector_graph_store.get_nodes(
        "Entity",
        [node.uuid for node in nodes],
    )
    assert len(fetched_nodes) == 3

    for fetched_node in fetched_nodes:
        assert fetched_node.uuid in {node.uuid for node in nodes}

    # Get subset with non-existent UUID
    fetched_nodes = await vector_graph_store.get_nodes(
        "Entity",
        [nodes[0].uuid, uuid4()],
    )
    assert len(fetched_nodes) == 1
    assert fetched_nodes[0] == nodes[0]


@pytest.mark.asyncio
async def test_delete_nodes(vector_graph_store: VectorGraphStore):
    """Test deleting nodes."""
    nodes = [
        Node(uuid=uuid4()) for _ in range(6)
    ]

    await vector_graph_store.add_nodes("Entity", nodes)

    # Verify all nodes exist
    fetched = await vector_graph_store.get_nodes(
        "Entity",
        [node.uuid for node in nodes],
    )
    assert len(fetched) == 6

    # Delete from wrong collection (should not delete anything)
    await vector_graph_store.delete_nodes("Bad", [node.uuid for node in nodes[:-3]])
    fetched = await vector_graph_store.get_nodes(
        "Entity",
        [node.uuid for node in nodes],
    )
    assert len(fetched) == 6

    # Delete from correct collection
    await vector_graph_store.delete_nodes("Entity", [node.uuid for node in nodes[:-3]])
    fetched = await vector_graph_store.get_nodes(
        "Entity",
        [node.uuid for node in nodes],
    )
    assert len(fetched) == 3


@pytest.mark.asyncio
async def test_delete_all_data(vector_graph_store: VectorGraphStore):
    """Test deleting all data from the store."""
    nodes = [
        Node(uuid=uuid4()) for _ in range(6)
    ]

    await vector_graph_store.add_nodes("Entity", nodes)

    # Verify nodes exist
    fetched = await vector_graph_store.get_nodes(
        "Entity",
        [node.uuid for node in nodes],
    )
    assert len(fetched) == 6

    # Delete all data
    await vector_graph_store.delete_all_data()

    # Verify all nodes are gone
    fetched = await vector_graph_store.get_nodes(
        "Entity",
        [node.uuid for node in nodes],
    )
    assert len(fetched) == 0
