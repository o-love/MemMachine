"""Fixtures for testing vector graph store implementations."""

import asyncio

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

from memmachine.common.vector_graph_store.age_postgres_vector_graph_store import (
    AgePostgresVectorGraphStore,
    AgePostgresVectorGraphStoreParams,
)
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)


@pytest.fixture(scope="module")
def neo4j_connection_info():
    """Provide Neo4j connection information via testcontainer."""
    neo4j_username = "neo4j"
    neo4j_password = "password"

    with Neo4jContainer(
        image="neo4j:latest",
        username=neo4j_username,
        password=neo4j_password,
    ) as neo4j:
        yield {
            "uri": neo4j.get_connection_url(),
            "username": neo4j_username,
            "password": neo4j_password,
        }


@pytest_asyncio.fixture(scope="module")
async def neo4j_driver(neo4j_connection_info):
    """Create async Neo4j driver."""
    driver = AsyncGraphDatabase.driver(
        neo4j_connection_info["uri"],
        auth=(
            neo4j_connection_info["username"],
            neo4j_connection_info["password"],
        ),
    )
    yield driver
    await driver.close()


@pytest.fixture(scope="module")
def postgres_age_connection_info():
    """Provide PostgreSQL with Apache AGE connection information via testcontainer."""
    with PostgresContainer(
        image="apache/age:latest",
        username="postgres",
        password="password",
    ) as postgres:
        yield {
            "host": postgres.get_container_host_ip(),
            "port": int(postgres.get_exposed_port(5432)),
            "database": postgres.dbname,
            "username": postgres.username,
            "password": postgres.password,
        }


@pytest_asyncio.fixture(scope="module")
async def postgres_age_engine(postgres_age_connection_info):
    """Create async PostgreSQL engine for Apache AGE."""
    engine = create_async_engine(
        URL.create(
            "postgresql+asyncpg",
            username=postgres_age_connection_info["username"],
            password=postgres_age_connection_info["password"],
            host=postgres_age_connection_info["host"],
            port=postgres_age_connection_info["port"],
            database=postgres_age_connection_info["database"],
        ),
    )
    yield engine
    await engine.dispose()


@pytest.fixture
def neo4j_vector_graph_store(neo4j_driver):
    """Create Neo4j vector graph store with exact similarity search."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=True,
        ),
    )


@pytest.fixture
def neo4j_vector_graph_store_ann(neo4j_driver):
    """Create Neo4j vector graph store with ANN similarity search."""
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=neo4j_driver,
            force_exact_similarity_search=False,
            filtered_similarity_search_fudge_factor=2,
            exact_similarity_search_fallback_threshold=0.5,
            range_index_hierarchies=[["group", "session"]],
            range_index_creation_threshold=0,
            vector_index_creation_threshold=0,
        ),
    )


@pytest.fixture
def age_postgres_vector_graph_store(postgres_age_engine):
    """Create Apache AGE PostgreSQL vector graph store."""
    return AgePostgresVectorGraphStore(
        AgePostgresVectorGraphStoreParams(
            engine=postgres_age_engine,
            graph_name="test_graph",
            vector_dimensions=3,
            create_indexes=True,
        ),
    )


@pytest.fixture(
    params=[
        pytest.param("neo4j_vector_graph_store", marks=pytest.mark.integration),
        pytest.param("age_postgres_vector_graph_store", marks=pytest.mark.integration),
    ],
)
def vector_graph_store(request):
    """Parametrized fixture for all vector graph store implementations."""
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        pytest.param("neo4j_vector_graph_store_ann", marks=pytest.mark.integration),
        # AGE doesn't have separate ANN implementation, uses same store
        pytest.param("age_postgres_vector_graph_store", marks=pytest.mark.integration),
    ],
)
def vector_graph_store_ann(request):
    """Parametrized fixture for ANN-enabled vector graph store implementations."""
    return request.getfixturevalue(request.param)


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup_neo4j(neo4j_driver):
    """Clean up Neo4j database before each test."""
    # Delete all nodes and relationships
    await neo4j_driver.execute_query("MATCH (n) DETACH DELETE n")

    # Drop all constraints
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW CONSTRAINTS YIELD name RETURN name",
    )
    drop_constraint_tasks = [
        neo4j_driver.execute_query(f"DROP CONSTRAINT {record['name']} IF EXISTS")
        for record in records
    ]

    # Drop all range indexes
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW RANGE INDEXES YIELD name RETURN name",
    )
    drop_range_index_tasks = [
        neo4j_driver.execute_query(f"DROP INDEX {record['name']} IF EXISTS")
        for record in records
    ]

    # Drop all vector indexes
    records, _, _ = await neo4j_driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name RETURN name",
    )
    drop_vector_index_tasks = [
        neo4j_driver.execute_query(f"DROP INDEX {record['name']} IF EXISTS")
        for record in records
    ]

    await asyncio.gather(*drop_constraint_tasks)
    await asyncio.gather(*drop_range_index_tasks)
    await asyncio.gather(*drop_vector_index_tasks)
    yield


@pytest_asyncio.fixture(autouse=True)
async def db_cleanup_age(postgres_age_engine):
    """Clean up Apache AGE database before each test."""
    # Drop and recreate the graph to ensure clean state
    try:
        async with postgres_age_engine.begin() as conn:
            await conn.execute(text("LOAD 'age'"))
            await conn.execute(text("SET search_path = ag_catalog, '$user', public"))
            try:
                await conn.execute(text("SELECT drop_graph('test_graph', true)"))
            except Exception:
                pass  # Graph might not exist yet

            await conn.execute(
                text("SELECT create_graph('test_graph')")
            )
    except Exception as e:
        # AGE might not be installed - skip cleanup for AGE tests
        pass
    yield
