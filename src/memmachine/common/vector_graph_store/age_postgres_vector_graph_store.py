"""
PostgreSQL with Apache AGE vector graph store implementation.

This module provides an asynchronous implementation
of a vector graph store using PostgreSQL with Apache AGE extension
for graph operations and pgvector for vector similarity search.

Note: This is a basic implementation. Apache AGE has limitations with
property types and vector operations compared to Neo4j.
"""

import asyncio
import json
import logging
from collections.abc import Iterable, Mapping
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.data_types import SimilarityMetric

from .data_types import (
    Edge,
    Node,
    OrderedPropertyValue,
    PropertyValue,
    demangle_embedding_name,
    demangle_property_name,
    is_mangled_embedding_name,
    is_mangled_property_name,
    mangle_embedding_name,
    mangle_property_name,
)
from .vector_graph_store import VectorGraphStore

logger = logging.getLogger(__name__)


class AgePostgresVectorGraphStoreParams(BaseModel):
    """
    Parameters for AgePostgresVectorGraphStore.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine instance.
        graph_name (str):
            Name of the Apache AGE graph to use.
            (default: "vector_graph").
        vector_dimensions (int):
            Dimensions for vector embeddings
            (default: 1536).
        create_indexes (bool):
            Whether to create indexes on UUID and properties
            (default: True).
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ...,
        description="Async SQLAlchemy engine instance",
    )
    graph_name: str = Field(
        "vector_graph",
        description="Name of the Apache AGE graph to use",
    )
    vector_dimensions: int = Field(
        1536,
        description="Dimensions for vector embeddings",
        gt=0,
    )
    create_indexes: bool = Field(
        True,
        description="Whether to create indexes on UUID and properties",
    )


class AgePostgresVectorGraphStore(VectorGraphStore):
    """Asynchronous PostgreSQL + Apache AGE implementation of VectorGraphStore."""

    def __init__(self, params: AgePostgresVectorGraphStoreParams) -> None:
        """Initialize the graph store with the provided parameters."""
        super().__init__()

        self._engine = params.engine
        self._graph_name = params.graph_name
        self._vector_dimensions = params.vector_dimensions
        self._create_indexes = params.create_indexes

        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Ensure Apache AGE extension and graph are initialized."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            async with self._engine.begin() as conn:
                # Enable Apache AGE extension
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS age"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                # Load AGE extension
                await conn.execute(text("LOAD 'age'"))
                await conn.execute(text("SET search_path = ag_catalog, '$user', public"))

                # Create graph if not exists
                await conn.execute(
                    text(f"SELECT create_graph('{self._graph_name}') WHERE NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = '{self._graph_name}')")
                )

                # Register vector type for asyncpg
                raw_conn = await conn.get_raw_connection()
                await register_vector(raw_conn.driver_connection)

            self._initialized = True

    async def add_nodes(
        self,
        collection: str,
        nodes: Iterable[Node],
    ) -> None:
        """Add nodes to a collection using Apache AGE."""
        await self._ensure_initialized()

        async with self._engine.begin() as conn:
            for node in nodes:
                properties = self._serialize_properties(node.properties)
                properties["uuid"] = str(node.uuid)
                properties["collection"] = collection

                # Add embeddings to properties
                for embedding_name, (embedding, similarity_metric) in node.embeddings.items():
                    mangled_name = mangle_embedding_name(embedding_name)
                    properties[mangled_name] = embedding
                    properties[f"{mangled_name}_metric"] = similarity_metric.value

                properties_json = json.dumps(properties)

                # Create vertex using Apache AGE
                query = text(
                    f"SELECT * FROM cypher('{self._graph_name}', $$ "
                    f"CREATE (n:{self._escape_label(collection)} {{props}}) "
                    "RETURN n "
                    "$$, :params) as (result agtype)"
                )

                await conn.execute(
                    query,
                    {"params": json.dumps({"props": properties})}
                )

    async def add_edges(
        self,
        relation: str,
        source_collection: str,
        target_collection: str,
        edges: Iterable[Edge],
    ) -> None:
        """Add edges between collections using Apache AGE."""
        await self._ensure_initialized()

        async with self._engine.begin() as conn:
            for edge in edges:
                properties = self._serialize_properties(edge.properties)
                properties["uuid"] = str(edge.uuid)

                # Add embeddings to properties
                for embedding_name, (embedding, similarity_metric) in edge.embeddings.items():
                    mangled_name = mangle_embedding_name(embedding_name)
                    properties[mangled_name] = embedding
                    properties[f"{mangled_name}_metric"] = similarity_metric.value

                # Create edge using Apache AGE
                query = text(
                    f"SELECT * FROM cypher('{self._graph_name}', $$ "
                    f"MATCH (source:{self._escape_label(source_collection)} {{uuid: $source_uuid}}), "
                    f"      (target:{self._escape_label(target_collection)} {{uuid: $target_uuid}}) "
                    f"CREATE (source)-[r:{self._escape_label(relation)} $props]->(target) "
                    "RETURN r "
                    "$$, :params) as (result agtype)"
                )

                await conn.execute(
                    query,
                    {
                        "params": json.dumps({
                            "source_uuid": str(edge.source_uuid),
                            "target_uuid": str(edge.target_uuid),
                            "props": properties,
                        })
                    }
                )

    async def search_similar_nodes(
        self,
        collection: str,
        embedding_name: str,
        query_embedding: list[float],
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        limit: int | None = 100,
        required_properties: Mapping[str, PropertyValue] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """Search nodes by vector similarity with optional property filters."""
        await self._ensure_initialized()

        if required_properties is None:
            required_properties = {}

        mangled_embedding_name = mangle_embedding_name(embedding_name)

        # Build property filter conditions
        where_clauses = []
        if required_properties:
            for prop_name, prop_value in required_properties.items():
                mangled_name = mangle_property_name(prop_name)
                if include_missing_properties:
                    where_clauses.append(
                        f"(n.{mangled_name} = '{self._escape_value(prop_value)}' OR n.{mangled_name} IS NULL)"
                    )
                else:
                    where_clauses.append(
                        f"n.{mangled_name} = '{self._escape_value(prop_value)}'"
                    )

        where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"

        # Use pgvector for similarity search
        distance_op = "<->" if similarity_metric == SimilarityMetric.COSINE else "<->"

        async with self._engine.begin() as conn:
            query = text(
                f"SELECT * FROM cypher('{self._graph_name}', $$ "
                f"MATCH (n:{self._escape_label(collection)}) "
                f"WHERE {where_clause} AND n.{mangled_embedding_name} IS NOT NULL "
                "RETURN n "
                f"ORDER BY n.{mangled_embedding_name} {distance_op} $embedding "
                f"{'LIMIT ' + str(limit) if limit else ''} "
                "$$, :params) as (result agtype)"
            )

            result = await conn.execute(
                query,
                {"params": json.dumps({"embedding": query_embedding})}
            )

            rows = result.fetchall()
            return [self._node_from_age_result(row[0]) for row in rows]

    async def search_related_nodes(
        self,
        relation: str,
        other_collection: str,
        this_collection: str,
        this_node_uuid: UUID,
        find_sources: bool = True,
        find_targets: bool = True,
        limit: int | None = None,
        required_edge_properties: Mapping[str, PropertyValue] | None = None,
        required_node_properties: Mapping[str, PropertyValue] | None = None,
        include_missing_edge_properties: bool = False,
        include_missing_node_properties: bool = False,
    ) -> list[Node]:
        """Search nodes connected by a relation with optional property filters."""
        await self._ensure_initialized()

        if required_edge_properties is None:
            required_edge_properties = {}
        if required_node_properties is None:
            required_node_properties = {}

        if not (find_sources or find_targets):
            return []

        # Build relationship pattern
        if find_sources and find_targets:
            rel_pattern = f"-[r:{self._escape_label(relation)}]-"
        elif find_sources:
            rel_pattern = f"<-[r:{self._escape_label(relation)}]-"
        else:
            rel_pattern = f"-[r:{self._escape_label(relation)}]->"

        # Build where clauses
        edge_where = self._build_where_clause(
            "r", required_edge_properties, include_missing_edge_properties
        )
        node_where = self._build_where_clause(
            "n", required_node_properties, include_missing_node_properties
        )

        async with self._engine.begin() as conn:
            query = text(
                f"SELECT * FROM cypher('{self._graph_name}', $$ "
                f"MATCH (m:{self._escape_label(this_collection)} {{uuid: $node_uuid}})"
                f"{rel_pattern}"
                f"(n:{self._escape_label(other_collection)}) "
                f"WHERE {edge_where} AND {node_where} "
                "RETURN DISTINCT n "
                f"{'LIMIT ' + str(limit) if limit else ''} "
                "$$, :params) as (result agtype)"
            )

            result = await conn.execute(
                query,
                {"params": json.dumps({"node_uuid": str(this_node_uuid)})}
            )

            rows = result.fetchall()
            return [self._node_from_age_result(row[0]) for row in rows]

    async def search_directional_nodes(
        self,
        collection: str,
        by_properties: Iterable[str],
        starting_at: Iterable[OrderedPropertyValue | None],
        order_ascending: Iterable[bool],
        include_equal_start: bool = False,
        limit: int | None = 1,
        required_properties: Mapping[str, PropertyValue] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """Find nodes ordered by property values in a chosen direction."""
        await self._ensure_initialized()

        if required_properties is None:
            required_properties = {}

        by_properties_list = list(by_properties)
        starting_at_list = list(starting_at)
        order_ascending_list = list(order_ascending)

        if not (len(by_properties_list) == len(starting_at_list) == len(order_ascending_list) > 0):
            raise ValueError(
                "Lengths of by_properties, starting_at, and order_ascending "
                "must be equal and greater than 0."
            )

        # Build comparison conditions
        comparisons = []
        for prop, start_val, ascending in zip(by_properties_list, starting_at_list, order_ascending_list):
            mangled_prop = mangle_property_name(prop)
            op = ">" if ascending else "<"
            if start_val is not None:
                comparisons.append(f"n.{mangled_prop} {op} '{self._escape_value(start_val)}'")
            else:
                comparisons.append(f"n.{mangled_prop} IS NOT NULL")

        where_clause = " OR ".join(comparisons)

        if include_equal_start:
            equal_conditions = [
                f"n.{mangle_property_name(prop)} = '{self._escape_value(start_val)}'"
                for prop, start_val in zip(by_properties_list, starting_at_list)
                if start_val is not None
            ]
            if equal_conditions:
                where_clause = f"({where_clause}) OR ({' AND '.join(equal_conditions)})"

        # Add required properties filter
        req_where = self._build_where_clause("n", required_properties, include_missing_properties)
        where_clause = f"({where_clause}) AND {req_where}"

        # Build ORDER BY clause
        order_by = ", ".join(
            f"n.{mangle_property_name(prop)} {'ASC' if asc else 'DESC'}"
            for prop, asc in zip(by_properties_list, order_ascending_list)
        )

        async with self._engine.begin() as conn:
            query = text(
                f"SELECT * FROM cypher('{self._graph_name}', $$ "
                f"MATCH (n:{self._escape_label(collection)}) "
                f"WHERE {where_clause} "
                f"RETURN n ORDER BY {order_by} "
                f"{'LIMIT ' + str(limit) if limit else ''} "
                "$$) as (result agtype)"
            )

            result = await conn.execute(query)
            rows = result.fetchall()
            return [self._node_from_age_result(row[0]) for row in rows]

    async def search_matching_nodes(
        self,
        collection: str,
        limit: int | None = None,
        required_properties: Mapping[str, PropertyValue] | None = None,
        include_missing_properties: bool = False,
    ) -> list[Node]:
        """Search nodes that match the provided property filters."""
        await self._ensure_initialized()

        if required_properties is None:
            required_properties = {}

        where_clause = self._build_where_clause("n", required_properties, include_missing_properties)

        async with self._engine.begin() as conn:
            query = text(
                f"SELECT * FROM cypher('{self._graph_name}', $$ "
                f"MATCH (n:{self._escape_label(collection)}) "
                f"WHERE {where_clause} "
                "RETURN n "
                f"{'LIMIT ' + str(limit) if limit else ''} "
                "$$) as (result agtype)"
            )

            result = await conn.execute(query)
            rows = result.fetchall()
            return [self._node_from_age_result(row[0]) for row in rows]

    async def get_nodes(
        self,
        collection: str,
        node_uuids: Iterable[UUID],
    ) -> list[Node]:
        """Retrieve nodes by uuid from a specific collection."""
        await self._ensure_initialized()

        uuid_list = [str(uuid) for uuid in node_uuids]

        async with self._engine.begin() as conn:
            query = text(
                f"SELECT * FROM cypher('{self._graph_name}', $$ "
                f"MATCH (n:{self._escape_label(collection)}) "
                "WHERE n.uuid IN $uuids "
                "RETURN n "
                "$$, :params) as (result agtype)"
            )

            result = await conn.execute(
                query,
                {"params": json.dumps({"uuids": uuid_list})}
            )

            rows = result.fetchall()
            return [self._node_from_age_result(row[0]) for row in rows]

    async def delete_nodes(
        self,
        collection: str,
        node_uuids: Iterable[UUID],
    ) -> None:
        """Delete nodes by uuid from a collection."""
        await self._ensure_initialized()

        uuid_list = [str(uuid) for uuid in node_uuids]

        async with self._engine.begin() as conn:
            query = text(
                f"SELECT * FROM cypher('{self._graph_name}', $$ "
                f"MATCH (n:{self._escape_label(collection)}) "
                "WHERE n.uuid IN $uuids "
                "DETACH DELETE n "
                "$$, :params) as (result agtype)"
            )

            await conn.execute(
                query,
                {"params": json.dumps({"uuids": uuid_list})}
            )

    async def delete_all_data(self) -> None:
        """Delete all nodes and relationships from the graph."""
        await self._ensure_initialized()

        async with self._engine.begin() as conn:
            query = text(
                f"SELECT * FROM cypher('{self._graph_name}', $$ "
                "MATCH (n) DETACH DELETE n "
                "$$) as (result agtype)"
            )
            await conn.execute(query)

    async def close(self) -> None:
        """Close the underlying database engine."""
        await self._engine.dispose()

    def _serialize_properties(self, properties: dict[str, PropertyValue]) -> dict[str, PropertyValue]:
        """Serialize properties for storage in Apache AGE."""
        serialized = {}
        for key, value in properties.items():
            mangled_key = mangle_property_name(key)
            if isinstance(value, datetime):
                serialized[mangled_key] = value.isoformat()
            elif isinstance(value, list):
                serialized[mangled_key] = value
            else:
                serialized[mangled_key] = value
        return serialized

    def _deserialize_properties(self, properties: dict) -> dict[str, PropertyValue]:
        """Deserialize properties from Apache AGE storage."""
        deserialized = {}
        for key, value in properties.items():
            if is_mangled_property_name(key):
                original_key = demangle_property_name(key)
                # Try to parse datetime strings
                if isinstance(value, str):
                    try:
                        deserialized[original_key] = datetime.fromisoformat(value)
                        continue
                    except (ValueError, AttributeError):
                        pass
                deserialized[original_key] = value
        return deserialized

    def _node_from_age_result(self, age_result: dict) -> Node:
        """Convert an Apache AGE result to a Node object."""
        # Parse AGE result (it comes as JSON)
        if isinstance(age_result, str):
            age_result = json.loads(age_result)

        uuid = UUID(age_result["uuid"])
        properties = {}
        embeddings = {}

        for key, value in age_result.items():
            if key in ("uuid", "collection"):
                continue

            if is_mangled_embedding_name(key):
                embedding_name = demangle_embedding_name(key)
                metric_key = f"{key}_metric"
                metric = SimilarityMetric(age_result.get(metric_key, SimilarityMetric.COSINE.value))
                embeddings[embedding_name] = (value, metric)
            elif is_mangled_property_name(key):
                original_key = demangle_property_name(key)
                properties[original_key] = value

        return Node(uuid=uuid, properties=properties, embeddings=embeddings)

    def _build_where_clause(
        self,
        alias: str,
        required_properties: Mapping[str, PropertyValue],
        include_missing: bool,
    ) -> str:
        """Build a WHERE clause for property filtering."""
        if not required_properties:
            return "TRUE"

        conditions = []
        for prop_name, prop_value in required_properties.items():
            mangled_name = mangle_property_name(prop_name)
            if include_missing:
                conditions.append(
                    f"({alias}.{mangled_name} = '{self._escape_value(prop_value)}' "
                    f"OR {alias}.{mangled_name} IS NULL)"
                )
            else:
                conditions.append(
                    f"{alias}.{mangled_name} = '{self._escape_value(prop_value)}'"
                )

        return " AND ".join(conditions)

    @staticmethod
    def _escape_label(label: str) -> str:
        """Escape a label for use in Cypher queries."""
        # Remove special characters and spaces
        return label.replace(" ", "_").replace("-", "_").replace(".", "_")

    @staticmethod
    def _escape_value(value: PropertyValue) -> str:
        """Escape a value for use in Cypher queries."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        # Escape single quotes in strings
        return str(value).replace("'", "\\'")
