import json
import logging
import time
from typing import Any

import numpy as np
from neo4j import AsyncDriver
from pydantic import BaseModel, InstanceOf

from memmachine.profile_memory.storage.storage_base import ProfileStorageBase

logger = logging.getLogger(__name__)


class Neo4jProfileStorage(ProfileStorageBase):
    """
    Neo4j implementation of ``ProfileStorageBase``.

    Profile entries are stored as ``ProfileEntry`` nodes with properties:
    ``profile_id`` (int), ``user_id`` (str), ``tag`` (str), ``feature`` (str),
    ``value`` (str), ``embedding`` (list[float]), ``metadata_json`` (str),
    prefixed isolation keys (``__iso__<key>``) and ``isolations_json`` (str).

    History entries are represented as ``HistoryEntry`` nodes with analogous
    properties. Citations are stored as ``(:ProfileEntry)-[:CITED]->(:HistoryEntry)``
    relationships.
    """

    class Params(BaseModel):
        driver: InstanceOf[AsyncDriver]
        database: str = ""

    def __init__(self, params: Params):
        self._driver = params.driver
        self._database = params.database

        self._isolation_prefix = "__iso__"
        self._iso_prefix_param = {"iso_prefix": self._isolation_prefix}

    @staticmethod
    def _normalize_isolations(
        isolations: dict[str, bool | int | float | str] | None,
    ) -> dict[str, bool | int | float | str]:
        if isolations is None:
            return {}
        return dict(isolations)

    def _build_isolation_properties(
        self, isolations: dict[str, bool | int | float | str]
    ) -> dict[str, bool | int | float | str]:
        return {
            f"{self._isolation_prefix}{key}": value for key, value in isolations.items()
        }

    def _isolation_params(
        self, isolations: dict[str, bool | int | float | str]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"isolations": isolations}
        params.update(self._iso_prefix_param)
        return params

    async def startup(self):
        await self._ensure_constraints()

    async def cleanup(self):
        if self._driver is None:
            return
        await self._driver.close()
        self._driver = None

    async def delete_all(self):
        await self._execute_query(
            """
            MATCH (p:ProfileEntry)
            DETACH DELETE p
            """
        )
        await self._execute_query(
            """
            MATCH (h:HistoryEntry)
            DETACH DELETE h
            """
        )
        await self._execute_query(
            """
            MATCH (s:Sequence)
            DELETE s
            """
        )

    async def get_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, dict[str, Any | list[Any]]]:
        isolations_map = self._normalize_isolations(isolations)
        params = {"user_id": user_id}
        params.update(self._isolation_params(isolations_map))
        records = await self._execute_query(
            """
            MATCH (p:ProfileEntry {user_id: $user_id})
            WHERE ALL(key IN keys($isolations) WHERE
                p[$iso_prefix + key] IS NOT NULL AND p[$iso_prefix + key] = $isolations[key])
            RETURN p.tag AS tag, p.feature AS feature, p.value AS value
            """,
            **params,
        )

        profile: dict[str, dict[str, Any | list[Any]]] = {}
        for record in records:
            tag = record["tag"]
            feature = record["feature"]
            value = {"value": record["value"]}
            tag_bucket = profile.setdefault(tag, {})
            existing = tag_bucket.setdefault(feature, [])
            assert isinstance(existing, list)
            existing.append(value)

        for tag, features in profile.items():
            for feature, values in list(features.items()):
                if isinstance(values, list) and len(values) == 1:
                    features[feature] = values[0]
        return profile

    async def get_citation_list(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[int]:
        isolations_map = self._normalize_isolations(isolations)
        params = {
            "user_id": user_id,
            "feature": feature,
            "value": str(value),
            "tag": tag,
        }
        params.update(self._isolation_params(isolations_map))
        records = await self._execute_query(
            """
            MATCH (p:ProfileEntry {user_id: $user_id, feature: $feature, value: $value, tag: $tag})
            WHERE ALL(key IN keys($isolations) WHERE
                p[$iso_prefix + key] IS NOT NULL AND p[$iso_prefix + key] = $isolations[key])
            MATCH (p)-[:CITED]->(h:HistoryEntry)
            RETURN DISTINCT h.history_id AS citation_id
            """,
            **params,
        )
        return [record["citation_id"] for record in records]

    async def delete_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations_map = self._normalize_isolations(isolations)
        params = {"user_id": user_id}
        params.update(self._isolation_params(isolations_map))
        await self._execute_query(
            """
            MATCH (p:ProfileEntry {user_id: $user_id})
            WHERE ALL(key IN keys($isolations) WHERE
                p[$iso_prefix + key] IS NOT NULL AND p[$iso_prefix + key] = $isolations[key])
            DETACH DELETE p
            """,
            **params,
        )

    async def add_profile_feature(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
        citations: list[int] | None = None,
    ):
        metadata = metadata or {}
        isolations_map = self._normalize_isolations(isolations)
        isolation_properties = self._build_isolation_properties(isolations_map)
        citations = citations or []

        embedding_list = [float(x) for x in np.asarray(embedding).flatten()]
        metadata_json = json.dumps(metadata)
        isolations_json = json.dumps(isolations_map)

        records = await self._execute_query(
            """
            CALL {
                MERGE (counter:Sequence {name: 'ProfileEntry'})
                ON CREATE SET counter.value = 1
                ON MATCH SET counter.value = counter.value + 1
                RETURN counter.value AS profile_id
            }
            CREATE (p:ProfileEntry {
                profile_id: profile_id,
                user_id: $user_id,
                feature: $feature,
                value: $value,
                tag: $tag,
                embedding: $embedding,
                metadata_json: $metadata_json,
                isolations_json: $isolations_json,
                created_at: datetime()
            })
            SET p += $isolation_properties
            RETURN profile_id
            """,
            user_id=user_id,
            feature=feature,
            value=str(value),
            tag=tag,
            embedding=embedding_list,
            metadata_json=metadata_json,
            isolations_json=isolations_json,
            isolation_properties=isolation_properties,
        )

        profile_id = records[0]["profile_id"]

        if citations:
            await self._execute_query(
                """
                MATCH (p:ProfileEntry {profile_id: $profile_id})
                UNWIND $citations AS citation_id
                MATCH (h:HistoryEntry {history_id: citation_id})
                MERGE (p)-[:CITED]->(h)
                """,
                profile_id=profile_id,
                citations=[int(c) for c in citations],
            )

    async def semantic_search(
        self,
        user_id: str,
        qemb: np.ndarray,
        k: int,
        min_cos: float,
        isolations: dict[str, bool | int | float | str] | None = None,
        include_citations: bool = False,
    ) -> list[dict[str, Any]]:
        isolations_map = self._normalize_isolations(isolations)
        params = {"user_id": user_id}
        params.update(self._isolation_params(isolations_map))
        records = await self._execute_query(
            """
            MATCH (p:ProfileEntry {user_id: $user_id})
            WHERE ALL(key IN keys($isolations) WHERE
                p[$iso_prefix + key] IS NOT NULL AND p[$iso_prefix + key] = $isolations[key])
            OPTIONAL MATCH (p)-[:CITED]->(h:HistoryEntry)
            WITH p, collect(h.content) AS citations
            RETURN p.profile_id AS profile_id,
                   p.tag AS tag,
                   p.feature AS feature,
                   p.value AS value,
                   p.embedding AS embedding,
                   citations
            """,
            **params,
        )

        qemb_array = np.asarray(qemb).flatten().astype(float)
        qnorm = float(np.linalg.norm(qemb_array))
        if qnorm == 0:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            embedding = np.array(record["embedding"], dtype=float)
            denom = float(np.linalg.norm(embedding)) * qnorm
            if denom == 0:
                continue
            score = float(np.dot(embedding, qemb_array) / denom)
            if score <= min_cos:
                continue
            citations = [
                citation
                for citation in (record["citations"] or [])
                if citation is not None
            ]
            metadata: dict[str, Any] = {
                "id": record["profile_id"],
                "similarity_score": score,
            }
            if include_citations:
                metadata["citations"] = citations
            scored.append(
                (
                    score,
                    {
                        "tag": record["tag"],
                        "feature": record["feature"],
                        "value": record["value"],
                        "metadata": metadata,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        if k > 0:
            scored = scored[:k]
        return [item[1] for item in scored]

    async def delete_profile_feature_by_id(self, pid: int):
        await self._execute_query(
            """
            MATCH (p:ProfileEntry {profile_id: $profile_id})
            DETACH DELETE p
            """,
            profile_id=int(pid),
        )

    async def get_all_citations_for_ids(
        self, pids: list[int]
    ) -> list[tuple[int, dict[str, bool | int | float | str]]]:
        if not pids:
            return []

        records = await self._execute_query(
            """
            MATCH (p:ProfileEntry)-[:CITED]->(h:HistoryEntry)
            WHERE p.profile_id IN $profile_ids
            RETURN DISTINCT h.history_id AS history_id,
                            h.isolations_json AS isolations_json
            """,
            profile_ids=[int(pid) for pid in pids],
        )

        results: list[tuple[int, dict[str, bool | int | float | str]]] = []
        for record in records:
            isolations_json = record["isolations_json"] or "{}"
            results.append((record["history_id"], json.loads(isolations_json)))
        return results

    async def delete_profile_feature(
        self,
        user_id: str,
        feature: str,
        tag: str,
        value: str | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations_map = self._normalize_isolations(isolations)
        query = [
            "MATCH (p:ProfileEntry {user_id: $user_id, feature: $feature, tag: $tag})",
            "WHERE ALL(key IN keys($isolations) WHERE",
            "    p[$iso_prefix + key] IS NOT NULL AND p[$iso_prefix + key] = $isolations[key])",
        ]
        parameters: dict[str, Any] = {
            "user_id": user_id,
            "feature": feature,
            "tag": tag,
        }
        parameters.update(self._isolation_params(isolations_map))
        if value is not None:
            query.append("AND p.value = $value")
            parameters["value"] = str(value)
        query.append("DETACH DELETE p")

        await self._execute_query("\n".join(query), **parameters)

    async def get_large_profile_sections(
        self,
        user_id: str,
        thresh: int,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        isolations_map = self._normalize_isolations(isolations)
        params = {"user_id": user_id, "thresh": thresh}
        params.update(self._isolation_params(isolations_map))
        records = await self._execute_query(
            """
            MATCH (p:ProfileEntry {user_id: $user_id})
            WHERE ALL(key IN keys($isolations) WHERE
                p[$iso_prefix + key] IS NOT NULL AND p[$iso_prefix + key] = $isolations[key])
            WITH p.tag AS tag, collect(p) AS entries
            WHERE size(entries) >= $thresh
            RETURN [entry IN entries | {
                tag: entry.tag,
                feature: entry.feature,
                value: entry.value,
                metadata: {id: entry.profile_id}
            }] AS section
            """,
            **params,
        )
        return [record["section"] for record in records]

    async def add_history(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, Any]:
        metadata_dict = metadata or {}
        isolations_map = self._normalize_isolations(isolations)
        isolation_properties = self._build_isolation_properties(isolations_map)

        metadata_json = json.dumps(metadata_dict)
        isolations_json = json.dumps(isolations_map)
        timestamp = float(time.time())

        records = await self._execute_query(
            """
            CALL {
                MERGE (counter:Sequence {name: 'HistoryEntry'})
                ON CREATE SET counter.value = 1
                ON MATCH SET counter.value = counter.value + 1
                RETURN counter.value AS history_id
            }
            CREATE (h:HistoryEntry {
                history_id: history_id,
                user_id: $user_id,
                content: $content,
                metadata_json: $metadata_json,
                isolations_json: $isolations_json,
                timestamp: $timestamp,
                ingested: false
            })
            SET h += $isolation_properties
            RETURN history_id
            """,
            user_id=user_id,
            content=content,
            metadata_json=metadata_json,
            isolations_json=isolations_json,
            timestamp=timestamp,
            isolation_properties=isolation_properties,
        )

        history_id = records[0]["history_id"]
        return {
            "id": history_id,
            "user_id": user_id,
            "content": content,
            "metadata": metadata_json,
            "isolations": isolations_json,
        }

    async def delete_history(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations_map = self._normalize_isolations(isolations)
        lower = float(start_time) if start_time else float("-inf")
        upper = float(end_time) if end_time else float("inf")
        params = {
            "user_id": user_id,
            "lower": lower,
            "upper": upper,
        }
        params.update(self._isolation_params(isolations_map))

        await self._execute_query(
            """
            MATCH (h:HistoryEntry {user_id: $user_id})
            WHERE ALL(key IN keys($isolations) WHERE
                h[$iso_prefix + key] IS NOT NULL AND h[$iso_prefix + key] = $isolations[key])
              AND h.timestamp >= $lower
              AND h.timestamp <= $upper
            DETACH DELETE h
            """,
            **params,
        )

    async def get_history_messages_by_ingestion_status(
        self,
        user_id: str,
        k: int = 0,
        is_ingested: bool = False,
    ) -> list[dict[str, Any]]:
        limit_clause = "LIMIT $limit" if k > 0 else ""
        records = await self._execute_query(
            f"""
            MATCH (h:HistoryEntry {{user_id: $user_id, ingested: $is_ingested}})
            RETURN h.history_id AS history_id,
                   h.user_id AS user_id,
                   h.content AS content,
                   h.metadata_json AS metadata_json,
                   h.isolations_json AS isolations_json,
                   h.timestamp AS timestamp
            ORDER BY h.timestamp DESC
            {limit_clause}
            """,
            user_id=user_id,
            is_ingested=is_ingested,
            limit=int(k),
        )

        return [
            {
                "id": record["history_id"],
                "user_id": record["user_id"],
                "content": record["content"],
                "metadata": record["metadata_json"],
                "isolations": record["isolations_json"],
            }
            for record in records
        ]

    async def get_uningested_history_messages_count(self) -> int:
        records = await self._execute_query(
            """
            MATCH (h:HistoryEntry {ingested: false})
            RETURN count(h) AS count
            """
        )
        return records[0]["count"]

    async def mark_messages_ingested(
        self,
        ids: list[int],
    ) -> None:
        if not ids:
            return
        await self._execute_query(
            """
            MATCH (h:HistoryEntry)
            WHERE h.history_id IN $ids
            SET h.ingested = true
            """,
            ids=[int(i) for i in ids],
        )

    async def get_history_message(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[str]:
        isolations_map = self._normalize_isolations(isolations)
        lower = float(start_time) if start_time else float("-inf")
        upper = float(end_time) if end_time else float("inf")
        params = {
            "user_id": user_id,
            "lower": lower,
            "upper": upper,
        }
        params.update(self._isolation_params(isolations_map))

        records = await self._execute_query(
            """
            MATCH (h:HistoryEntry {user_id: $user_id})
            WHERE ALL(key IN keys($isolations) WHERE
                h[$iso_prefix + key] IS NOT NULL AND h[$iso_prefix + key] = $isolations[key])
              AND h.timestamp >= $lower
              AND h.timestamp <= $upper
            RETURN h.content AS content
            ORDER BY h.timestamp ASC
            """,
            **params,
        )
        return [record["content"] for record in records]

    async def purge_history(
        self,
        user_id: str,
        start_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations_map = self._normalize_isolations(isolations)
        threshold = float(start_time) if start_time else float("-inf")
        params = {"user_id": user_id, "threshold": threshold}
        params.update(self._isolation_params(isolations_map))

        await self._execute_query(
            """
            MATCH (h:HistoryEntry {user_id: $user_id})
            WHERE ALL(key IN keys($isolations) WHERE
                h[$iso_prefix + key] IS NOT NULL AND h[$iso_prefix + key] = $isolations[key])
              AND h.timestamp <= $threshold
            DETACH DELETE h
            """,
            **params,
        )

    async def _ensure_constraints(self):
        await self._execute_query(
            """
            CREATE CONSTRAINT profile_entry_id IF NOT EXISTS
            FOR (p:ProfileEntry) REQUIRE p.profile_id IS UNIQUE
            """
        )
        await self._execute_query(
            """
            CREATE CONSTRAINT history_entry_id IF NOT EXISTS
            FOR (h:HistoryEntry) REQUIRE h.history_id IS UNIQUE
            """
        )
        await self._execute_query(
            """
            CREATE CONSTRAINT sequence_name IF NOT EXISTS
            FOR (s:Sequence) REQUIRE s.name IS UNIQUE
            """
        )

    async def _execute_query(self, query: str, **parameters: Any):
        if self._driver is None:
            raise RuntimeError("Neo4jProfileStorage has not been started")

        exec_params = {key: value for key, value in parameters.items()}
        if "database_" not in exec_params and self._database:
            exec_params["database_"] = self._database

        records, _, _ = await self._driver.execute_query(query, **exec_params)
        return records
