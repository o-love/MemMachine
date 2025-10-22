import asyncio
import json
import time
from collections import defaultdict
from typing import Any, Iterable
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, InstanceOf

from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore
from memmachine.profile_memory.storage.storage_base import ProfileStorageBase


class VectorGraphProfileStorage(ProfileStorageBase):
    """
    ``ProfileStorageBase`` implementation backed by a ``VectorGraphStore``.
    """

    class Params(BaseModel):
        vector_graph_store: InstanceOf[VectorGraphStore]
        close_store_on_cleanup: bool = False

    def __init__(self, params: Params):
        self._graph_store: VectorGraphStore = params.vector_graph_store
        self._close_on_cleanup = params.close_store_on_cleanup

        self._isolation_prefix = "__iso__"
        self._profile_label = "ProfileEntry"
        self._history_label = "HistoryEntry"
        self._ingestion_marker_label = "IngestionMarker"
        self._embedding_property = "embedding"
        self._citation_relation = "CITED"
        self._ingestion_relation = "INGESTED"

    async def startup(self):
        return None

    async def cleanup(self):
        if self._close_on_cleanup:
            await self._graph_store.close()

    async def delete_all(self):
        profile_nodes = await self._find_profile_nodes({})
        await self._delete_nodes(node.uuid for node in profile_nodes)

        history_nodes = await self._find_history_nodes({})
        await self._delete_history_nodes(history_nodes)

        marker_nodes = await self._graph_store.search_matching_nodes(
            required_labels={self._ingestion_marker_label},
        )
        await self._delete_nodes(node.uuid for node in marker_nodes)

    async def get_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, dict[str, Any | list[Any]]]:
        isolations_map = self._normalize_isolations(isolations)
        profile_nodes = await self._graph_store.search_matching_nodes(
            required_labels={self._profile_label},
            required_properties=self._apply_isolations_to_properties(
                {"user_id": user_id}, isolations_map
            ),
        )

        profile: dict[str, dict[str, Any | list[Any]]] = {}
        for node in profile_nodes:
            tag = node.properties.get("tag")
            feature = node.properties.get("feature")
            value = node.properties.get("value")
            if tag is None or feature is None:
                continue
            payload = {"value": value}
            tag_bucket = profile.setdefault(str(tag), {})
            existing = tag_bucket.setdefault(str(feature), [])
            assert isinstance(existing, list)
            existing.append(payload)

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
    ) -> list[str]:
        isolations_map = self._normalize_isolations(isolations)
        profile_nodes = await self._find_profile_nodes(
            self._apply_isolations_to_properties(
                {
                    "user_id": user_id,
                    "feature": feature,
                    "tag": tag,
                    "value": str(value),
                },
                isolations_map,
            )
        )
        citations: list[str] = []
        seen: set[str] = set()
        for node in profile_nodes:
            related_history_nodes = await self._graph_store.search_related_nodes(
                node_uuid=node.uuid,
                allowed_relations={self._citation_relation},
                find_sources=False,
                find_targets=True,
                required_labels={self._history_label},
            )
            for history_node in related_history_nodes:
                history_uuid = str(history_node.uuid)
                if history_uuid not in seen:
                    seen.add(history_uuid)
                    citations.append(history_uuid)
        return citations

    async def delete_profile(
        self,
        user_id: str,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations_map = self._normalize_isolations(isolations)
        nodes = await self._find_profile_nodes(
            self._apply_isolations_to_properties({"user_id": user_id}, isolations_map)
        )
        await self._delete_nodes(node.uuid for node in nodes)

    async def add_profile_feature(
        self,
        user_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
        citations: list[int | str] | None = None,
    ):
        metadata = metadata or {}
        isolations_map = self._normalize_isolations(isolations)
        citation_uuids = [
            uuid
            for uuid in (
                self._parse_uuid(identifier) for identifier in (citations or [])
            )
            if uuid is not None
        ]

        embedding_list = [float(x) for x in np.asarray(embedding).flatten()]
        metadata_json = json.dumps(metadata)
        isolations_json = json.dumps(isolations_map)

        properties: dict[str, Any] = {
            "user_id": user_id,
            "feature": feature,
            "value": str(value),
            "tag": tag,
            self._embedding_property: embedding_list,
            "metadata_json": metadata_json,
            "isolations_json": isolations_json,
            "created_at": float(time.time()),
        }
        properties.update(self._build_isolation_properties(isolations_map))

        profile_node = Node(
            uuid=uuid4(),
            labels={self._profile_label},
            properties=properties,
        )
        await self._graph_store.add_nodes([profile_node])

        if citation_uuids:
            history_nodes = await self._find_history_nodes_by_uuids(citation_uuids)
            edges = [
                Edge(
                    uuid=uuid4(),
                    source_uuid=profile_node.uuid,
                    target_uuid=history_node.uuid,
                    relation=self._citation_relation,
                )
                for history_node in history_nodes
            ]
            if edges:
                await self._graph_store.add_edges(edges)

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
        query_vector = np.asarray(qemb).flatten().astype(float)
        qnorm = float(np.linalg.norm(query_vector))
        if qnorm == 0:
            return []

        profile_nodes = await self._graph_store.search_similar_nodes(
            query_embedding=query_vector.tolist(),
            embedding_property_name=self._embedding_property,
            limit=k if k > 0 else None,
            required_labels={self._profile_label},
            required_properties=self._apply_isolations_to_properties(
                {"user_id": user_id}, isolations_map
            ),
        )

        scored: list[tuple[float, Node]] = []
        for node in profile_nodes:
            embedding_values = node.properties.get(self._embedding_property)
            if embedding_values is None:
                continue
            embedding_vector = np.asarray(embedding_values, dtype=float).flatten()
            denom = float(np.linalg.norm(embedding_vector)) * qnorm
            if denom == 0:
                continue
            score = float(np.dot(embedding_vector, query_vector) / denom)
            if score <= min_cos:
                continue
            scored.append((score, node))

        scored.sort(key=lambda item: item[0], reverse=True)
        if k > 0:
            scored = scored[:k]
        if not scored:
            return []

        citation_contents: dict[UUID, list[str]] = {}
        if include_citations:
            citation_contents = await self._get_citation_contents(
                [node for _, node in scored]
            )

        results: list[dict[str, Any]] = []
        for score, node in scored:
            tag = node.properties.get("tag")
            feature = node.properties.get("feature")
            value = node.properties.get("value")
            if tag is None or feature is None or value is None:
                continue
            metadata: dict[str, Any] = {
                "id": str(node.uuid),
                "similarity_score": score,
            }
            if include_citations:
                metadata["citations"] = citation_contents.get(node.uuid, [])
            results.append(
                {
                    "tag": tag,
                    "feature": feature,
                    "value": value,
                    "metadata": metadata,
                }
            )
        return results

    async def delete_profile_feature_by_id(self, pid: int | str):
        node_uuid = self._parse_uuid(pid)
        if node_uuid is None:
            return
        node = await self._find_profile_node_by_uuid(node_uuid)
        if node is None:
            return
        await self._delete_nodes([node.uuid])

    async def get_all_citations_for_ids(
        self, pids: list[int | str]
    ) -> list[tuple[str, dict[str, bool | int | float | str]]]:
        results: list[tuple[str, dict[str, bool | int | float | str]]] = []
        seen: set[str] = set()
        for pid in pids:
            node_uuid = self._parse_uuid(pid)
            if node_uuid is None:
                continue
            node = await self._find_profile_node_by_uuid(node_uuid)
            if node is None:
                continue
            related_history_nodes = await self._graph_store.search_related_nodes(
                node_uuid=node.uuid,
                allowed_relations={self._citation_relation},
                find_sources=False,
                find_targets=True,
                required_labels={self._history_label},
            )
            for history_node in related_history_nodes:
                history_uuid = str(history_node.uuid)
                if history_uuid in seen:
                    continue
                seen.add(history_uuid)
                isolations_raw = history_node.properties.get("isolations_json")
                isolations_dict: dict[str, bool | int | float | str]
                if isinstance(isolations_raw, str) and isolations_raw:
                    isolations_dict = json.loads(isolations_raw)
                else:
                    isolations_dict = {}
                results.append((history_uuid, isolations_dict))
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
        filters: dict[str, Any] = {
            "user_id": user_id,
            "feature": feature,
            "tag": tag,
        }
        if value is not None:
            filters["value"] = str(value)
        nodes = await self._find_profile_nodes(
            self._apply_isolations_to_properties(filters, isolations_map)
        )
        await self._delete_nodes(node.uuid for node in nodes)

    async def get_large_profile_sections(
        self,
        user_id: str,
        thresh: int,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        isolations_map = self._normalize_isolations(isolations)
        nodes = await self._find_profile_nodes(
            self._apply_isolations_to_properties({"user_id": user_id}, isolations_map)
        )

        sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            tag = node.properties.get("tag")
            feature = node.properties.get("feature")
            value = node.properties.get("value")
            if tag is None or feature is None or value is None:
                continue
            sections[str(tag)].append(
                {
                    "tag": tag,
                    "feature": feature,
                    "value": value,
                    "metadata": {"id": str(node.uuid)},
                }
            )

        return [entries for entries in sections.values() if len(entries) >= thresh]

    async def add_history(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> dict[str, Any]:
        metadata_dict = metadata or {}
        isolations_map = self._normalize_isolations(isolations)

        metadata_json = json.dumps(metadata_dict)
        isolations_json = json.dumps(isolations_map)
        timestamp = float(time.time())

        properties: dict[str, Any] = {
            "user_id": user_id,
            "content": content,
            "metadata_json": metadata_json,
            "isolations_json": isolations_json,
            "timestamp": timestamp,
        }
        properties.update(self._build_isolation_properties(isolations_map))

        history_node = Node(
            uuid=uuid4(),
            labels={self._history_label},
            properties=properties,
        )
        await self._graph_store.add_nodes([history_node])

        return {
            "id": str(history_node.uuid),
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
        nodes = await self._find_history_nodes(
            self._apply_isolations_to_properties({"user_id": user_id}, isolations_map)
        )
        lower = float(start_time) if start_time else float("-inf")
        upper = float(end_time) if end_time else float("inf")
        nodes_to_delete = [
            node
            for node in nodes
            if lower
            <= float(node.properties.get("timestamp", float("-inf")))
            <= upper
        ]
        await self._delete_history_nodes(nodes_to_delete)

    async def get_history_messages_by_ingestion_status(
        self,
        user_id: str,
        k: int = 0,
        is_ingested: bool = False,
    ) -> list[dict[str, Any]]:
        nodes = await self._find_history_nodes({"user_id": user_id})
        ingestion_status = await self._ingestion_status_map(nodes)

        filtered = [
            node
            for node in nodes
            if ingestion_status.get(node.uuid, False) is is_ingested
        ]
        filtered.sort(
            key=lambda node: float(node.properties.get("timestamp", 0.0)), reverse=True
        )
        if k > 0:
            filtered = filtered[:k]

        return [self._history_node_to_dict(node) for node in filtered]

    async def get_uningested_history_messages_count(self) -> int:
        nodes = await self._find_history_nodes({})
        ingestion_status = await self._ingestion_status_map(nodes)
        return sum(not ingestion_status.get(node.uuid, False) for node in nodes)

    async def mark_messages_ingested(
        self,
        ids: list[int | str],
    ) -> None:
        for history_id in ids:
            history_uuid = self._parse_uuid(history_id)
            if history_uuid is None:
                continue
            history_node = await self._find_history_node_by_uuid(history_uuid)
            if history_node is None:
                continue
            if await self._is_history_ingested(history_node):
                continue

            marker_node = Node(
                uuid=uuid4(),
                labels={self._ingestion_marker_label},
                properties={
                    "history_uuid": str(history_uuid),
                    "created_at": float(time.time()),
                },
            )
            await self._graph_store.add_nodes([marker_node])
            await self._graph_store.add_edges(
                [
                    Edge(
                        uuid=uuid4(),
                        source_uuid=history_node.uuid,
                        target_uuid=marker_node.uuid,
                        relation=self._ingestion_relation,
                    )
                ]
            )

    async def get_history_message(
        self,
        user_id: str,
        start_time: int = 0,
        end_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ) -> list[str]:
        isolations_map = self._normalize_isolations(isolations)
        nodes = await self._find_history_nodes(
            self._apply_isolations_to_properties({"user_id": user_id}, isolations_map)
        )
        lower = float(start_time) if start_time else float("-inf")
        upper = float(end_time) if end_time else float("inf")
        matching_nodes = [
            node
            for node in nodes
            if lower
            <= float(node.properties.get("timestamp", float("-inf")))
            <= upper
        ]
        matching_nodes.sort(
            key=lambda node: float(node.properties.get("timestamp", 0.0))
        )
        return [
            str(node.properties.get("content"))
            for node in matching_nodes
            if node.properties.get("content") is not None
        ]

    async def purge_history(
        self,
        user_id: str,
        start_time: int = 0,
        isolations: dict[str, bool | int | float | str] | None = None,
    ):
        isolations_map = self._normalize_isolations(isolations)
        nodes = await self._find_history_nodes(
            self._apply_isolations_to_properties({"user_id": user_id}, isolations_map)
        )
        threshold = float(start_time) if start_time else float("-inf")
        nodes_to_delete = [
            node
            for node in nodes
            if float(node.properties.get("timestamp", float("-inf"))) <= threshold
        ]
        await self._delete_history_nodes(nodes_to_delete)

    @staticmethod
    def _normalize_isolations(
        isolations: dict[str, bool | int | float | str] | None,
    ) -> dict[str, bool | int | float | str]:
        if isolations is None:
            return {}
        return dict(isolations)

    def _build_isolation_properties(
        self, isolations: dict[str, bool | int | float | str]
    ) -> dict[str, Any]:
        return {
            f"{self._isolation_prefix}{key}": value for key, value in isolations.items()
        }

    def _apply_isolations_to_properties(
        self,
        base: dict[str, Any],
        isolations: dict[str, bool | int | float | str],
    ) -> dict[str, Any]:
        merged = dict(base)
        merged.update(self._build_isolation_properties(isolations))
        return merged

    async def _find_profile_nodes(self, filters: dict[str, Any]) -> list[Node]:
        return await self._graph_store.search_matching_nodes(
            required_labels={self._profile_label},
            required_properties=filters,
        )

    async def _find_profile_node_by_uuid(self, node_uuid: UUID) -> Node | None:
        nodes = await self._graph_store.search_matching_nodes(
            limit=1,
            required_labels={self._profile_label},
            required_properties={"uuid": str(node_uuid)},
        )
        return nodes[0] if nodes else None

    async def _find_history_nodes(self, filters: dict[str, Any]) -> list[Node]:
        return await self._graph_store.search_matching_nodes(
            required_labels={self._history_label},
            required_properties=filters,
        )

    async def _find_history_node_by_uuid(self, node_uuid: UUID) -> Node | None:
        nodes = await self._graph_store.search_matching_nodes(
            limit=1,
            required_labels={self._history_label},
            required_properties={"uuid": str(node_uuid)},
        )
        return nodes[0] if nodes else None

    async def _find_history_nodes_by_uuids(
        self, history_uuids: Iterable[UUID]
    ) -> list[Node]:
        tasks = [
            self._find_history_node_by_uuid(history_uuid)
            for history_uuid in history_uuids
        ]
        if not tasks:
            return []
        nodes = await asyncio.gather(*tasks)
        return [node for node in nodes if node is not None]

    async def _delete_nodes(self, node_uuids: Iterable[UUID]):
        uuids = list(dict.fromkeys(node_uuids))
        if not uuids:
            return
        await self._graph_store.delete_nodes(uuids)

    async def _delete_history_nodes(self, nodes: Iterable[Node]):
        nodes_list = list(nodes)
        if not nodes_list:
            return

        marker_nodes: list[Node] = []
        for node in nodes_list:
            related_markers = await self._graph_store.search_related_nodes(
                node_uuid=node.uuid,
                allowed_relations={self._ingestion_relation},
                find_sources=False,
                find_targets=True,
                required_labels={self._ingestion_marker_label},
            )
            marker_nodes.extend(related_markers)

        await self._delete_nodes(node.uuid for node in nodes_list)
        await self._delete_nodes(marker.uuid for marker in marker_nodes)

    async def _get_citation_contents(
        self, profile_nodes: Iterable[Node]
    ) -> dict[UUID, list[str]]:
        nodes = list(profile_nodes)
        if not nodes:
            return {}
        tasks = [
            self._graph_store.search_related_nodes(
                node_uuid=node.uuid,
                allowed_relations={self._citation_relation},
                find_sources=False,
                find_targets=True,
                required_labels={self._history_label},
            )
            for node in nodes
        ]
        related_lists = await asyncio.gather(*tasks)
        citation_contents: dict[UUID, list[str]] = {}
        for node, related in zip(nodes, related_lists):
            citation_contents[node.uuid] = [
                str(history_node.properties.get("content"))
                for history_node in related
                if history_node.properties.get("content") is not None
            ]
        return citation_contents

    async def _ingestion_status_map(
        self, history_nodes: Iterable[Node]
    ) -> dict[UUID, bool]:
        nodes = list(history_nodes)
        if not nodes:
            return {}
        tasks = [
            self._graph_store.search_related_nodes(
                node_uuid=node.uuid,
                allowed_relations={self._ingestion_relation},
                find_sources=False,
                find_targets=True,
                required_labels={self._ingestion_marker_label},
            )
            for node in nodes
        ]
        related_lists = await asyncio.gather(*tasks)
        return {
            node.uuid: len(related) > 0
            for node, related in zip(nodes, related_lists)
        }

    async def _is_history_ingested(self, history_node: Node) -> bool:
        related = await self._graph_store.search_related_nodes(
            node_uuid=history_node.uuid,
            allowed_relations={self._ingestion_relation},
            find_sources=False,
            find_targets=True,
            required_labels={self._ingestion_marker_label},
        )
        return len(related) > 0

    def _history_node_to_dict(self, node: Node) -> dict[str, Any]:
        metadata_json = node.properties.get("metadata_json") or "{}"
        isolations_json = node.properties.get("isolations_json") or "{}"
        return {
            "id": str(node.uuid),
            "user_id": node.properties.get("user_id"),
            "content": node.properties.get("content"),
            "metadata": metadata_json,
            "isolations": isolations_json,
        }

    @staticmethod
    def _parse_uuid(identifier: int | str | UUID | None) -> UUID | None:
        if isinstance(identifier, UUID):
            return identifier
        if identifier is None:
            return None
        try:
            return UUID(str(identifier))
        except (TypeError, ValueError):
            return None
