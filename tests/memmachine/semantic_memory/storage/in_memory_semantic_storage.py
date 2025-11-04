from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np
from pydantic import AwareDatetime, InstanceOf

from memmachine.semantic_memory.semantic_model import HistoryMessage, SemanticFeature
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> Optional[float]:
    lnorm = float(np.linalg.norm(lhs))
    rnorm = float(np.linalg.norm(rhs))
    if lnorm == 0 or rnorm == 0:
        return None
    return float(np.dot(lhs, rhs) / (lnorm * rnorm))


@dataclass
class _FeatureEntry:
    id: int
    set_id: str
    semantic_type_id: str
    tag: str
    feature: str
    value: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass
class _HistoryEntry:
    id: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


class InMemorySemanticStorage(SemanticStorageBase):
    """In-memory implementation of :class:`SemanticStorageBase` used for testing."""

    def __init__(self):
        self._features_by_id: dict[int, _FeatureEntry] = {}
        self._feature_ids_by_set: dict[str, list[int]] = {}
        self._history_by_id: dict[int, _HistoryEntry] = {}
        self._set_history_map: dict[str, dict[int, bool]] = {}
        self._history_to_sets: dict[int, dict[str, bool]] = {}
        self._next_feature_id = 1
        self._next_history_id = 1
        self._lock = asyncio.Lock()

    async def startup(self):
        return None

    async def cleanup(self):
        return None

    async def delete_all(self):
        async with self._lock:
            self._features_by_id.clear()
            self._feature_ids_by_set.clear()
            self._history_by_id.clear()
            self._set_history_map.clear()
            self._history_to_sets.clear()
            self._next_feature_id = 1
            self._next_history_id = 1

    async def get_feature(
        self,
        feature_id: int,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        async with self._lock:
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return None
            return self._feature_to_model(entry, load_citations=load_citations)

    async def add_feature(
        self,
        *,
        set_id: str,
        type_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        metadata = dict(metadata or {})
        async with self._lock:
            feature_id = self._next_feature_id
            self._next_feature_id += 1
            entry = _FeatureEntry(
                id=feature_id,
                set_id=set_id,
                semantic_type_id=type_name,
                tag=tag,
                feature=feature,
                value=value,
                embedding=np.array(embedding, dtype=float, copy=True),
                metadata=metadata,
            )
            self._features_by_id[feature_id] = entry
            self._feature_ids_by_set.setdefault(set_id, []).append(feature_id)
            return feature_id

    async def update_feature(
        self,
        feature_id: int,
        *,
        set_id: Optional[str] = None,
        type_name: Optional[str] = None,
        feature: Optional[str] = None,
        value: Optional[str] = None,
        tag: Optional[str] = None,
        embedding: Optional[InstanceOf[np.ndarray]] = None,
        metadata: dict[str, Any] | None = None,
    ):
        async with self._lock:
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            if set_id is not None and set_id != entry.set_id:
                old_set_ids = self._feature_ids_by_set.get(entry.set_id, [])
                if feature_id in old_set_ids:
                    old_set_ids.remove(feature_id)
                    if not old_set_ids:
                        self._feature_ids_by_set.pop(entry.set_id, None)
                self._feature_ids_by_set.setdefault(set_id, []).append(feature_id)
                entry.set_id = set_id

            if type_name is not None:
                entry.semantic_type_id = type_name
            if feature is not None:
                entry.feature = feature
            if value is not None:
                entry.value = value
            if tag is not None:
                entry.tag = tag
            if embedding is not None:
                entry.embedding = np.array(embedding, dtype=float, copy=True)
            if metadata is not None:
                entry.metadata = dict(metadata)

            entry.updated_at = _utcnow()

    async def delete_features(self, feature_ids: list[int]):
        if not feature_ids:
            return

        async with self._lock:
            for feature_id in feature_ids:
                entry = self._features_by_id.pop(feature_id, None)
                if entry is None:
                    continue
                ids = self._feature_ids_by_set.get(entry.set_id)
                if ids and feature_id in ids:
                    ids.remove(feature_id)
                    if not ids:
                        self._feature_ids_by_set.pop(entry.set_id, None)

    async def get_feature_set(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        type_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
        tag_threshold: Optional[int] = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        async with self._lock:
            entries = self._filter_features(
                set_ids=set_ids,
                type_names=type_names,
                feature_names=feature_names,
                tags=tags,
                k=k,
                vector_search_opts=vector_search_opts,
                tag_threshold=tag_threshold,
            )
            return [
                self._feature_to_model(entry, load_citations=load_citations)
                for entry in entries
            ]

    async def delete_feature_set(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        type_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        thresh: Optional[int] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
    ):
        async with self._lock:
            to_remove = self._filter_features(
                set_ids=set_ids,
                type_names=type_names,
                feature_names=feature_names,
                tags=tags,
                k=k,
                vector_search_opts=vector_search_opts,
                tag_threshold=thresh,
            )
            for entry in to_remove:
                self._features_by_id.pop(entry.id, None)
                ids = self._feature_ids_by_set.get(entry.set_id)
                if ids and entry.id in ids:
                    ids.remove(entry.id)
                    if not ids:
                        self._feature_ids_by_set.pop(entry.set_id, None)

    async def add_citations(self, feature_id: int, history_ids: list[int]):
        if not history_ids:
            return

        async with self._lock:
            entry = self._features_by_id.get(feature_id)
            if entry is None:
                return

            existing: set[int] = set(entry.citations)
            for history_id in history_ids:
                if history_id not in self._history_by_id:
                    continue
                if history_id not in existing:
                    entry.citations.append(history_id)
                    existing.add(history_id)
            entry.updated_at = _utcnow()

    async def add_history(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
        created_at: Optional[AwareDatetime] = None,
    ) -> int:
        metadata = dict(metadata or {})
        async with self._lock:
            history_id = self._next_history_id
            self._next_history_id += 1
            entry = _HistoryEntry(
                id=history_id,
                content=content,
                metadata=metadata,
                created_at=created_at if created_at is not None else _utcnow(),
            )
            self._history_by_id[history_id] = entry
            return history_id

    async def get_history(
        self,
        history_id: int,
    ) -> Optional[HistoryMessage]:
        async with self._lock:
            entry = self._history_by_id.get(history_id)
            if entry is None:
                return None
            return self._history_to_model(entry)

    async def delete_history(self, history_ids: list[int]):
        if not history_ids:
            return

        async with self._lock:
            for history_id in history_ids:
                entry = self._history_by_id.pop(history_id, None)
                if entry is None:
                    continue
                sets = self._history_to_sets.pop(history_id, {})
                for set_id in sets:
                    mapping = self._set_history_map.get(set_id)
                    if mapping and history_id in mapping:
                        mapping.pop(history_id)
                        if not mapping:
                            self._set_history_map.pop(set_id, None)

    async def delete_history_messages(
        self,
        *,
        start_time: Optional[AwareDatetime] = None,
        end_time: Optional[AwareDatetime] = None,
    ):
        async with self._lock:
            to_delete: list[int] = []
            for history_id, entry in self._history_by_id.items():
                if start_time is not None and entry.created_at < start_time:
                    continue
                if end_time is not None and entry.created_at > end_time:
                    continue
                to_delete.append(history_id)

            for history_id in to_delete:
                entry = self._history_by_id.pop(history_id, None)
                if entry is None:
                    continue
                sets = self._history_to_sets.pop(history_id, {})
                for set_id in sets:
                    mapping = self._set_history_map.get(set_id)
                    if mapping and history_id in mapping:
                        mapping.pop(history_id)
                        if not mapping:
                            self._set_history_map.pop(set_id, None)

    async def get_history_messages(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        k: Optional[int] = None,
        start_time: Optional[AwareDatetime] = None,
        end_time: Optional[AwareDatetime] = None,
        is_ingested: Optional[bool] = None,
        set_id: Optional[str] = None,
    ) -> list[HistoryMessage]:
        if set_ids is not None and set_id is not None:
            raise ValueError("Provide either set_id or set_ids, not both")
        if set_id is not None:
            set_ids = [set_id]

        async with self._lock:
            entries = self._filter_history_entries(
                set_ids=set_ids,
                start_time=start_time,
                end_time=end_time,
                is_ingested=is_ingested,
            )

            if k is not None:
                entries = entries[:k]

            return [self._history_to_model(entry) for entry in entries]

    async def get_history_messages_count(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        k: Optional[int] = None,
        start_time: Optional[AwareDatetime] = None,
        end_time: Optional[AwareDatetime] = None,
        is_ingested: Optional[bool] = None,
        set_id: Optional[str] = None,
    ) -> int:
        if set_ids is not None and set_id is not None:
            raise ValueError("Provide either set_id or set_ids, not both")
        if set_id is not None:
            set_ids = [set_id]

        async with self._lock:
            entries = self._filter_history_entries(
                set_ids=set_ids,
                start_time=start_time,
                end_time=end_time,
                is_ingested=is_ingested,
            )
            if k is not None:
                return len(entries[:k])
            return len(entries)

    async def add_history_to_set(
        self,
        set_id: str,
        history_id: int,
    ) -> None:
        async with self._lock:
            if history_id not in self._history_by_id:
                raise ValueError(f"History id {history_id} not found")

            set_map = self._set_history_map.setdefault(set_id, {})
            history_map = self._history_to_sets.setdefault(history_id, {})

            if history_id not in set_map:
                set_map[history_id] = False
            if set_id not in history_map:
                history_map[set_id] = False

    async def mark_messages_ingested(
        self,
        *,
        set_id: str,
        ids: list[int],
    ) -> None:
        if not ids:
            raise ValueError("No ids provided")

        async with self._lock:
            set_map = self._set_history_map.get(set_id)
            if set_map is None:
                return

            for history_id in ids:
                if history_id in set_map:
                    set_map[history_id] = True
                    self._history_to_sets.setdefault(history_id, {})[set_id] = True

    def _feature_to_model(
        self,
        entry: _FeatureEntry,
        *,
        load_citations: bool,
    ) -> SemanticFeature:
        citations: Optional[list[HistoryMessage]] = None
        if load_citations:
            citations = [
                self._history_to_model(self._history_by_id[history_id])
                for history_id in entry.citations
                if history_id in self._history_by_id
            ]

        return SemanticFeature(
            set_id=entry.set_id,
            type=entry.semantic_type_id,
            tag=entry.tag,
            feature=entry.feature,
            value=entry.value,
            metadata=SemanticFeature.Metadata(
                id=entry.id,
                citations=citations,
            ),
        )

    def _history_to_model(self, entry: _HistoryEntry) -> HistoryMessage:
        return HistoryMessage(
            content=entry.content,
            created_at=entry.created_at,
            metadata=HistoryMessage.Metadata(
                id=entry.id,
            ),
        )

    def _filter_features(
        self,
        *,
        set_ids: Optional[list[str]],
        type_names: Optional[list[str]],
        feature_names: Optional[list[str]],
        tags: Optional[list[str]],
        k: Optional[int],
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts],
        tag_threshold: Optional[int],
    ) -> list[_FeatureEntry]:
        entries: list[_FeatureEntry] = list(self._features_by_id.values())

        if set_ids:
            allowed = set(set_ids)
            entries = [entry for entry in entries if entry.set_id in allowed]
        if type_names:
            allowed = set(type_names)
            entries = [entry for entry in entries if entry.semantic_type_id in allowed]
        if feature_names:
            allowed = set(feature_names)
            entries = [entry for entry in entries if entry.feature in allowed]
        if tags:
            allowed = set(tags)
            entries = [entry for entry in entries if entry.tag in allowed]

        if vector_search_opts is not None:
            filtered: list[tuple[float, _FeatureEntry]] = []
            for entry in entries:
                similarity = _cosine_similarity(
                    entry.embedding, vector_search_opts.query_embedding
                )
                # Treat None (zero-norm vectors) as having -infinity similarity
                # so they appear last in results but are still included
                if similarity is None:
                    similarity = float("-inf")
                min_cos = vector_search_opts.min_distance
                if min_cos is not None and similarity < min_cos:
                    continue
                filtered.append((similarity, entry))

            filtered.sort(key=lambda pair: pair[0], reverse=True)
            entries = [entry for _, entry in filtered]
        else:
            # Only sort by creation time when not doing vector search
            entries.sort(key=lambda e: (e.created_at, e.id))

        if k is not None:
            entries = entries[:k]

        if tag_threshold is not None:
            counts = Counter(entry.tag for entry in entries)
            entries = [entry for entry in entries if counts[entry.tag] >= tag_threshold]

        return entries

    def _filter_history_entries(
        self,
        *,
        set_ids: Optional[list[str]],
        start_time: Optional[AwareDatetime],
        end_time: Optional[AwareDatetime],
        is_ingested: Optional[bool],
    ) -> list[_HistoryEntry]:
        entries: Iterable[_HistoryEntry]
        entries = list(self._history_by_id.values())

        if set_ids is not None or is_ingested is not None:
            allowed_sets = set(set_ids) if set_ids is not None else None
            filtered: list[_HistoryEntry] = []
            for entry in entries:
                set_status = self._history_to_sets.get(entry.id, {})

                if allowed_sets is not None:
                    relevant_sets = set(set_status.keys()) & allowed_sets
                    if not relevant_sets:
                        continue
                else:
                    relevant_sets = set(set_status.keys())
                    if is_ingested is not None and not relevant_sets:
                        continue

                if is_ingested is None:
                    filtered.append(entry)
                    continue

                if any(set_status[set_id] == is_ingested for set_id in relevant_sets):
                    filtered.append(entry)

            entries = filtered

        results: list[_HistoryEntry] = []
        for entry in entries:
            if start_time is not None and entry.created_at < start_time:
                continue
            if end_time is not None and entry.created_at > end_time:
                continue
            results.append(entry)

        results.sort(key=lambda e: (e.created_at, e.id))
        return results
