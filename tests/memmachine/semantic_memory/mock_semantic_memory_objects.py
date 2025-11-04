from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock

import numpy as np

from memmachine.common.embedder import Embedder, SimilarityMetric
from memmachine.semantic_memory.semantic_model import (
    HistoryMessage,
    Resources,
    SemanticFeature,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


class MockSemanticStorage(SemanticStorageBase):
    def __init__(self):
        self.get_history_messages_mock = AsyncMock()
        self.get_feature_set_mock = AsyncMock()
        self.add_feature_mock = AsyncMock()
        self.add_citations_mock = AsyncMock()
        self.delete_feature_set_mock = AsyncMock()
        self.mark_messages_ingested_mock = AsyncMock()
        self.delete_features_mock = AsyncMock()

    async def startup(self):
        raise NotImplementedError

    async def cleanup(self):
        raise NotImplementedError

    async def delete_all(self):
        raise NotImplementedError

    async def get_feature(self, feature_id: int, load_citations: bool = False):
        raise NotImplementedError

    async def add_feature(
        self,
        *,
        set_id: str,
        type_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        return await self.add_feature_mock(
            set_id=set_id,
            type_name=type_name,
            feature=feature,
            value=value,
            tag=tag,
            embedding=embedding,
            metadata=metadata,
        )

    async def update_feature(
        self,
        feature_id: int,
        *,
        set_id: Optional[str] = None,
        type_name: Optional[str] = None,
        feature: Optional[str] = None,
        value: Optional[str] = None,
        tag: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        raise NotImplementedError

    async def delete_features(self, feature_ids: list[int]):
        await self.delete_features_mock(feature_ids)

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
        return await self.get_feature_set_mock(
            set_ids=set_ids,
            type_names=type_names,
            feature_names=feature_names,
            tags=tags,
            k=k,
            vector_search_opts=vector_search_opts,
            tag_threshold=tag_threshold,
            load_citations=load_citations,
        )

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
        await self.delete_feature_set_mock(
            set_ids=set_ids,
            type_names=type_names,
            feature_names=feature_names,
            tags=tags,
            thresh=thresh,
            k=k,
            vector_search_opts=vector_search_opts,
        )

    async def add_citations(self, feature_id: int, history_ids: list[int]):
        await self.add_citations_mock(feature_id, history_ids)

    async def add_history(
        self,
        content: str,
        metadata: Optional[dict[str, str]] = None,
        created_at: Optional[datetime] = None,
    ) -> int:
        raise NotImplementedError

    async def get_history(self, history_id: int):
        raise NotImplementedError

    async def delete_history(self, history_ids: list[int]):
        raise NotImplementedError

    async def delete_history_messages(
        self,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        raise NotImplementedError

    async def get_history_messages(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        k: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        is_ingested: Optional[bool] = None,
    ) -> list[HistoryMessage]:
        return await self.get_history_messages_mock(
            set_ids=set_ids,
            k=k,
            start_time=start_time,
            end_time=end_time,
            is_ingested=is_ingested,
        )

    async def get_history_messages_count(
        self,
        *,
        set_ids: Optional[list[str]] = None,
        k: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        is_ingested: Optional[bool] = None,
    ) -> int:
        raise NotImplementedError

    async def add_history_to_set(self, set_id: str, history_id: int):
        raise NotImplementedError

    async def mark_messages_ingested(self, *, set_id: str, ids: list[int]) -> None:
        await self.mark_messages_ingested_mock(set_id=set_id, ids=ids)


class MockEmbedder(Embedder):
    def __init__(self):
        self.ingest_calls: list[list[str]] = []

    async def ingest_embed(self, inputs: list[Any], max_attempts: int = 1):
        self.ingest_calls.append(list(inputs))
        return [[float(len(value)), float(len(value)) * -1] for value in inputs]

    async def search_embed(self, queries: list[Any], max_attempts: int = 1):
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return "embedder-double"

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class MockResourceRetriever:
    def __init__(self, resources: Resources):
        self._resources = resources
        self.seen_ids: list[str] = []

    def get_resources(self, set_id: str) -> Resources:
        self.seen_ids.append(set_id)
        return self._resources
