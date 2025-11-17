from typing import Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.reranker import Reranker
from memmachine.common.vector_graph_store import VectorGraphStore


@runtime_checkable
class CommonResourceManager(Protocol):
    async def build(self):
        raise NotImplementedError

    async def close(self):
        raise NotImplementedError

    async def get_sql_engine(self, name: str) -> AsyncEngine:
        raise NotImplementedError

    async def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        raise NotImplementedError

    async def get_embedder(self, name: str) -> Embedder:
        raise NotImplementedError

    async def get_language_model(self, name: str) -> LanguageModel:
        raise NotImplementedError

    async def get_reranker(self, name: str) -> Reranker:
        raise NotImplementedError

    async def get_metrics_factory(self, name: str) -> MetricsFactory:
        raise NotImplementedError
