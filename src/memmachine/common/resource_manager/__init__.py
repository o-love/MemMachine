from typing import Protocol, runtime_checkable

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.metrics_factory import MetricsFactory
from memmachine.common.reranker import Reranker
from memmachine.common.vector_graph_store import VectorGraphStore


@runtime_checkable
class CommonResourceManager(Protocol):
    def build(self):
        raise NotImplementedError

    async def close(self):
        raise NotImplementedError

    def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        raise NotImplementedError

    def get_embedder(self, name: str) -> Embedder:
        raise NotImplementedError

    def get_language_model(self, name: str) -> LanguageModel:
        raise NotImplementedError

    def get_reranker(self, name: str) -> Reranker:
        raise NotImplementedError

    def get_metrics_factory(self, name: str) -> MetricsFactory:
        raise NotImplementedError
