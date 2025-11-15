import asyncio
from typing import Protocol, runtime_checkable

from pydantic import InstanceOf

from memmachine.common.configuration import Configuration
from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.reranker import Reranker
from memmachine.common.resource_manager.embedder_manager import EmbedderManager
from memmachine.common.resource_manager.language_model_manager import (
    LanguageModelManager,
)
from memmachine.common.resource_manager.reranker_manager import RerankerManager
from memmachine.common.resource_manager.storage_manager import StorageManager
from memmachine.common.vector_graph_store import VectorGraphStore


class ResourceManager:
    def __init__(self, conf: Configuration):
        self._conf = conf
        self._conf.logging.apply()
        self._storage_manager: StorageManager = StorageManager(self._conf.storage)
        self._embedder_manager: EmbedderManager = EmbedderManager(self._conf.embeder)
        self._model_manager: LanguageModelManager = LanguageModelManager(
            self._conf.model
        )
        self._reranker_manager: RerankerManager = RerankerManager(self._conf.reranker)

    def build(self):
        self._storage_manager.build_all(validate=True)
        self._embedder_manager.build_all()
        self._model_manager.build_all()
        self._reranker_manager.build_all(self._embedder_manager.embedders)

    async def close(self):
        tasks = []
        if self._semantic_manager is not None:
            tasks.append(self._semantic_manager.close())

        tasks.append(self._storage_manager.close())

        await asyncio.gather(*tasks)

    def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        return self._storage_manager.get_vector_graph_store(name)

    def get_embedder(self, name: str) -> Embedder:
        return self._embedder_manager.get_embedder(name)

    def get_language_model(self, name: str) -> LanguageModel:
        return self._model_manager.get_language_model(name)

    def get_reranker(self, name: str) -> Reranker:
        return self._reranker_manager.get_reranker(name)
