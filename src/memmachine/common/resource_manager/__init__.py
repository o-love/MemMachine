import asyncio
from typing import Protocol, runtime_checkable

from pydantic import InstanceOf

from memmachine.common.configuration import Configuration
from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.reranker import Reranker
from memmachine.common.resource_manager.embedder_manager import EmbedderManager
from memmachine.common.resource_manager.language_model_manager import LanguageModelManager
from memmachine.common.resource_manager.reranker_manager import RerankerManager
from memmachine.common.resource_manager.semantic_manager import SemanticManager
from memmachine.common.resource_manager.storage_manager import StorageManager
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.episodic_memory_manager import EpisodicMemoryManager, EpisodicMemoryManagerParams
from memmachine.history_store.history_sqlalchemy_store import SqlAlchemyHistoryStore
from memmachine.history_store.history_storage import HistoryStorage
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager
from memmachine.session_manager import SessionDataManagerImpl
from memmachine.session_manager_interface import SessionDataManager


class ResourceManager:
    def __init__(self, conf: Configuration):
        self._conf = conf
        self._conf.logging.apply()
        self._storage_manager: StorageManager = StorageManager(self._conf.storage)
        self._embedder_manager: EmbedderManager = EmbedderManager(self._conf.embeder)
        self._model_manager: LanguageModelManager = LanguageModelManager(self._conf.model)
        self._reranker_manager: RerankerManager = RerankerManager(self._conf.reranker)
        self._session_data_manager: SessionDataManager | None = None
        self._episodic_memory_manager: EpisodicMemoryManager | None = None
        self._history_storage: HistoryStorage | None = None
        self._semantic_manager: SemanticManager | None = None

    def build(self):
        self._storage_manager.build_all(validate=True)
        self._embedder_manager.build_all()
        self._model_manager.build_all()
        self._reranker_manager.build_all(self._embedder_manager.embedders)

    async def close(self):
        tasks = []
        if self._episodic_memory_manager is not None:
            tasks.append(self._episodic_memory_manager.close())

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

    @property
    def session_data_manager(self) -> SessionDataManager:
        if self._session_data_manager is not None:
            return self._session_data_manager
        engine = self._storage_manager.get_sql_engine(self._conf.sessiondb.storage_id)
        self._session_data_manager = SessionDataManagerImpl(engine)
        return self._session_data_manager

    @property
    def episodic_memory_manager(self) -> EpisodicMemoryManager:
        if self._episodic_memory_manager is not None:
            return self._episodic_memory_manager
        params = EpisodicMemoryManagerParams(resource_manager=self)
        self._episodic_memory_manager = EpisodicMemoryManager(params)
        return self._episodic_memory_manager

    @property
    def history_storage(self) -> HistoryStorage:
        if self._history_storage is not None:
            return self._history_storage

        conf = self._conf.history_storage
        engine = self._storage_manager.get_sql_engine(conf.database)

        self._history_storage = SqlAlchemyHistoryStore(engine)

        return self._history_storage

    @property
    def semantic_manager(self):
        if self._semantic_manager is not None:
            return self._semantic_manager

        self._semantic_manager = SemanticManager(
            semantic_conf=self._conf.semantic_memory,
            prompt_conf=self._conf.prompt,
            storage_manager=self._storage_manager,
            embedder_manager=self._embedder_manager,
            model_manager=self._model_manager,
            history_storage=self.history_storage,
        )
        return self._semantic_manager

    @property
    def semantic_session_manager(self) -> SemanticSessionManager:
        return self.semantic_manager.semantic_session_manager
