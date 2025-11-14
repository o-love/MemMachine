from typing import Protocol, runtime_checkable

from pydantic import InstanceOf

from ...common.configuration import Configuration
from ...common.embedder import Embedder
from ...common.language_model import LanguageModel
from ...common.resource_mgr.embedder_mgr import EmbedderManager
from ...common.resource_mgr.language_model_mgr import LanguageModelManager
from ...common.resource_mgr.storage_mgr import StorageManager
from ...episodic_memory_manager import (
    EpisodicMemoryManager,
    EpisodicMemoryManagerParams,
)
from ...history_store.history_sqlalchemy_store import SqlAlchemyHistoryStore
from ...history_store.history_storage import HistoryStorage
from ...semantic_memory.semantic_memory import SemanticService
from ...semantic_memory.semantic_model import ResourceRetriever, Resources, SetIdT
from ...semantic_memory.semantic_session_manager import SemanticSessionManager
from ...semantic_memory.semantic_session_resource import SessionIdManager
from ...semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)
from ...session_manager import SessionDataManagerImpl
from ...session_manager_interface import SessionDataManager
from ..reranker import Reranker
from ..vector_graph_store import VectorGraphStore
from .reranker_mgr import RerankerManager


class ResourceManager:
    def __init__(self, conf: Configuration):
        self._conf = conf
        self._conf.logging.apply()
        self._storage_mgr: StorageManager = StorageManager(self._conf.storage)
        self._embedder_mgr: EmbedderManager = EmbedderManager(self._conf.embeder)
        self._model_mgr: LanguageModelManager = LanguageModelManager(self._conf.model)
        self._reranker_mgr: RerankerManager = RerankerManager(self._conf.reranker)
        self._session_data_mgr: SessionDataManager | None = None
        self._episodic_memory_manager: EpisodicMemoryManager | None = None
        self._history_storage: HistoryStorage | None = None
        self._simple_semantic_session_id_manager: SessionIdManager | None = None
        self._semantic_session_resource_manager: (
            InstanceOf[ResourceRetriever] | None
        ) = None
        self._semantic_service: SemanticService | None = None
        self._semantic_session_manager: SemanticSessionManager | None = None

    def build(self):
        self._storage_mgr.build_all(validate=True)
        self._embedder_mgr.build_all()
        self._model_mgr.build_all()
        self._reranker_mgr.build_all(self._embedder_mgr.embedders)

    def close(self):
        if self._episodic_memory_manager is not None:
            self._episodic_memory_manager.close()
        # if self._profile_memory is not None:
        #     self._profile_memory.cleanup()
        self._storage_mgr.close()

    def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        return self._storage_mgr.get_vector_graph_store(name)

    def get_embedder(self, name: str) -> Embedder:
        return self._embedder_mgr.get_embedder(name)

    def get_language_model(self, name: str) -> LanguageModel:
        return self._model_mgr.get_language_model(name)

    def get_reranker(self, name: str) -> Reranker:
        return self._reranker_mgr.get_reranker(name)

    @property
    def session_data_manager(self) -> SessionDataManager:
        if self._session_data_mgr is not None:
            return self._session_data_mgr
        engine = self._storage_mgr.get_sql_engine(self._conf.sessiondb.storage_id)
        self._session_data_mgr = SessionDataManagerImpl(engine)
        return self._session_data_mgr

    @property
    def episodic_memory_manager(self) -> EpisodicMemoryManager:
        if self._episodic_memory_manager is not None:
            return self._episodic_memory_manager
        params = EpisodicMemoryManagerParams(resource_mgr=self)
        self._episodic_memory_manager = EpisodicMemoryManager(params)
        return self._episodic_memory_manager

    @property
    def history_storage(self) -> HistoryStorage:
        if self._history_storage is not None:
            return self._history_storage

        conf = self._conf.history_storage
        engine = self._storage_mgr.get_sql_engine(conf.database)

        self._history_storage = SqlAlchemyHistoryStore(engine)

        return self._history_storage

    @property
    def simple_semantic_session_id_manager(self) -> SessionIdManager:
        if self._simple_semantic_session_id_manager is not None:
            return self._simple_semantic_session_id_manager

        self._simple_semantic_session_id_manager = SessionIdManager()
        return self._simple_semantic_session_id_manager

    @property
    def semantic_session_resource_manager(self) -> InstanceOf[ResourceRetriever]:
        if self._semantic_session_resource_manager is not None:
            return self._semantic_session_resource_manager

        semantic_categories_by_isolation = self._conf.prompt.default_semantic_categories

        default_model_id = self._conf.semantic_service.model_id
        default_model = self.get_language_model(default_model_id)

        default_embedder_id = self._conf.semantic_service.embedder_id
        default_embedder = self.get_embedder(default_embedder_id)

        simple_session_id_manager = self.simple_semantic_session_id_manager

        class SemanticResourceRetriever:
            def get_resources(self, set_id: SetIdT) -> Resources:
                isolation_type = simple_session_id_manager.set_id_isolation_type(set_id)

                return Resources(
                    language_model=default_model,
                    embedder=default_embedder,
                    semantic_categories=semantic_categories_by_isolation[
                        isolation_type
                    ],
                )

        self._semantic_session_resource_manager = SemanticResourceRetriever()
        return self._semantic_session_resource_manager

    @property
    def semantic_service(self) -> SemanticService:
        if self._semantic_service is not None:
            return self._semantic_service

        conf = self._conf.semantic_service

        engine = self._storage_mgr.get_sql_engine(conf.database)
        semantic_storage = SqlAlchemyPgVectorSemanticStorage(engine)

        history_store = self.history_storage

        self._semantic_service = SemanticService(
            SemanticService.Params(
                semantic_storage=semantic_storage,
                history_storage=history_store,
            )
        )
        return self._semantic_service

    @property
    def semantic_session_manager(self):
        if self._semantic_session_manager is not None:
            return self._semantic_session_manager

        self._semantic_session_manager = SemanticSessionManager(
            self.semantic_service,
        )
        return self._semantic_session_manager
