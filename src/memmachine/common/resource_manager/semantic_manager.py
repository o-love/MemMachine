import asyncio

from pydantic import InstanceOf

from memmachine.common.configuration import PromptConf, SemanticMemoryConf
from memmachine.common.resource_manager import CommonResourceManager
from memmachine.episode_store.episode_storage import EpisodeStorage
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import (
    ResourceRetriever,
    Resources,
    SetIdT,
)
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager
from memmachine.semantic_memory.semantic_session_resource import (
    SessionIdManager,
)
from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)


class SemanticResourceManager:
    def __init__(
        self,
        *,
        semantic_conf: SemanticMemoryConf,
        prompt_conf: PromptConf,
        resource_manager: InstanceOf[CommonResourceManager],
        history_storage: EpisodeStorage,
    ) -> None:
        self._resource_manager = resource_manager
        self._conf = semantic_conf
        self._prompt_conf = prompt_conf
        self._history_storage = history_storage

        self._simple_semantic_session_id_manager: SessionIdManager | None = None
        self._semantic_session_resource_manager: (
            InstanceOf[ResourceRetriever] | None
        ) = None
        self._semantic_service: SemanticService | None = None
        self._semantic_session_manager: SemanticSessionManager | None = None

    async def close(self) -> None:
        tasks = []

        if self._semantic_service is not None:
            tasks.append(self._semantic_service.stop())

        await asyncio.gather(*tasks)

    @property
    def simple_semantic_session_id_manager(self) -> SessionIdManager:
        if self._simple_semantic_session_id_manager is not None:
            return self._simple_semantic_session_id_manager

        self._simple_semantic_session_id_manager = SessionIdManager()
        return self._simple_semantic_session_id_manager

    async def get_semantic_session_resource_manager(
        self,
    ) -> InstanceOf[ResourceRetriever]:
        if self._semantic_session_resource_manager is not None:
            return self._semantic_session_resource_manager

        semantic_categories_by_isolation = self._conf.prompt.default_semantic_categories

        simple_session_id_manager = self.simple_semantic_session_id_manager

        default_embedder = await self._resource_manager.get_embedder(
            self._conf.embedding_model,
        )
        default_model = await self._resource_manager.get_language_model(
            self._conf.llm_model,
        )

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

    async def get_semantic_service(self) -> SemanticService:
        if self._semantic_service is not None:
            return self._semantic_service

        conf = self._conf.semantic_service

        engine = await self._resource_manager.get_sql_engine(conf.database)
        semantic_storage = SqlAlchemyPgVectorSemanticStorage(engine)

        history_store = self._history_storage
        resource_retriever = await self.get_semantic_session_resource_manager()

        self._semantic_service = SemanticService(
            SemanticService.Params(
                semantic_storage=semantic_storage,
                history_storage=history_store,
                resource_retriever=resource_retriever,
            ),
        )
        return self._semantic_service

    async def get_semantic_session_manager(self) -> SemanticSessionManager:
        if self._semantic_session_manager is not None:
            return self._semantic_session_manager

        self._semantic_session_manager = SemanticSessionManager(
            await self.get_semantic_service(),
        )
        return self._semantic_session_manager
