# Load custom prompts from files if specified in the config,
# overriding defaults.
import asyncio

from pydantic import InstanceOf

from memmachine.common.configuration import PromptConf, SemanticMemoryConf
from memmachine.common.resource_manager import StorageManager, EmbedderManager, LanguageModelManager
from memmachine.history_store.history_storage import HistoryStorage
from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import ResourceRetriever, SetIdT, Resources
from memmachine.semantic_memory.semantic_session_manager import SemanticSessionManager
from memmachine.semantic_memory.semantic_session_resource import SessionIdManager, IsolationType
from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import SqlAlchemyPgVectorSemanticStorage


def load_prompt(prompt_config: PromptConf, key: str, default_value: str) -> str:
    """
    Helper function to load a prompt from a file path specified in the
    config.

    Args:
        prompt_config: The dictionary containing prompt file paths.
        key: The key for the specific prompt to load.
        default_value: The default prompt content to use if the file
        is not specified or found.

    Returns:
        The content of the prompt.
    """

    custom_prompt_path = prompt_config.get(key)
    if custom_prompt_path:
        with open(custom_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return default_value

class SemanticManager:
    def __init__(
            self,
            semantic_conf: SemanticMemoryConf,
            prompt_conf: PromptConf,
            storage_manager: StorageManager,
            embedder_manager: EmbedderManager,
            model_manager: LanguageModelManager,
            history_storage: HistoryStorage,
         ):
        self._conf = semantic_conf
        self._prompt_conf = prompt_conf
        self._storage_manager = storage_manager
        self._embedder_manager = embedder_manager
        self._model_manager = model_manager
        self._history_storage = history_storage



        self._simple_semantic_session_id_manager: SessionIdManager | None = None
        self._semantic_session_resource_manager: (
                InstanceOf[ResourceRetriever] | None
        ) = None
        self._semantic_service: SemanticService | None = None
        self._semantic_session_manager: SemanticSessionManager | None = None

    async def close(self):
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

    @property
    def semantic_session_resource_manager(self) -> InstanceOf[ResourceRetriever]:
        if self._semantic_session_resource_manager is not None:
            return self._semantic_session_resource_manager

        semantic_categories_by_isolation = self._conf.prompt.default_semantic_categories

        simple_session_id_manager = self.simple_semantic_session_id_manager

        default_embedder = self._embedder_manager.get_embedder(self._conf.embedding_model)
        default_model = self._model_manager.get_language_model(self._conf.llm_model)

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

        engine = self._storage_manager.get_sql_engine(conf.database)
        semantic_storage = SqlAlchemyPgVectorSemanticStorage(engine)

        history_store = self._history_storage
        resource_retriever = self.semantic_session_resource_manager

        self._semantic_service = SemanticService(
            SemanticService.Params(
                semantic_storage=semantic_storage,
                history_storage=history_store,
                resource_retriever=resource_retriever,
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
