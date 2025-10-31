from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

from pydantic import AwareDatetime

from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


@runtime_checkable
class SessionData(Protocol):
    def producer_id(self) -> str | None:
        raise NotImplementedError

    def session_id(self) -> str | None:
        raise NotImplementedError


class SemanticSessionManager(ABC):
    @abstractmethod
    async def add_message(
        self,
        message: str,
        session_data: SessionData,
        created_at: Optional[AwareDatetime] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    async def search_memories(
        self, message: str, session_data: SessionData
    ) -> list[SemanticFeature]:
        raise NotImplementedError

    @abstractmethod
    async def get_memories(
        self, session_data: SessionData, type_names: Optional[list[str]]
    ) -> list[SemanticFeature]:
        raise NotImplementedError


def _get_set_ids(session_data: SessionData) -> list[str]:
    s: list[str] = []
    if session_data.producer_id() is not None:
        s.append(session_data.producer_id())

    if session_data.session_id() is not None:
        s.append(session_data.session_id())

    return s


def build() -> SemanticSessionManager:
    # TODO: build resource manager and semantic_service with it.
    return _SemanticSessionManagerImpl(
        semantic_service=SemanticService(),
        semantic_storage=SemanticStorageBase(),
    )


class _SemanticSessionManagerImpl(SemanticSessionManager):
    def __init__(
        self,
        semantic_service: SemanticService,
        semantic_storage: SemanticStorageBase,
    ):
        self._semantic_service: SemanticService = semantic_service
        self._semantic_storage: SemanticStorageBase = semantic_storage

    async def add_message(
        self,
        message: str,
        session_data: SessionData,
        created_at: Optional[AwareDatetime] = None,
    ):
        h_id = await self._semantic_storage.add_history(
            content=message,
            metadata=session_data.__dict__,
        )

        set_ids = _get_set_ids(session_data)

        await self._semantic_service.add_message_to_sets(h_id, set_ids)

    async def search_memories(
        self, message: str, session_data: SessionData
    ) -> list[SemanticFeature]:
        set_ids = _get_set_ids(session_data)
        return await self._semantic_service.search(set_ids=set_ids, query=message)

    async def get_memories(
        self, session_data: SessionData, type_names: Optional[list[str]]
    ) -> list[SemanticFeature]:
        set_ids = _get_set_ids(session_data)

        return await self._semantic_service.get_set_features(
            SemanticService.FeatureSearchOpts(
                set_ids=set_ids,
                type_names=type_names,
            )
        )
