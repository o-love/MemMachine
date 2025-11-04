from typing import Optional, Protocol

from pydantic import AwareDatetime, InstanceOf

from memmachine.semantic_memory.semantic_memory import SemanticService
from memmachine.semantic_memory.semantic_model import SemanticFeature
from memmachine.semantic_memory.semantic_session_resource import (
    ALL_MEMORY_TYPES,
    IsolationType,
    SessionData,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


class HistoryStorage(Protocol):
    def add_history(
        self, content: str, metadata: dict, created_at: Optional[AwareDatetime] = None
    ) -> int:
        raise NotImplementedError


class SemanticSessionManager:
    def __init__(
        self,
        semantic_service: SemanticService,
        history_storage: InstanceOf[HistoryStorage],
    ):
        self._semantic_service: SemanticService = semantic_service
        self._history_storage: SemanticStorageBase = history_storage

    async def add_message(
        self,
        message: str,
        session_data: SessionData,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        created_at: Optional[AwareDatetime] = None,
    ) -> int:
        h_id = await self._history_storage.add_history(
            content=message,
            metadata=session_data.__dict__,
            created_at=created_at,
        )

        set_ids = self._get_set_ids(session_data, memory_type)

        await self._semantic_service.add_message_to_sets(h_id, set_ids)

        return h_id

    async def search(
        self,
        message: str,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        min_distance: Optional[float] = None,
        type_names: Optional[list[str]] = None,
        tag_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        k: Optional[int] = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        set_ids = self._get_set_ids(session_data, memory_type)

        optionals_dft_args = {}
        if min_distance is not None:
            optionals_dft_args["min_distance"] = min_distance

        if k is not None:
            optionals_dft_args["k"] = k

        return await self._semantic_service.search(
            set_ids=set_ids,
            query=message,
            type_names=type_names,
            tag_names=tag_names,
            feature_names=feature_names,
            load_citations=load_citations,
            **optionals_dft_args,
        )

    async def number_of_uningested_messages(
        self,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
    ) -> int:
        set_ids = self._get_set_ids(session_data, memory_type)

        return await self._semantic_service.number_of_uningested(
            set_ids=set_ids,
        )

    async def add_feature(
        self,
        session_data: SessionData,
        *,
        memory_type: IsolationType,
        type_id: str,
        feature: str,
        value: str,
        tag: str,
        metadata: dict[str, str] | None = None,
        citations: list[int] | None = None,
    ) -> int:
        set_ids = self._get_set_ids(session_data, [memory_type])
        if len(set_ids) != 1:
            raise ValueError("Invalid set_ids", set_ids)
        set_id = set_ids[0]

        return await self._semantic_service.add_new_feature(
            set_id=set_id,
            type_name=type_id,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
            citations=citations,
        )

    async def get_feature(
        self, feature_id: int, load_citations: bool
    ) -> SemanticFeature | None:
        return await self._semantic_service.get_feature(
            feature_id, load_citations=load_citations
        )

    async def update_feature(
        self,
        feature_id: int,
        *,
        type_id: Optional[str] = None,
        feature: Optional[str] = None,
        value: Optional[str] = None,
        tag: Optional[str] = None,
        metadata: dict[str, str] | None = None,
    ):
        await self._semantic_service.update_feature(
            feature_id,
            type_id=type_id,
            feature=feature,
            value=value,
            tag=tag,
            metadata=metadata,
        )

    async def delete_features(self, feature_ids: list[int]):
        await self._history_storage.delete_features(feature_ids)

    async def get_set_features(
        self,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        type_names: Optional[list[str]],
        tag_names: Optional[list[str]],
        feature_names: Optional[list[str]],
    ) -> list[SemanticFeature]:
        set_ids = self._get_set_ids(session_data, memory_type)

        return await self._semantic_service.get_set_features(
            SemanticService.FeatureSearchOpts(
                set_ids=set_ids,
                type_names=type_names,
                feature_names=feature_names,
                tags=tag_names,
            )
        )

    async def delete_feature_set(
        self,
        session_data: SessionData,
        *,
        memory_type: list[IsolationType] = ALL_MEMORY_TYPES,
        type_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        thresh: Optional[int] = None,
        k: Optional[int] = None,
    ):
        set_ids = self._get_set_ids(session_data, memory_type)

        return await self._history_storage.delete_feature_set(
            set_ids=set_ids,
            type_names=type_names,
            feature_names=feature_names,
            tags=tags,
            thresh=thresh,
            k=k,
        )

    @staticmethod
    def _get_set_ids(
        session_data: SessionData,
        isolation_level: list[IsolationType],
    ) -> list[str]:
        s: list[str] = []
        if IsolationType.SESSION in isolation_level:
            session_id = session_data.session_id()
            if session_id is not None:
                s.append(session_id)

        if IsolationType.PROFILE in isolation_level:
            profile_id = session_data.profile_id()
            if profile_id is not None:
                s.append(profile_id)

        return s
