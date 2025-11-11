from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import AwareDatetime, InstanceOf

from memmachine.semantic_memory.semantic_model import HistoryMessage, SemanticFeature


class SemanticStorageBase(ABC):
    """
    The base class for Semantic storage
    """

    @abstractmethod
    async def startup(self):
        """
        initializations for the semantic storage,
        such as creating a connection to the database
        """
        raise NotImplementedError

    @abstractmethod
    async def cleanup(self):
        """
        cleanup for the semantic storage
        such as closing connection to the database
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self):
        """
        delete all semantic features in the storage
        such as truncating the database table
        """
        raise NotImplementedError

    @abstractmethod
    async def get_feature(
        self,
        feature_id: int,
        load_citations: bool = False,
    ) -> SemanticFeature | None:
        raise NotImplementedError

    @abstractmethod
    async def add_feature(
        self,
        *,
        set_id: str,
        category_name: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Add a new feature to the user.
        """
        raise NotImplementedError

    @abstractmethod
    async def update_feature(
        self,
        feature_id: int,
        *,
        set_id: str | None = None,
        category_name: str | None = None,
        feature: str | None = None,
        value: str | None = None,
        tag: str | None = None,
        embedding: InstanceOf[np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    async def delete_features(self, feature_ids: list[int]):
        raise NotImplementedError

    @dataclass
    class VectorSearchOpts:
        """Parameters controlling vector similarity constraints for retrieval."""

        query_embedding: InstanceOf[np.ndarray]
        min_distance: float | None = 0.7

    @abstractmethod
    async def get_feature_set(
        self,
        *,
        set_ids: list[str] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        k: int | None = None,
        vector_search_opts: VectorSearchOpts | None = None,
        tag_threshold: int | None = None,
        load_citations: bool = False,
    ) -> list[SemanticFeature]:
        """
        Get feature set by user id
        Return: A list of KV for each feature and value.
           The value is an array with: feature value, feature tag and deleted, update time, create time and delete time.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_feature_set(
        self,
        *,
        set_ids: list[str] | None = None,
        category_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        tags: list[str] | None = None,
        thresh: int | None = None,
        k: int | None = None,
        vector_search_opts: VectorSearchOpts | None = None,
    ):
        """
        Delete all the features by id
        """
        raise NotImplementedError

    @abstractmethod
    async def add_citations(self, feature_id: int, history_ids: list[int]):
        raise NotImplementedError

    @abstractmethod
    async def add_history(
        self,
        content: str,
        metadata: dict[str, str] | None = None,
        created_at: AwareDatetime | None = None,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    async def get_history(
        self,
        history_id: int,
    ) -> HistoryMessage | None:
        raise NotImplementedError

    @abstractmethod
    async def delete_history(
        self,
        history_ids: list[int],
    ):
        raise NotImplementedError

    @abstractmethod
    async def delete_history_messages(
        self,
        *,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    async def get_history_messages(
        self,
        *,
        set_ids: list[str] | None = None,
        k: int | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        is_ingested: bool | None = None,
    ) -> list[HistoryMessage]:
        """
        retrieve the list of the history messages for the user
        with the ingestion status, up to k messages if k > 0
        """
        raise NotImplementedError

    @abstractmethod
    async def get_history_messages_count(
        self,
        *,
        set_ids: list[str] | None = None,
        k: int | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        is_ingested: bool | None = None,
    ) -> int:
        """
        retrieve the count of the history messages
        """
        raise NotImplementedError

    @abstractmethod
    async def add_history_to_set(self, set_id: str, history_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    async def mark_messages_ingested(
        self,
        *,
        set_id: str,
        ids: list[int],
    ) -> None:
        """
        mark the messages with the id as ingested
        """
        raise NotImplementedError
