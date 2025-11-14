from typing import Any, cast

from pydantic import BaseModel, Field, InstanceOf

from ..data_types import ContentType, Episode, ResourceMgrProto
from ..declarative_memory import DeclarativeMemory, DeclarativeMemoryParams
from ..declarative_memory.data_types import (
    ContentType as DeclarativeMemoryContentType,
)
from ..declarative_memory.data_types import Episode as DeclarativeMemoryEpisode
from ...common.configuration.episodic_config import LongTermMemoryConf
from ...common.embedder import Embedder
from ...common.reranker import Reranker
from ...common.vector_graph_store import VectorGraphStore

content_type_to_declarative_memory_content_type_map = {
    ContentType.STRING: DeclarativeMemoryContentType.STRING,
}

declarative_memory_content_type_to_content_type_map = {
    DeclarativeMemoryContentType.STRING: ContentType.STRING,
}


class LongTermMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_id (str):
            Session identifier.
        max_chunk_length (int):
            Maximum length of a chunk in characters
            (default: 1000).
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
    """

    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    max_chunk_length: int = Field(
        1000,
        description="Maximum length of a chunk in characters.",
        gt=0,
    )
    vector_graph_store: InstanceOf[VectorGraphStore] = Field(
        ...,
        description="VectorGraphStore instance for storing and retrieving memories",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )
    enabled: bool = True


class LongTermMemory:
    _shared_resources: dict[str, Any] = {}

    def __init__(self,
                 resource_mgr: ResourceMgrProto,
                 conf: LongTermMemoryConf):
        # Note: Things look a bit weird during refactor...
        # Internal session_id is used for external group_id. This is intentional.
        embedder = resource_mgr.get_embedder(conf.embedder)
        reranker = resource_mgr.get_reranker(conf.reranker)
        graph_store = resource_mgr.get_graph_store(conf.vector_graph_store)
        self._conf = conf
        self._declarative_memory = DeclarativeMemory(
            DeclarativeMemoryParams(
                session_id=conf.session_id,
                content_metadata_template="[$timestamp] $producer_id: $content",
                vector_graph_store=graph_store,
                embedder=embedder,
                reranker=reranker,
            )
        )

    async def add_episode(self, episode: Episode):
        declarative_memory_episode = DeclarativeMemoryEpisode(
            uuid=episode.uuid,
            episode_type="default",
            content_type=content_type_to_declarative_memory_content_type_map[
                episode.content_type
            ],
            content=episode.content,
            timestamp=episode.timestamp,
            filterable_properties={
                key: value
                for key, value in {
                    "session_id": episode.session_key,
                    "producer_id": episode.producer_id,
                    "produced_for_id": episode.produced_for_id,
                }.items()
                if value is not None
            },
            user_metadata=episode.user_metadata,
        )
        await self._declarative_memory.add_episode(declarative_memory_episode)

    async def search(
        self,
        query: str,
        num_episodes_limit: int,
        id_filter: dict[str, str] = {},
    ):
        id_filter = {
            key: value for key, value in id_filter.items() if key != "group_id"
        }

        declarative_memory_episodes = await self._declarative_memory.search(
            query,
            num_episodes_limit=num_episodes_limit,
            property_filter=dict(id_filter),
        )
        return [
            Episode(
                uuid=declarative_memory_episode.uuid,
                episode_type=declarative_memory_episode.episode_type,
                content_type=(
                    declarative_memory_content_type_to_content_type_map[
                        declarative_memory_episode.content_type
                    ]
                ),
                content=declarative_memory_episode.content,
                timestamp=declarative_memory_episode.timestamp,
                session_key=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "session_id", ""
                    ),
                ),
                producer_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "producer_id", ""
                    ),
                ),
                produced_for_id=cast(
                    str,
                    declarative_memory_episode.filterable_properties.get(
                        "produced_for_id", ""
                    ),
                ),
                user_metadata=declarative_memory_episode.user_metadata,
            )
            for declarative_memory_episode in declarative_memory_episodes
        ]

    async def clear(self):
        await self._declarative_memory.forget_all()

    async def forget_session(self):
        await self._declarative_memory.forget_filtered_episodes(
            property_filter={
                "session_id": self._conf.session_id,
            }
        )

    def close(self):
        pass
