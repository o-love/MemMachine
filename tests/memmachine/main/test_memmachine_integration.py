"""Integration tests for :mod:`memmachine.main.memmachine`."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from memmachine.common.configuration import (
    Configuration,
    EpisodeStoreConf,
    EpisodicMemoryConfPartial,
    LogConf,
    PromptConf,
    ResourcesConf,
    SemanticMemoryConf,
    SessionManagerConf,
)
from memmachine.common.configuration.episodic_config import (
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.configuration.log_conf import LogLevel
from memmachine.episode_store.episode_model import EpisodeEntry
from memmachine.main.memmachine import MemMachine, MemoryType
from memmachine.semantic_memory.semantic_session_resource import SessionIdManager


LONG_MEM_EVAL_PATH = Path("tests/data/longmemeval_snippet.json")


class IntegrationSessionData:
    """Simple session descriptor that satisfies both MemMachine protocols."""

    def __init__(
        self,
        *,
        session_key: str,
        user_id: str,
        role_id: str | None = None,
    ) -> None:
        manager = SessionIdManager()
        generated = manager.generate_session_data(
            user_id=user_id,
            session_id=session_key,
            role_id=role_id,
        )
        self._session_key = session_key
        self._session_id = generated.session_id()
        self._user_profile_id = generated.user_profile_id()
        self._role_profile_id = generated.role_profile_id()

    @property
    def session_key(self) -> str:
        return self._session_key

    # The semantic session manager expects callable attributes.
    def session_id(self) -> str | None:  # pragma: no cover - trivial accessor
        return self._session_id

    def user_profile_id(self) -> str | None:  # pragma: no cover - trivial accessor
        return self._user_profile_id

    def role_profile_id(self) -> str | None:  # pragma: no cover - trivial accessor
        return self._role_profile_id


@pytest.fixture
def long_mem_raw_question() -> dict:
    with LONG_MEM_EVAL_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def long_mem_convos(long_mem_raw_question: dict) -> list[list[dict[str, str]]]:
    return long_mem_raw_question["haystack_sessions"]


@pytest.fixture
def long_mem_question(long_mem_raw_question: dict) -> str:
    return long_mem_raw_question["question"]


@pytest.fixture
def llm_model(real_llm_model):
    return real_llm_model


@pytest.fixture
def memmachine_config(
    pg_server,
    neo4j_container,
    openai_integration_config,
) -> Configuration:
    pg_db_id = "pg_memory"
    graph_store_id = "neo4j_graph"

    pg_config = {
        "provider": "postgres",
        "config": {
            "host": pg_server["host"],
            "port": pg_server["port"],
            "user": pg_server["user"],
            "password": pg_server["password"],
            "db_name": pg_server["database"],
        },
    }

    neo4j_url = urlparse(neo4j_container["uri"])
    neo4j_config = {
        "provider": "neo4j",
        "config": {
            "host": neo4j_url.hostname,
            "port": neo4j_url.port,
            "user": neo4j_container["username"],
            "password": neo4j_container["password"],
        },
    }

    resources_conf = ResourcesConf(
        **{
            "databases": {
                pg_db_id: pg_config,
                graph_store_id: neo4j_config,
            },
            "embedders": {
                "openai_embedder": {
                    "provider": "openai",
                    "config": {
                        "model": openai_integration_config["embedding_model"],
                        "api_key": openai_integration_config["api_key"],
                        "dimensions": 1536,
                    },
                }
            },
            "language_models": {
                "openai_responses_model": {
                    "provider": "openai-responses",
                    "config": {
                        "model": openai_integration_config["llm_model"],
                        "api_key": openai_integration_config["api_key"],
                    },
                }
            },
            "rerankers": {
                "identity_reranker": {
                    "provider": "identity",
                }
            },
        }
    )

    summary_user_prompt = (
        "Based on the following episodes: {episodes}, and the previous summary: {summary}, "
        "please update the summary. Keep it under {max_length} characters."
    )

    episodic_conf = EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(
            vector_graph_store=graph_store_id,
            embedder="openai_embedder",
            reranker="identity_reranker",
        ),
        short_term_memory=ShortTermMemoryConfPartial(
            llm_model="openai_responses_model",
            summary_prompt_system="You are a helpful assistant.",
            summary_prompt_user=summary_user_prompt,
        ),
    )

    return Configuration(
        logging=LogConf(level=LogLevel.INFO, path="test-memachine.log"),
        prompt=PromptConf(),
        episodic_memory=episodic_conf,
        semantic_memory=SemanticMemoryConf(
            database=pg_db_id,
            llm_model="openai_responses_model",
            embedding_model="openai_embedder",
        ),
        session_manager=SessionManagerConf(database=pg_db_id),
        episode_store=EpisodeStoreConf(database=pg_db_id),
        resources=resources_conf,
    )


@pytest_asyncio.fixture
async def memmachine_runtime(memmachine_config: Configuration):
    memmachine = MemMachine(memmachine_config)
    await memmachine.start()
    try:
        yield memmachine
    finally:
        try:
            episode_storage = await memmachine._resources.get_episode_storage()
            await episode_storage.delete_episode_messages()

            session_manager = await memmachine._resources.get_session_data_manager()
            await session_manager.drop_tables()

            semantic_service = await memmachine._resources.get_semantic_service()
            # Access private storage for cleanup in tests.
            await semantic_service._semantic_storage.delete_all()
        finally:
            await memmachine.stop()


@pytest.fixture
def memmachine_session_data() -> IntegrationSessionData:
    session_key = f"longmem-session-{uuid.uuid4().hex}"
    return IntegrationSessionData(
        session_key=session_key,
        user_id="longmem-user",
        role_id="longmem-role",
    )


class TestMemMachineLongMemEval:
    """Run LongMemEval snippets end-to-end through MemMachine."""

    @staticmethod
    def _conversation_to_entries(convo: list[dict[str, str]]) -> list[EpisodeEntry]:
        return [
            EpisodeEntry(
                content=turn["content"],
                producer_id=f"{turn['role']}_profile",
                producer_role=turn["role"],
            )
            for turn in convo
        ]

    @classmethod
    async def _ingest_conversations(
        cls,
        memmachine: MemMachine,
        session_data: IntegrationSessionData,
        conversations: list[list[dict[str, str]]],
    ) -> None:
        for conversation in conversations:
            entries = cls._conversation_to_entries(conversation)
            await memmachine.add_episodes(
                session_data=session_data,
                episode_entries=entries,
            )

    @staticmethod
    async def _wait_for_semantic_ingestion(
        memmachine: MemMachine,
        session_data: IntegrationSessionData,
        *,
        max_polls: int,
    ) -> None:
        semantic_manager = await memmachine._resources.get_semantic_session_manager()
        remaining = -1
        for _ in range(max_polls):
            remaining = await semantic_manager.number_of_uningested_messages(
                session_data=session_data,
            )
            if remaining == 0:
                return
            await asyncio.sleep(1)

        pytest.fail(f"Messages are not ingested, count={remaining}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_long_mem_eval_smoke(
        self,
        memmachine_runtime: MemMachine,
        memmachine_session_data: IntegrationSessionData,
        long_mem_convos: list[list[dict[str, str]]],
    ) -> None:
        await memmachine_runtime.create_session(
            memmachine_session_data.session_key,
            description="LongMemEval smoke",
            embedder_name="openai_embedder",
            reranker_name="identity_reranker",
        )

        smoke_convos = long_mem_convos[0]
        if len(smoke_convos) > 2:
            smoke_convos = smoke_convos[:2]

        await self._ingest_conversations(
            memmachine_runtime,
            memmachine_session_data,
            [smoke_convos],
        )
        await self._wait_for_semantic_ingestion(
            memmachine_runtime,
            memmachine_session_data,
            max_polls=180,
        )

        result = await memmachine_runtime.list_search(
            memmachine_session_data,
            target_memories=[MemoryType.Semantic],
        )
        assert result.semantic_memory is not None
        assert len(result.semantic_memory) > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_long_mem_eval_answers_question(
        self,
        memmachine_runtime: MemMachine,
        memmachine_session_data: IntegrationSessionData,
        long_mem_convos: list[list[dict[str, str]]],
        long_mem_question: str,
        llm_model,
    ) -> None:
        await memmachine_runtime.create_session(
            memmachine_session_data.session_key,
            description="LongMemEval full ingestion",
            embedder_name="openai_embedder",
            reranker_name="identity_reranker",
        )

        await self._ingest_conversations(
            memmachine_runtime,
            memmachine_session_data,
            long_mem_convos,
        )
        await self._wait_for_semantic_ingestion(
            memmachine_runtime,
            memmachine_session_data,
            max_polls=1200,
        )

        search_response = await memmachine_runtime.query_search(
            memmachine_session_data,
            target_memories=[MemoryType.Semantic],
            query=long_mem_question,
            limit=4,
        )
        semantic_context = (search_response.semantic_memory or [])[:4]

        system_prompt = (
            "You are an AI assistant who answers questions based on provided information. "
            "Use the persona profile to answer the question, and state when information "
            "cannot be found with the exact phrase 'The relevant information is not found in the provided context.'"
        )
        answer_prompt_template = "Persona Profile:\n{}\nQuestion: {}\nAnswer:"
        eval_prompt = answer_prompt_template.format(semantic_context, long_mem_question)
        eval_resp = await llm_model.generate_response(system_prompt, eval_prompt)

        assert (
            "The relevant information is not found in the provided context"
            not in eval_resp
        )
