import asyncio
from asyncio import Lock
from collections import defaultdict
from typing import Protocol

import boto3
from pydantic import InstanceOf
from typing_extensions import runtime_checkable

from memmachine.common.configuration.reranker_conf import RerankerConf
from memmachine.common.embedder import Embedder
from memmachine.common.reranker import Reranker


@runtime_checkable
class EmbedderFactory(Protocol):
    async def get_embedder(self, name: str) -> Embedder:
        raise NotImplementedError


class RerankerManager:
    def __init__(
        self, conf: RerankerConf, embedder_factory: InstanceOf[EmbedderFactory]
    ):
        self.conf = conf
        self.rerankers: dict[str, Reranker] = {}

        self._embedder_factory: EmbedderFactory = embedder_factory
        self._lock = Lock()
        self._rerankers_lock = defaultdict(Lock)

    async def build_all(self) -> dict[str, Reranker]:
        names = [
            name
            for d in [
                self.conf.bm25,
                self.conf.cross_encoder,
                self.conf.amazon_bedrock,
                self.conf.embedder,
                self.conf.identity,
                self.conf.rrf_hybrid,
            ]
            for name in d.keys()
        ]
        tasks = [self.get_reranker(name) for name in names]
        await asyncio.gather(*tasks)
        return self.rerankers

    async def get_reranker(self, name: str) -> Reranker:
        if name in self.rerankers:
            return self.rerankers[name]

        if name not in self._rerankers_lock:
            async with self._lock:
                self._rerankers_lock.setdefault(name, Lock())

        async with self._rerankers_lock[name]:
            if name in self.rerankers:
                return self.rerankers[name]

            reranker = await self._build_reranker(name)
            self.rerankers[name] = reranker
            return reranker

    async def _build_reranker(self, name: str) -> Reranker:
        if name in self.conf.bm25:
            return await self._build_bm25_reranker(name)
        elif name in self.conf.cross_encoder:
            return await self._build_cross_encoder_reranker(name)
        elif name in self.conf.amazon_bedrock:
            return await self._build_amazon_bedrock_reranker(name)
        elif name in self.conf.embedder:
            return await self._build_embedder_reranker(name)
        elif name in self.conf.identity:
            return await self._build_identity_reranker(name)
        elif name in self.conf.rrf_hybrid:
            return await self._build_rrf_hybrid_reranker(name)
        else:
            raise ValueError(f"Reranker with name {name} not found.")

    async def _build_bm25_reranker(self, name: str) -> Reranker:
        from memmachine.common.reranker.bm25_reranker import BM25Reranker

        conf = self.conf.bm25[name]
        self.rerankers[name] = BM25Reranker(conf)
        return self.rerankers[name]

    async def _build_cross_encoder_reranker(self, name: str) -> Reranker:
        from memmachine.common.reranker.cross_encoder_reranker import (
            CrossEncoderReranker,
        )

        conf = self.conf.cross_encoder[name]
        self.rerankers[name] = CrossEncoderReranker(conf)
        return self.rerankers[name]

    async def _build_amazon_bedrock_reranker(self, name: str) -> Reranker:
        from memmachine.common.reranker.amazon_bedrock_reranker import (
            AmazonBedrockReranker,
            AmazonBedrockRerankerParams,
        )

        conf = self.conf.amazon_bedrock[name]

        client = boto3.client(
            "bedrock-agent-runtime",
            region_name=conf.region,
            aws_access_key_id=conf.aws_access_key_id,
            aws_secret_access_key=conf.aws_secret_access_key,
        )
        params = AmazonBedrockRerankerParams(
            client=client,
            region=conf.region,
            model_id=conf.model_id,
            additional_model_request_fields=conf.additional_model_request_fields,
        )
        self.rerankers[name] = AmazonBedrockReranker(params)
        return self.rerankers[name]

    async def _build_embedder_reranker(self, name: str) -> Reranker:
        from ..reranker.embedder_reranker import (
            EmbedderReranker,
            EmbedderRerankerParams,
        )

        conf = self.conf.embedder[name]
        embedder = await self._embedder_factory.get_embedder(conf.embedder_id)
        params = EmbedderRerankerParams(embedder=embedder)
        self.rerankers[name] = EmbedderReranker(params)
        return self.rerankers[name]

    async def _build_identity_reranker(self, name: str) -> Reranker:
        from ..reranker.identity_reranker import IdentityReranker

        self.rerankers[name] = IdentityReranker()
        return self.rerankers[name]

    async def _build_rrf_hybrid_reranker(self, name: str) -> Reranker:
        """Build RRF hybrid rerankers by combining existing rerankers.

        This method must be called after all individual rerankers have been built,
        """
        from ..reranker.rrf_hybrid_reranker import (
            RRFHybridReranker,
            RRFHybridRerankerParams,
        )

        conf = self.conf.rrf_hybrid[name]
        rerankers = []
        for reranker_id in conf.reranker_ids:
            try:
                reranker = await self.get_reranker(reranker_id)
                rerankers.append(reranker)
            except Exception as e:
                raise ValueError(
                    f"Failed to get reranker with id {reranker_id} for "
                    f"RRFHybridReranker {name}: {e}"
                ) from e
        params = RRFHybridRerankerParams(rerankers=rerankers, k=conf.k)
        self.rerankers[name] = RRFHybridReranker(params)
        return self.rerankers[name]
