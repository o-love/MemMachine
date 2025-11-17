"""
Builder for LanguageModel instances.
"""

import asyncio
from asyncio import Lock

from memmachine.common.configuration.model_conf import LanguageModelConf
from memmachine.common.language_model.language_model import LanguageModel


class LanguageModelManager:
    def __init__(self, conf: LanguageModelConf):
        self._lock = Lock()
        self._language_models_lock: dict[str, Lock] = {}

        self.conf = conf
        self._language_models: dict[str, LanguageModel] = {}

    async def build_all(self) -> dict[str, LanguageModel]:
        names = set()
        for name in self.conf.openai_confs:
            names.add(name)
        for name in self.conf.aws_bedrock_confs:
            names.add(name)
        for name in self.conf.openai_compatible_confs:
            names.add(name)

        await asyncio.gather(*[self.get_language_model(name) for name in names])

        return self._language_models

    async def get_language_model(self, name: str) -> LanguageModel:
        if name in self._language_models:
            return self._language_models[name]

        if name not in self._language_models_lock:
            async with self._lock:
                self._language_models_lock.setdefault(name, Lock())

        async with self._language_models_lock[name]:
            if name in self._language_models:
                return self._language_models[name]

            llm_model = self._build_language_model(name)
            self._language_models[name] = llm_model
            return llm_model

    def _build_language_model(self, name: str) -> LanguageModel:
        if name in self.conf.openai_confs:
            return self._build_openai_model(name)
        elif name in self.conf.aws_bedrock_confs:
            return self._build_aws_bedrock_model(name)
        elif name in self.conf.openai_compatible_confs:
            return self._build_openai_compatible_model(name)
        else:
            raise ValueError(f"Language model with name {name} not found.")

    def _build_openai_model(self, name: str) -> LanguageModel:
        from memmachine.common.language_model.openai_language_model import (
            OpenAILanguageModel,
        )

        conf = self.conf.openai_confs[name]
        return OpenAILanguageModel(conf)

    def _build_aws_bedrock_model(self, name: str) -> LanguageModel:
        from memmachine.common.language_model.amazon_bedrock_language_model import (
            AmazonBedrockLanguageModel,
        )

        conf = self.conf.aws_bedrock_confs[name]
        return AmazonBedrockLanguageModel(conf)

    def _build_openai_compatible_model(self, name: str) -> LanguageModel:
        from memmachine.common.language_model.openai_compatible_language_model import (
            OpenAICompatibleLanguageModel,
        )

        conf = self.conf.openai_compatible_confs[name]
        return OpenAICompatibleLanguageModel(conf)
