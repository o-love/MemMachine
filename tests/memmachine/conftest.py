# conftest.py
import os
from unittest.mock import create_autospec

import pytest

from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine.common.language_model import LanguageModel
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel
from tests.memmachine.common.reranker.fake_embedder import FakeEmbedder


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="need --integration option to run")

    if not config.getoption("--integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture
def mock_llm_model():
    return create_autospec(LanguageModel, instance=True)


@pytest.fixture
def mock_llm_embedder():
    return FakeEmbedder()

@pytest.fixture
def openai_integration_config():
    open_api_key = os.environ.get("OPENAI_API_KEY")
    return {
        "api_key": open_api_key,
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
    }


@pytest.fixture
def openai_embedder(openai_integration_config):
    return OpenAIEmbedder(
        {"api_key": openai_integration_config["api_key"], "model": openai_integration_config["embedding_model"]}
    )

@pytest.fixture
def openai_llm_model(openai_integration_config):
    return OpenAILanguageModel(
        {"api_key": openai_integration_config["api_key"], "model": openai_integration_config["llm_model"]}
    )

