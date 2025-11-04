import pytest

from memmachine.semantic_memory.semantic_llm import (
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.server.prompt.profile_prompt import UserProfileSemanticType

pytestmark = pytest.mark.integration


@pytest.fixture(
    params=[
        pytest.param("bedrock", marks=pytest.mark.integration),
        pytest.param("openai", marks=pytest.mark.integration),
    ]
)
def llm_model(request):
    match request.param:
        case "bedrock":
            return request.getfixturevalue("bedrock_llm_model")
        case "openai":
            return request.getfixturevalue("openai_llm_model")
        case _:
            raise ValueError(f"Unknown LLM model type: {request.param}")


@pytest.fixture
def semantic_prompt():
    return UserProfileSemanticType.prompt


@pytest.fixture
def update_prompt(semantic_prompt):
    return semantic_prompt.update_prompt


@pytest.fixture
def consolidation_prompt(semantic_prompt):
    return semantic_prompt.consolidation_prompt


@pytest.mark.asyncio
async def test_semantic_llm_update_with_empty_profile(
    llm_model,
    update_prompt,
):
    commands = await llm_feature_update(
        features=[],
        message_content="I like blue cars made in Berlin, Germany",
        model=llm_model,
        update_prompt=update_prompt,
    )

    assert commands is not None


@pytest.mark.asyncio
async def test_semantic_llm_consolidate_with_empty_profile(
    llm_model,
    consolidation_prompt,
):
    result = await llm_consolidate_features(
        features=[],
        consolidate_prompt=consolidation_prompt,
        model=llm_model,
    )

    assert result is not None
