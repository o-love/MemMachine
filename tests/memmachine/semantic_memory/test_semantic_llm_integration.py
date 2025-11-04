import pytest

from memmachine.semantic_memory.semantic_llm import llm_feature_update
from memmachine.server.prompt.profile_prompt import ProfilePromptType

pytestmark = pytest.mark.integration


@pytest.fixture
def llm_model(openai_llm_model):
    return openai_llm_model


@pytest.fixture
def semantic_prompt():
    return ProfilePromptType.prompt


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
