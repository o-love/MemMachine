from unittest.mock import MagicMock

import pytest

from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_llm import (
    llm_consolidate_features,
    llm_feature_update,
)
from memmachine.semantic_memory.semantic_model import SemanticCommand, SemanticFeature


@pytest.fixture
def magic_mock_llm_model():
    yield MagicMock(spec=LanguageModel)


@pytest.fixture
def basic_features():
    return [
        SemanticFeature(
            type="Profile",
            tag="food",
            feature="favorite_pizza",
            value="peperoni pizza",
        ),
        SemanticFeature(
            type="Profile",
            tag="food",
            feature="favorite_bread",
            value="whole grain",
        ),
    ]


@pytest.mark.asyncio
async def test_empty_update_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an empty LLM response from the prompt
    magic_mock_llm_model.generate_response.return_value = ("{}", None)

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    # Expect no commands to be returned
    assert commands == []


@pytest.mark.asyncio
async def test_single_command_update_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given a single LLM response from the prompt
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "0": {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue"
            }
        }
        """,
        None,
    )

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert commands == [
        SemanticCommand(
            command="add",
            tag="car",
            feature="favorite_car_color",
            value="blue",
        )
    ]


@pytest.mark.asyncio
async def test_single_command_with_think_update_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_response.return_value = (
        """
        <think>
        I am thinking
        </think>
        {
            "0": {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue"
            }
        }
        """,
        None,
    )

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    assert commands == [
        SemanticCommand(
            command="add",
            tag="car",
            feature="favorite_car_color",
            value="blue",
        )
    ]


@pytest.mark.asyncio
async def test_empty_consolidate_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_response.return_value = ("{}", None)

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is None


@pytest.mark.asyncio
async def test_no_action_consolidate_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "keep_memories": [],
            "consolidated_memories": []
        }
        """,
        None,
    )

    new_feature_resp = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    assert new_feature_resp is not None
    assert new_feature_resp.keep_memories == []
    assert new_feature_resp.consolidated_memories == []


@pytest.mark.asyncio
async def test_llm_feature_update_handles_non_dict_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM that returns a list instead of dict
    magic_mock_llm_model.generate_response.return_value = ("[]", None)

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    # Expect an empty command list with warning logged
    assert commands == []


@pytest.mark.asyncio
async def test_llm_feature_update_handles_invalid_command_structure(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM response with invalid command structures
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "0": {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car_color",
                "value": "blue"
            },
            "1": "not a dict",
            "2": {
                "command": "add",
                "missing_required_field": "oops"
            }
        }
        """,
        None,
    )

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like blue cars",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    # Expect only valid command to be processed
    assert len(commands) == 1
    assert commands[0].command == "add"
    assert commands[0].feature == "favorite_car_color"


@pytest.mark.asyncio
async def test_llm_feature_update_handles_model_api_error(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    from memmachine.common.data_types import ExternalServiceAPIError

    # Given an LLM that raises API error
    magic_mock_llm_model.generate_response.side_effect = ExternalServiceAPIError(
        "API timeout"
    )

    with pytest.raises(
        Exception, match="Model Error when processing semantic features"
    ):
        await llm_feature_update(
            features=basic_features,
            message_content="I like blue cars",
            model=magic_mock_llm_model,
            update_prompt="Update features",
        )


@pytest.mark.asyncio
async def test_llm_feature_update_handles_value_error(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM that raises ValueError
    magic_mock_llm_model.generate_response.side_effect = ValueError("Invalid input")

    with pytest.raises(
        Exception, match="Model Error when processing semantic features"
    ):
        await llm_feature_update(
            features=basic_features,
            message_content="I like blue cars",
            model=magic_mock_llm_model,
            update_prompt="Update features",
        )


@pytest.mark.asyncio
async def test_llm_feature_update_handles_malformed_json(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM that returns invalid JSON
    magic_mock_llm_model.generate_response.return_value = (
        "{invalid json here}",
        None,
    )

    with pytest.raises(
        ExceptionGroup, match="Unable to load language model output as JSON"
    ):
        await llm_feature_update(
            features=basic_features,
            message_content="I like blue cars",
            model=magic_mock_llm_model,
            update_prompt="Update features",
        )


@pytest.mark.asyncio
async def test_llm_consolidate_features_handles_non_dict_response(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM that returns a list instead of dict
    magic_mock_llm_model.generate_response.return_value = ("[]", None)

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    # Expect None with warning logged
    assert result is None


@pytest.mark.asyncio
async def test_llm_consolidate_features_handles_invalid_structure(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM response with wrong structure
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "wrong_key": [],
            "another_wrong_key": []
        }
        """,
        None,
    )

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    # Expect None with warning logged
    assert result is None


@pytest.mark.asyncio
async def test_llm_consolidate_features_filters_invalid_memories(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM response with mixed valid and invalid memories
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "keep_memories": [1, 2],
            "consolidated_memories": [
                {
                    "type": "Profile",
                    "tag": "food",
                    "feature": "favorite_pizza",
                    "value": "pepperoni"
                },
                {
                    "missing_required_fields": "oops"
                },
                {
                    "type": "Profile",
                    "tag": "food",
                    "feature": "favorite_drink",
                    "value": "water"
                }
            ]
        }
        """,
        None,
    )

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    # Expect only valid memories to be included
    assert result is not None
    assert result.keep_memories == [1, 2]
    assert len(result.consolidated_memories) == 2
    assert result.consolidated_memories[0].feature == "favorite_pizza"
    assert result.consolidated_memories[1].feature == "favorite_drink"


@pytest.mark.asyncio
async def test_llm_consolidate_features_handles_invalid_keep_memories(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM response with non-integer keep_memories
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "keep_memories": [1, "not_an_int", 3, "also_invalid"],
            "consolidated_memories": []
        }
        """,
        None,
    )

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    # Expect only valid integers to be kept
    assert result is not None
    assert result.keep_memories == [1, 3]
    assert result.consolidated_memories == []


@pytest.mark.asyncio
async def test_llm_consolidate_features_handles_non_list_keep_memories(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM response with keep_memories as non-list
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "keep_memories": "not a list",
            "consolidated_memories": []
        }
        """,
        None,
    )

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    # Expect empty list with warning logged
    assert result is not None
    assert result.keep_memories == []
    assert result.consolidated_memories == []


@pytest.mark.asyncio
async def test_llm_consolidate_features_handles_non_list_consolidated_memories(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM response with consolidated_memories as non-list
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "keep_memories": [],
            "consolidated_memories": "not a list"
        }
        """,
        None,
    )

    result = await llm_consolidate_features(
        features=basic_features,
        model=magic_mock_llm_model,
        consolidate_prompt="Consolidate features",
    )

    # Expect empty list with warning logged
    assert result is not None
    assert result.keep_memories == []
    assert result.consolidated_memories == []


@pytest.mark.asyncio
async def test_llm_feature_update_with_multiple_valid_commands(
    magic_mock_llm_model: LanguageModel,
    basic_features: list[SemanticFeature],
):
    # Given an LLM response with multiple valid commands
    magic_mock_llm_model.generate_response.return_value = (
        """
        {
            "0": {
                "command": "add",
                "tag": "car",
                "feature": "favorite_car",
                "value": "Tesla"
            },
            "1": {
                "command": "delete",
                "tag": "food",
                "feature": "favorite_pizza",
                "value": ""
            },
            "2": {
                "command": "add",
                "tag": "music",
                "feature": "favorite_genre",
                "value": "rock"
            }
        }
        """,
        None,
    )

    commands = await llm_feature_update(
        features=basic_features,
        message_content="I like Tesla cars and rock music",
        model=magic_mock_llm_model,
        update_prompt="Update features",
    )

    # Expect all three commands to be processed
    assert len(commands) == 3
    assert commands[0].command == "add"
    assert commands[0].feature == "favorite_car"
    assert commands[1].command == "delete"
    assert commands[1].feature == "favorite_pizza"
    assert commands[2].command == "add"
    assert commands[2].feature == "favorite_genre"
