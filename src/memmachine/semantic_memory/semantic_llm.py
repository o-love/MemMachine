import json
import logging

from pydantic import (
    BaseModel,
    InstanceOf,
    TypeAdapter,
    validate_call,
)

from memmachine.common.language_model import LanguageModel
from memmachine.semantic_memory.semantic_model import SemanticCommand, SemanticFeature

logger = logging.getLogger(__name__)


def _features_to_llm_format(
    features: list[SemanticFeature],
) -> dict[str, dict[str, str]]:
    structured_features: dict[str, dict[str, str]] = {}

    for feature in features:
        if feature.tag not in structured_features:
            structured_features[feature.tag] = {}

        structured_features[feature.tag][feature.feature] = feature.value

    return structured_features


class _SemanticFeatureUpdateRes(BaseModel):
    commands: list[SemanticCommand]


@validate_call
async def llm_feature_update(
    features: list[SemanticFeature],
    message_content: str,
    model: InstanceOf[LanguageModel],
    update_prompt: str,
) -> list[SemanticCommand]:
    user_prompt = (
        "The old feature set is provided below:\n"
        "<OLD_PROFILE>\n"
        "{feature_set}\n"
        "</OLD_PROFILE>\n"
        "\n"
        "The history is provided below:\n"
        "<HISTORY>\n"
        "{message_content}\n"
        "</HISTORY>\n"
    ).format(
        feature_set=json.dumps(_features_to_llm_format(features)),
        message_content=message_content,
    )

    parsed_output = await model.generate_parsed_response(
        system_prompt=update_prompt,
        user_prompt=user_prompt,
        output_format=_SemanticFeatureUpdateRes,
    )

    validated_output = TypeAdapter(_SemanticFeatureUpdateRes).validate_python(
        parsed_output
    )
    return validated_output.commands


class LLMReducedFeature(BaseModel):
    tag: str
    feature: str
    value: str


class SemanticConsolidateMemoryRes(BaseModel):
    consolidated_memories: list[LLMReducedFeature]
    keep_memories: list[int] | None


@validate_call
async def llm_consolidate_features(
    features: list[SemanticFeature],
    model: InstanceOf[LanguageModel],
    consolidate_prompt: str,
) -> SemanticConsolidateMemoryRes | None:
    parsed_output = await model.generate_parsed_response(
        system_prompt=consolidate_prompt,
        user_prompt=json.dumps(_features_to_llm_format(features)),
        output_format=SemanticConsolidateMemoryRes,
    )

    validated_output = TypeAdapter(SemanticConsolidateMemoryRes).validate_python(
        parsed_output
    )
    return validated_output
