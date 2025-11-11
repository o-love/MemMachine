from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, InstanceOf

from memmachine.common.embedder import Embedder
from memmachine.common.language_model import LanguageModel


class SemanticCommandType(Enum):
    """Semantic memory actions that can be applied to a feature."""

    ADD = "add"
    DELETE = "delete"


class SemanticCommand(BaseModel):
    """Normalized instruction emitted by the LLM to mutate semantic features."""

    command: SemanticCommandType
    feature: str
    tag: str
    value: str


class HistoryMessage(BaseModel):
    """Conversation message stored in history together with persistence metadata."""

    class Metadata(BaseModel):
        """Optional storage details for a history message (id, provider-specific info)."""

        id: int | None = None
        other: dict[str, Any] | None = None

    content: str
    created_at: datetime

    metadata: Metadata = Metadata()


@dataclass
class SemanticPrompt:
    """Pair of prompt templates driving update and consolidation LLM calls."""

    update_prompt: str
    consolidation_prompt: str

    @staticmethod
    def load_from_module(prompt_module: ModuleType):
        update_prompt = getattr(prompt_module, "UPDATE_PROMPT", "")
        consolidation_prompt = getattr(prompt_module, "CONSOLIDATION_PROMPT", "")

        return SemanticPrompt(
            update_prompt=update_prompt,
            consolidation_prompt=consolidation_prompt,
        )


class SemanticFeature(BaseModel):
    """Semantic memory entry composed of category, tag, feature name, and textual value."""

    class Metadata(BaseModel):
        """Storage metadata for a semantic feature, including id and citations."""

        citations: list[HistoryMessage] | None = None
        id: int | None = None
        other: dict[str, Any] | None = None

    set_id: str | None = None
    category: str
    tag: str
    feature_name: str
    value: str
    metadata: Metadata = Metadata()

    @staticmethod
    def group_features(
        features: list["SemanticFeature"],
    ) -> dict[tuple[str, str, str], list["SemanticFeature"]]:
        grouped_features: dict[tuple[str, str, str], list[SemanticFeature]] = {}

        for f in features:
            key = (f.category, f.tag, f.feature_name)

            if key not in grouped_features:
                grouped_features[key] = []

            grouped_features[key].append(f)

        return grouped_features

    @staticmethod
    def group_features_by_tag(
        features: list["SemanticFeature"],
    ) -> dict[tuple[str, str], list["SemanticFeature"]]:
        grouped_features: dict[tuple[str, str], list[SemanticFeature]] = {}

        for f in features:
            key = (f.tag, f.feature_name)

            if key not in grouped_features:
                grouped_features[key] = []

            grouped_features[key].append(f)

        return grouped_features


class SemanticCategory(BaseModel):
    """Defines a semantic feature category, its allowed tags, and prompt strategy."""

    id: int | None = None

    name: str
    tags: set[str]
    prompt: SemanticPrompt


class Resources(BaseModel):
    """Resource bundle (embedder, language model, semantic categories) for a set_id."""

    embedder: InstanceOf[Embedder]
    language_model: InstanceOf[LanguageModel]
    semantic_categories: list[InstanceOf[SemanticCategory]]


@runtime_checkable
class ResourceRetriever(Protocol):
    """Protocol for locating the `Resources` bundle associated with a set_id."""

    def get_resources(self, set_id: str) -> Resources:
        raise NotImplementedError
