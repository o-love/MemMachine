import asyncio
import datetime
import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from memmachine.common.data_types import ExternalServiceAPIError
from memmachine.common.language_model import LanguageModel

logger = logging.getLogger(__name__)


class SemanticMemory(BaseModel):
    class Metadata(BaseModel):
        citations: list[int] | None = None
        id: int | None = None

    tag: str
    feature: str
    value: str
    metadata: Metadata


class SemanticUpdateTracker:
    """Tracks feature update activity for a feature set.
    When a message is added to a feature set, this class keeps track of how many
    messages have been sent and when the first message was sent.
    This is used to determine when to trigger feature updates based
    on message count and time intervals.
    """

    def __init__(self, set_id: str, message_limit: int, time_limit_sec: float):
        self._set_id = set_id
        self._message_limit: int = message_limit
        self._time_limit: float = time_limit_sec
        self._message_count: int = 0
        self._first_updated: datetime.datetime | None = None

    def mark_update(self):
        """Marks that a new message has been added to the feature set.
        Increments the message count and sets the first updated time
        if this is the first message.
        """
        self._message_count += 1
        if self._first_updated is None:
            self._first_updated = datetime.datetime.now()

    def _seconds_from_first_update(self) -> float | None:
        """Returns the number of seconds since the first message was sent.
        If no messages have been sent, returns None.
        """
        if self._first_updated is None:
            return None
        delta = datetime.datetime.now() - self._first_updated
        return delta.total_seconds()

    def reset(self):
        """Resets the tracker state.
        Clears the message count and first updated time.
        """
        self._message_count = 0
        self._first_updated = None

    def should_update(self) -> bool:
        """Determines if a feature update should be triggered.
        A feature update is triggered if either the message count
        exceeds the limit or the time since the first message exceeds
        the time limit.

        Returns:
            bool: True if a feature update should be triggered, False otherwise.
        """
        if self._message_count == 0:
            return False
        elapsed = self._seconds_from_first_update()
        exceed_time_limit = elapsed is not None and elapsed >= self._time_limit
        exceed_msg_limit = self._message_count >= self._message_limit
        return exceed_time_limit or exceed_msg_limit


class SemanticUpdateTrackerManager:
    """Manages SemanticUpdateTracker instances for multiple feature sets."""

    def __init__(self, message_limit: int, time_limit_sec: float):
        self._trackers: dict[str, SemanticUpdateTracker] = {}
        self._trackers_lock = asyncio.Lock()
        self._message_limit = message_limit
        self._time_limit_sec = time_limit_sec

    def _new_tracker(self, set_id: str) -> SemanticUpdateTracker:
        return SemanticUpdateTracker(
            set_id=set_id,
            message_limit=self._message_limit,
            time_limit_sec=self._time_limit_sec,
        )

    async def mark_update(self, set_id: str):
        """Marks that a new message has been assigned to a feature set.
        Creates a new tracker if one does not exist for the feature set.
        """
        async with self._trackers_lock:
            if set_id not in self._trackers:
                self._trackers[set_id] = self._new_tracker(set_id)
            self._trackers[set_id].mark_update()

    async def get_sets_to_update(self) -> list[str]:
        """Returns a list of sets whose features need to be updated.
        A feature update is needed if the set's tracker indicates
        that an update should be triggered.
        """
        async with self._trackers_lock:
            ret = []
            for set, tracker in self._trackers.items():
                if tracker.should_update():
                    ret.append(set)
                    tracker.reset()
            return ret


class SemanticCommand(BaseModel):
    command: str
    feature: str
    tag: str
    value: str | None = None


def _process_commands(commands) -> list[SemanticCommand]:
    valid_commands = []
    for command in commands:
        if not isinstance(command, dict):
            logger.warning(
                "AI response format incorrect: "
                "expected feature update command to be dict, got %s %s",
                type(command).__name__,
                command,
            )
            continue

        try:
            pydantic_command = SemanticCommand(**command)
        except Exception as e:
            logger.warning(
                "AI response format incorrect: unable to parse feature update command %s, error %s",
                command,
                str(e),
            )
            continue

        valid_commands.append(pydantic_command)

    return valid_commands


async def _llm_think(
    model: LanguageModel,
    system_prompt: str,
    user_prompt: str,
):
    try:
        response_text, _ = await model.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (ExternalServiceAPIError, ValueError, RuntimeError) as e:
        logger.error("Model Error when processing semantic features: %s", str(e))
        return

    # Get thinking and JSON from language model response.
    thinking, _, response_json = response_text.removeprefix("<think>").rpartition(
        "</think>"
    )
    thinking = thinking.strip()

    try:
        response = json.loads(response_json)
    except ValueError as e:
        raise ValueError(
            "Unable to load language model output '%s' as JSON, Error %s",
            str(response_json),
            e,
        )

    return thinking, response


async def llm_feature_update(
    features,
    message_content: str,
    model: LanguageModel,
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
        feature_set=str(features),
        message_content=message_content,
    )

    try:
        thinking, raw_commands = await _llm_think(
            model=model,
            system_prompt=update_prompt,
            user_prompt=user_prompt,
        )
    except ValueError:
        return []

    logger.info(
        "PROFILE MEMORY INGESTOR",
        extra={
            "queries_to_ingest": message_content,
            "thoughts": thinking,
            "outputs": raw_commands,
        },
    )

    # This should probably just be a list of commands
    # instead of a dictionary mapping
    # from integers in strings (not even bare ints!)
    # to commands.
    # TODO: Consider improving this design in a breaking change.
    if not isinstance(raw_commands, dict):
        logger.warning(
            "AI response format incorrect: expected dict, got %s %s",
            type(raw_commands).__name__,
            raw_commands,
        )
        return []

    return _process_commands(raw_commands.values())


class SemanticConsolidateMemoryRes(BaseModel):
    consolidate_memories: list[SemanticMemory]
    keep_memories: list[int] | None

    @field_validator("consolidate_memories", mode="before")
    @classmethod
    def _filter_and_validate_memories(cls, v: Any) -> list[SemanticMemory]:
        if v is None:
            return []
        if not isinstance(v, list):
            logger.warning(
                "AI response format incorrect: 'consolidate_memories' not a list, got %s %s",
                type(v).__name__,
                v,
            )
            return []

        cleaned: list[SemanticMemory] = []
        for i, item in enumerate(v):
            try:
                cleaned.append(SemanticMemory.model_validate(item))
            except ValidationError as e:
                logger.warning(
                    "Dropping invalid memory at index %d. Error: %s. Item: %r",
                    i,
                    e.errors(),
                    item,
                )
        return cleaned

    @field_validator("keep_memories", mode="before")
    @classmethod
    def _normalize_keep_memories(cls, v: Any) -> list[int] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            logger.warning(
                "AI response format incorrect: 'keep_memories' not a list, got %s %s",
                type(v).__name__,
                v,
            )
            return []
        ints: list[int] = []
        for i, x in enumerate(v):
            try:
                ints.append(int(x))
            except Exception:
                logger.warning(
                    "Dropping invalid keep_memories[%d]=%r (not coercible to int)", i, x
                )
        return ints


async def llm_consolidate_features(
    memories: list[SemanticMemory],
    model: LanguageModel,
    consolidate_prompt: str,
) -> SemanticConsolidateMemoryRes | None:
    try:
        thinking, updated_feature_entries = await _llm_think(
            model=model,
            system_prompt=consolidate_prompt,
            user_prompt=json.dumps(memories),
        )
    except ValueError as e:
        logger.exception("Unable to consolidate features from LLM")
        raise e

    logger.info(
        "PROFILE MEMORY CONSOLIDATOR",
        extra={
            "receives": memories,
            "thoughts": thinking,
            "outputs": updated_feature_entries,
        },
    )

    if not isinstance(updated_feature_entries, dict):
        logger.warning(
            "AI response format incorrect: expected dict, got %s %s",
            type(updated_feature_entries).__name__,
            updated_feature_entries,
        )
        return None

    return SemanticConsolidateMemoryRes.model_validate(updated_feature_entries)
