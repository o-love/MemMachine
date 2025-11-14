from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from memmachine.common.data_types import JSONValue

class ContentType(Enum):
    """Enumeration for the type of content within an Episode."""

    STRING = "string"
    # Other content types like 'vector', 'image' could be added here.


class EpisodeType(Enum):
    """Enumeration for the type of an Episode."""

    MESSAGE = "message"
    # Other episode types like 'thought', 'action' could be added here.


@dataclass(kw_only=True)
class Episode:
    """
    Represents a single, atomic event or piece of data in the memory system.
    `kw_only=True` enforces that all fields must be specified as keyword
    arguments during instantiation, improving clarity.
    """

    uuid: UUID
    """A unique identifier (UUID) for the episode."""
    sequence_num: int
    """Sequence number of the Episode"""
    session_key: str
    """The identifier for the session to which this episode belongs."""
    episode_type: EpisodeType
    """
    A string indicating the type of the episode (e.g., 'message', 'thought',
    'action').
    """
    content_type: ContentType
    """The type of the data stored in the 'content' field."""
    content: Any
    """The actual data of the episode, which can be of any type."""
    timestamp: datetime
    """The date and time when the episode occurred."""
    producer_id: str
    """The identifier of the user or agent that created this episode."""
    producer_role: str
    """The role of the producer (e.g., 'HR', 'agent', 'engineer')."""
    produced_for_id: str | None = None
    """The identifier of the intended recipient, if any."""
    user_metadata: JSONValue = None
    """
    A dictionary for any additional, user-defined metadata in a
    JSON-compatible format."""
