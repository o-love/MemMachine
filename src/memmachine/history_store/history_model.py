from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from memmachine.common.data_types import JSONValue

EpisodeIdT = str


class ContentType(Enum):
    """Enumeration for the type of content within an Episode."""

    STRING = "string"
    # Other content types like 'vector', 'image' could be added here.


class EpisodeType(Enum):
    """Enumeration for the type of an Episode."""

    MESSAGE = "message"
    # Other episode types like 'thought', 'action' could be added here.


class Episode(BaseModel):
    """Conversation message stored in history together with persistence metadata."""

    uuid: EpisodeIdT | None = None
    content: str
    session_key: str

    created_at: datetime

    producer_id: str
    producer_role: str
    produced_for_id: str | None = None

    sequence_num: int | None = None

    episode_type: EpisodeType | None = None
    content_type: ContentType | None = None

    metadata: dict[str, JSONValue] | None = None
