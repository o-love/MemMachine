from typing import Optional

from pydantic import BaseModel


class SemanticFeature(BaseModel):
    class Metadata(BaseModel):
        citations: list[int] | None = None
        id: int | None = None

    set_id: Optional[str] = None
    type: str
    tag: str
    feature: str
    value: str
    metadata: Metadata = Metadata()


class SemanticCommand(BaseModel):
    command: str
    feature: str
    tag: str
    value: str
