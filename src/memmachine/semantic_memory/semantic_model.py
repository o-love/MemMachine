from datetime import datetime
from typing import Optional, Tuple

from pydantic import BaseModel


class SemanticFeature(BaseModel):
    class Metadata(BaseModel):
        citations: Optional[list[int]] = None
        id: Optional[int] = None

    set_id: Optional[str] = None
    type: str
    tag: str
    feature: str
    value: str
    metadata: Metadata = Metadata()

    @staticmethod
    def group_features(
        features: list["SemanticFeature"],
    ) -> dict[Tuple[str, str, str], list["SemanticFeature"]]:
        grouped_features: dict[Tuple[str, str, str], list[SemanticFeature]] = {}

        for f in features:
            key = (f.type, f.tag, f.feature)

            if key not in grouped_features:
                grouped_features[key] = []

            grouped_features[key].append(f)

        return grouped_features


class SemanticCommand(BaseModel):
    command: str
    feature: str
    tag: str
    value: str


class SemanticHistory(BaseModel):
    class Metadata(BaseModel):
        id: Optional[int] = None

    content: str
    set_id: str
    created_at: datetime

    ingested: bool = False

    metadata: Metadata = Metadata()
