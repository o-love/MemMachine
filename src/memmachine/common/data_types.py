"""Common data types for MemMachine."""

from enum import Enum

from pydantic import JsonValue

FilterablePropertyValue = bool | int | str
JSONValue = JsonValue


class SimilarityMetric(Enum):
    """Similarity metrics supported by embedding operations."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class ExternalServiceAPIError(Exception):
    """Raised when an API error occurs for an external service."""
