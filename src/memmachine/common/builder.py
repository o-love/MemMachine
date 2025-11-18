"""Base interfaces for building resources with declared dependencies."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T")


class Builder(ABC):
    """Abstract base class for constructing resources and their dependencies."""

    @staticmethod
    @abstractmethod
    def get_dependency_ids(name: str, config: dict[str, Any]) -> set[str]:
        """Return dependency IDs required for building the resource."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build(name: str, config: dict[str, Any], injections: dict[str, Any]) -> T:
        """Build the resource using the provided configuration and injections."""
        raise NotImplementedError
