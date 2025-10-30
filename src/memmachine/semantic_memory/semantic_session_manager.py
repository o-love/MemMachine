from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionData(Protocol):
    def producer_id(self) -> str | None:
        raise NotImplementedError

    def session_id(self) -> str | None:
        raise NotImplementedError


class SemanticSessionManager(ABC):
    @abstractmethod
    async def add_message(self, message: str, session_data: SessionData):
        raise NotImplementedError

    @abstractmethod
    async def search(self, message: str, session_data: SessionData):
        raise NotImplementedError

    @abstractmethod
    async def get_memories(self, session_data: SessionData):
        raise NotImplementedError


