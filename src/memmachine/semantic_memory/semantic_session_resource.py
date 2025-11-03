from enum import Enum
from typing import runtime_checkable, Protocol, Final, Optional

from memmachine.semantic_memory.semantic_model import ResourceRetriever, Resources


class IsolationType(Enum):
    PROFILE = "profile"
    SESSION = "session"


ALL_MEMORY_TYPES: Final[list[IsolationType]] = [m for m in IsolationType]


@runtime_checkable
class SessionData(Protocol):
    def profile_id(self) -> str | None:
        raise NotImplementedError

    def session_id(self) -> str | None:
        raise NotImplementedError


@runtime_checkable
class SessionIdTypeChecker(Protocol):
    def set_id_type(self, _id: str) -> IsolationType:
        raise NotImplementedError


class SessionIdManager:
    def __init__(self):
        pass

    SESSION_ID_PREFIX: Final[str] = "mem_session_"
    PROFILE_ID_PREFIX: Final[str] = "mem_profile_"

    def generate_session_data(
        self,
        *,
        profile_id: Optional[str],
        session_id: Optional[str],
    ) -> SessionData:
        class _SessionDataImpl:
            def __init__(self, _profile_id, _session_id):
                self._profile_id = _profile_id
                self._session_id = _session_id

            def profile_id(self) -> Optional[str]:
                return self._profile_id

            def session_id(self) -> Optional[str]:
                return self._session_id

        return _SessionDataImpl(
            self.PROFILE_ID_PREFIX + profile_id,
            self.SESSION_ID_PREFIX + session_id,
        )

    def set_id_type(self, _id: str) -> IsolationType:
        if self.is_session_id(_id):
            return IsolationType.SESSION
        elif self.is_producer_id(_id):
            return IsolationType.PROFILE
        else:
            raise ValueError(f"Invalid id: {_id}")

    def is_session_id(self, _id: str) -> bool:
        return _id.startswith(self.SESSION_ID_PREFIX)

    def is_producer_id(self, _id: str) -> bool:
        return _id.startswith(self.PROFILE_ID_PREFIX)


class SessionResourceRetriever:
    def __init__(
        self,
        session_id_manager: SessionIdTypeChecker,
        default_resources: dict[IsolationType, Resources],
    ):
        self._session_id_manager = session_id_manager
        self._default_resources = default_resources

    def get_resources(self, set_id: str) -> Resources:
        custom_resources = self._get_set_id_resources(set_id)
        if custom_resources is not None:
            return custom_resources

        set_id_type = self._session_id_manager.set_id_type(set_id)

        return self._default_resources[set_id_type]

    @staticmethod
    def _get_set_id_resources(_: str) -> Resources | None:
        # TODO: Load set_id custom resources
        return None
