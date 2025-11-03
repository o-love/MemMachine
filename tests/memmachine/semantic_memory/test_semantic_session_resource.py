from memmachine.semantic_memory.semantic_session_resource import (
    SessionIdManager,
    SessionIdTypeChecker,
)


def test_semantic_session_id_manager_is_instance():
    assert isinstance(SessionIdManager(), SessionIdTypeChecker)
