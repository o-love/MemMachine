"""Unit tests for the ProfileMemory class."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memmachine.profile_memory.profile_memory import ProfileMemory

# Since we are not using a test class, we'll use pytest's features.
# The 'asyncio' marker is used to run tests with an asyncio event loop.
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_dbconfig():
    """Provides a mock configuration dictionary."""
    return {
    }

@pytest.fixture
def mock_model():
    """Provides a mock model instance."""
    model = MagicMock()
    return model

@pytest.fixture
def mock_embedding():
    """Provides a mock embedding instance."""
    embedding = MagicMock()
    return embedding

@pytest.fixture
def mock_prompt():
    """Provides a mock prompt instance."""
    prompt = MagicMock()
    return prompt

@pytest.fixture
def profile_memory_instance(mock_model, mock_embedding, mock_dbconfig, mock_prompt):
    with patch("memmachine.profile_memory.storage.asyncpg_profile.AsyncPgProfileStorage") as MockProfileStorage:
        yield ProfileMemory(
            mock_model,
            mock_embedding,
            mock_dbconfig,
            mock_prompt,
        )

async def test_profile_memory_initialization(profile_memory_instance):