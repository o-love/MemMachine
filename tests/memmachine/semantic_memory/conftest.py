import pytest_asyncio

from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    SqlAlchemyPgVectorSemanticStorage,
)
from tests.memmachine.semantic_memory.storage.in_memory_semantic_storage import (
    InMemorySemanticStorage,
)


@pytest_asyncio.fixture
async def in_memory_profile_storage():
    store = InMemorySemanticStorage()
    await store.startup()
    yield store
    await store.cleanup()


# @pytest_asyncio.fixture
# async def unbacked_in_memory_profile_storage():
#     base = InMemorySemanticStorage()
#     storage = UnBackedCachedSemanticStorage(base)
#
#     await storage.startup()
#     yield storage
#     await storage.delete_all()
#     await storage.cleanup()


@pytest_asyncio.fixture
async def sqlalchemy_profile_storage(sqlalchemy_pg_engine):
    storage = SqlAlchemyPgVectorSemanticStorage(sqlalchemy_pg_engine)
    await storage.startup()
    yield storage
    await storage.delete_all()
    await storage.cleanup()


# @pytest_asyncio.fixture
# async def unbacked_sqlalchemy_profile_storage(sqlalchemy_pg_engine):
#     base = SqlAlchemyPgVectorSemanticStorage(sqlalchemy_pg_engine)
#     storage = UnBackedCachedSemanticStorage(base)
#
#     await storage.startup()
#     yield storage
#     await storage.delete_all()
#     await storage.cleanup()
