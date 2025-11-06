"""Alembic environment for semantic storage migrations."""

from __future__ import annotations

from alembic import context

from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    BaseSemanticStorage,
)


def _get_connection():
    connection = context.config.attributes.get("connection")
    if connection is None:
        raise RuntimeError(
            "Alembic migration requires an active SQLAlchemy connection."
        )
    return connection


def run_migrations_online() -> None:
    connection = _get_connection()
    context.configure(
        connection=connection,
        target_metadata=BaseSemanticStorage.metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    raise RuntimeError("Offline migrations are not supported in this environment.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
