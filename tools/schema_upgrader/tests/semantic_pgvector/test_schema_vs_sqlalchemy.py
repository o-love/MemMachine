from __future__ import annotations

from collections.abc import Mapping

import pytest
from postgres_schema_manager import initialize_postgres_database
from sqlalchemy import inspect
from sqlalchemy.engine import Inspector

from memmachine.semantic_memory.storage.sqlalchemy_pgvector_semantic import (
    BaseSemanticStorage,
)

from .helpers import engine_for_config

pytestmark = pytest.mark.integration


@pytest.fixture
def migrated_engine(db_config_factory):
    config = db_config_factory("schema_upgrader_migrated")
    initialize_postgres_database(config)
    with engine_for_config(config) as engine:
        yield engine


@pytest.fixture
def sqlalchemy_engine(db_config_factory):
    config = db_config_factory("schema_upgrader_sqlalchemy")
    with engine_for_config(config) as engine:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        BaseSemanticStorage.metadata.create_all(engine)
        yield engine


def test_schemas_match(migrated_engine, sqlalchemy_engine):
    migrated_inspector = inspect(migrated_engine)
    sqlalchemy_inspector = inspect(sqlalchemy_engine)

    migrated_tables = set(migrated_inspector.get_table_names())
    sqlalchemy_tables = set(sqlalchemy_inspector.get_table_names())

    assert "alembic_version" in migrated_tables, "alembic_version table is missing"
    migrated_tables_without_version = migrated_tables - {"alembic_version"}
    assert migrated_tables_without_version == sqlalchemy_tables, (
        f"Table mismatch. Migrated only: {sorted(migrated_tables_without_version - sqlalchemy_tables)}; "
        f"SQLAlchemy only: {sorted(sqlalchemy_tables - migrated_tables_without_version)}"
    )

    for table_name in sorted(sqlalchemy_tables):
        migrated_columns = get_table_columns(migrated_inspector, table_name)
        sqlalchemy_columns = get_table_columns(sqlalchemy_inspector, table_name)

        assert set(migrated_columns) == set(sqlalchemy_columns), (
            f"Column mismatch in {table_name}. "
            f"Migrated only: {sorted(set(migrated_columns) - set(sqlalchemy_columns))}; "
            f"SQLAlchemy only: {sorted(set(sqlalchemy_columns) - set(migrated_columns))}"
        )

        for column in sqlalchemy_columns:
            migrated_meta = migrated_columns[column]
            sqlalchemy_meta = sqlalchemy_columns[column]

            assert normalize_type(migrated_meta["type"]) == normalize_type(
                sqlalchemy_meta["type"]
            ), f"Type mismatch for {table_name}.{column}"

            if not sqlalchemy_meta["nullable"]:
                assert not migrated_meta["nullable"], (
                    f"{table_name}.{column} should be NOT NULL "
                    "to match the SQLAlchemy schema"
                )

        migrated_index_columns = get_index_column_sets(migrated_inspector, table_name)
        sqlalchemy_index_columns = get_index_column_sets(
            sqlalchemy_inspector, table_name
        )
        assert migrated_index_columns == sqlalchemy_index_columns, (
            f"Index column mismatch in {table_name}. "
            f"Migrated only: {sorted(migrated_index_columns - sqlalchemy_index_columns)}; "
            f"SQLAlchemy only: {sorted(sqlalchemy_index_columns - migrated_index_columns)}"
        )


def get_table_columns(inspector: Inspector, table_name: str) -> Mapping[str, dict]:
    columns = inspector.get_columns(table_name)
    return {
        column["name"]: {
            "type": str(column["type"]),
            "nullable": column["nullable"],
        }
        for column in columns
    }


def get_index_column_sets(
    inspector: Inspector, table_name: str
) -> set[tuple[str, ...]]:
    indexes = inspector.get_indexes(table_name)
    return {
        tuple(index["column_names"]) for index in indexes if index.get("column_names")
    }


def normalize_type(type_name: str) -> str:
    normalized = type_name.upper()
    normalized = normalized.replace("CHARACTER VARYING", "VARCHAR")
    normalized = normalized.replace("TIMESTAMP WITHOUT TIME ZONE", "TIMESTAMP")
    if normalized.startswith("VARCHAR") or normalized == "TEXT":
        return "TEXT"
    return normalized
