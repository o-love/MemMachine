from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import sqlalchemy
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

REQUIRED_TABLES = {
    "alembic_version",
    "citations",
    "feature",
    "history",
    "set_ingested_history",
}

EXPECTED_COLUMNS = {
    "feature": {
        "id",
        "set_id",
        "semantic_type_id",
        "tag_id",
        "feature",
        "value",
        "created_at",
        "updated_at",
        "embedding",
        "metadata",
    },
    "history": {
        "id",
        "content",
        "created_at",
        "metadata",
    },
    "citations": {
        "feature_id",
        "history_id",
    },
    "set_ingested_history": {
        "set_id",
        "history_id",
        "ingested",
    },
}

EXPECTED_INDEXES = {
    "feature": {
        "idx_feature_set_id",
        "idx_feature_set_id_semantic_type",
        "idx_feature_set_semantic_type_tag",
        "idx_feature_set_semantic_type_tag_feature",
    },
}


SCHEMA_UPGRADER_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = SCHEMA_UPGRADER_ROOT / "src"


def create_engine_from_config(config: dict[str, str | int]) -> Engine:
    return sqlalchemy.create_engine(
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )


@contextmanager
def engine_for_config(config: dict[str, str | int]) -> Engine:
    engine = create_engine_from_config(config)
    try:
        yield engine
    finally:
        engine.dispose()


def run_cli(args: Sequence[str], env: Mapping[str, str] | None = None) -> None:
    cmd = [sys.executable, "-m", "main", *args]

    merged_env = os.environ.copy()
    pythonpath = merged_env.get("PYTHONPATH")
    src_path = str(SRC_DIR)
    if pythonpath:
        merged_env["PYTHONPATH"] = f"{src_path}{os.pathsep}{pythonpath}"
    else:
        merged_env["PYTHONPATH"] = src_path

    if env:
        merged_env.update(env)

    subprocess.run(
        cmd,
        check=True,
        cwd=SCHEMA_UPGRADER_ROOT,
        env=merged_env,
    )


def assert_modern_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    missing_tables = REQUIRED_TABLES - tables
    assert not missing_tables, f"Missing tables: {sorted(missing_tables)}"

    with engine.connect() as conn:
        assert _record_exists(
            conn,
            text("SELECT 1 FROM pg_extension WHERE extname = :name"),
            name="vector",
        ), "pgvector extension not installed"
        assert _record_exists(
            conn,
            text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :name"),
            name="metadata",
        ), "metadata schema not created"
        assert _record_exists(
            conn,
            text(
                "SELECT 1 FROM pg_tables WHERE schemaname = 'metadata' "
                "AND tablename = 'migration_tracker'"
            ),
        ), "metadata.migration_tracker table missing"

    for table, columns in EXPECTED_COLUMNS.items():
        actual = {column["name"] for column in inspector.get_columns(table)}
        missing = columns - actual
        assert not missing, f"{table} missing columns: {sorted(missing)}"

    for table, indexes in EXPECTED_INDEXES.items():
        actual = {index["name"] for index in inspector.get_indexes(table)}
        missing = indexes - actual
        assert not missing, f"{table} missing indexes: {sorted(missing)}"


def apply_sql_file(engine: Engine, sql_path: Path) -> None:
    statements = _read_sql_statements(sql_path)
    if not statements:
        return

    with engine.begin() as conn:
        for statement in statements:
            conn.exec_driver_sql(statement)


def _record_exists(connection: Connection, statement, **params) -> bool:
    result = connection.execute(statement, params)
    return result.scalar() is not None


def _read_sql_statements(sql_path: Path) -> list[str]:
    content = sql_path.read_text()
    statements: list[str] = []
    buffer: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip().rstrip(";")
            if statement:
                statements.append(statement)
            buffer.clear()

    trailing = "\n".join(buffer).strip().rstrip(";")
    if trailing:
        statements.append(trailing)

    return statements
