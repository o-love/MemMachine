from __future__ import annotations

from pathlib import Path

import pytest
from postgres_schema_manager import initialize_postgres_database

from .helpers import apply_sql_file, assert_modern_schema, engine_for_config, run_cli

ORIGINAL_SCHEMA_PATH = Path(__file__).with_name("original_schema.sql")
pytestmark = pytest.mark.integration


def test_initialize_postgres_database(db_config_factory) -> None:
    config = db_config_factory("schema_upgrader_init")
    initialize_postgres_database(config)

    with engine_for_config(config) as engine:
        assert_modern_schema(engine)


def test_upgrade_from_original_schema(db_config_factory) -> None:
    config = db_config_factory("schema_upgrader_from_original")

    with engine_for_config(config) as engine:
        apply_sql_file(engine, ORIGINAL_SCHEMA_PATH)

    initialize_postgres_database(config)

    with engine_for_config(config) as engine:
        assert_modern_schema(engine)


def test_initialize_idempotent(db_config_factory) -> None:
    config = db_config_factory("schema_upgrader_idempotent")

    initialize_postgres_database(config)
    initialize_postgres_database(config)

    with engine_for_config(config) as engine:
        assert_modern_schema(engine)


def test_cli_from_args_initialization(db_config_factory) -> None:
    config = db_config_factory("schema_upgrader_cli_args")

    run_cli(
        [
            "semantic-pg",
            "from-args",
            "--host",
            str(config["host"]),
            "--port",
            str(config["port"]),
            "--user",
            str(config["user"]),
            "--password",
            str(config["password"]),
            "--database",
            str(config["database"]),
        ]
    )

    with engine_for_config(config) as engine:
        assert_modern_schema(engine)


def test_cli_from_env_upgrade_original_schema(db_config_factory) -> None:
    config = db_config_factory("schema_upgrader_cli_env")

    with engine_for_config(config) as engine:
        apply_sql_file(engine, ORIGINAL_SCHEMA_PATH)

    run_cli(
        ["semantic-pg", "from-env"],
        env={
            "POSTGRES_HOST": str(config["host"]),
            "POSTGRES_PORT": str(config["port"]),
            "POSTGRES_USER": str(config["user"]),
            "POSTGRES_PASSWORD": str(config["password"]),
            "POSTGRES_DB": str(config["database"]),
        },
    )

    with engine_for_config(config) as engine:
        assert_modern_schema(engine)
