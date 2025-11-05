from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict

import pytest
import sqlalchemy
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer

SCHEMA_UPGRADER_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = SCHEMA_UPGRADER_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MEMMACHINE_SRC = SCHEMA_UPGRADER_ROOT.parent.parent / "src"
if MEMMACHINE_SRC.exists() and str(MEMMACHINE_SRC) not in sys.path:
    sys.path.insert(0, str(MEMMACHINE_SRC))


ConfigFactory = Callable[[str, bool], Dict[str, str | int]]


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="need --integration option to run")

    if not config.getoption("--integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def pg_container() -> "PostgresContainer":
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest.fixture
def db_config_factory(pg_container: "PostgresContainer") -> ConfigFactory:
    host = pg_container.get_container_host_ip()
    port = int(pg_container.get_exposed_port(5432))
    user = pg_container.username
    password = pg_container.password
    managed_databases: list[str] = []

    def _factory(database_name: str, recreate: bool = True) -> Dict[str, str | int]:
        if not database_name:
            msg = "database_name must be provided to db_config_factory"
            raise ValueError(msg)

        if recreate:
            admin_engine = sqlalchemy.create_engine(
                f"postgresql://{user}:{password}@{host}:{port}/postgres"
            )
            try:
                with admin_engine.connect() as conn:
                    conn.execution_options(isolation_level="AUTOCOMMIT")
                    conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}"'))
                    conn.execute(text(f'CREATE DATABASE "{database_name}"'))
            finally:
                admin_engine.dispose()

            if database_name not in managed_databases:
                managed_databases.append(database_name)

        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database_name,
        }

    yield _factory

    if managed_databases:
        admin_engine = sqlalchemy.create_engine(
            f"postgresql://{user}:{password}@{host}:{port}/postgres"
        )
        try:
            with admin_engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                for database_name in managed_databases:
                    conn.execute(text(f'DROP DATABASE IF EXISTS "{database_name}"'))
        finally:
            admin_engine.dispose()
