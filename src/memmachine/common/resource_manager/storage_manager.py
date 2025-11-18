import asyncio
import logging
from asyncio import Lock
from typing import Self

from neo4j import AsyncGraphDatabase
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from memmachine.common.configuration.storage_conf import StorageConf
from memmachine.common.vector_graph_store import VectorGraphStore
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)

logger = logging.getLogger(__name__)


class StorageManager:
    def __init__(self, conf: StorageConf) -> None:
        self.conf = conf
        self.graph_stores: dict[str, VectorGraphStore] = {}
        self.sql_engines: dict[str, AsyncEngine] = {}

        self._lock = Lock()

    async def build_all(self, validate: bool = False) -> Self:
        """Build and verify all configured database connections."""
        async with self._lock:
            await asyncio.gather(
                self._build_neo4j(),
                self._build_sql_engines(),
            )
            if validate:
                await asyncio.gather(
                    self._validate_neo4j(), self._validate_sql_engines(),
                )
        return self

    async def close(self) -> None:
        """Close all database connections."""
        async with self._lock:
            tasks = []
            for name, store in self.graph_stores.items():

                def close_store() -> None:
                    try:
                        store.close()
                    except Exception as e:
                        logger.error(f"Error closing graph store '{name}': {e}")
                    store.close()

                tasks.append(store.close())
            for engine in self.sql_engines.values():
                tasks.append(engine.dispose())

            await asyncio.gather(*tasks)

    def get_vector_graph_store(self, name: str) -> VectorGraphStore:
        if name not in self.graph_stores:
            raise ValueError(f"Neo4J driver '{name}' not found.")
        return self.graph_stores[name]

    def get_sql_engine(self, name: str) -> AsyncEngine:
        if name not in self.sql_engines:
            raise ValueError(f"SQL connection '{name}' not found.")
        return self.sql_engines[name]

    async def _build_neo4j(self) -> None:
        for name, conf in self.conf.neo4j_confs.items():
            if name in self.graph_stores:
                continue
            neo4j_host = conf.host
            if "neo4j+s://" in neo4j_host:
                neo4j_uri = neo4j_host
            else:
                neo4j_port = conf.port
                neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

            driver = AsyncGraphDatabase.driver(
                neo4j_uri, auth=(conf.user, conf.password),
            )
            params = Neo4jVectorGraphStoreParams(
                driver=driver,
                force_exact_similarity_search=conf.force_exact_similarity_search,
            )
            self.graph_stores[name] = Neo4jVectorGraphStore(params)

    async def _validate_neo4j(self) -> None:
        for name, driver in self.conf.neo4j_confs.items():
            try:
                async with driver.session() as session:
                    result = await session.run("RETURN 1 AS ok")
                    record = await result.single()
                    if not record or record["ok"] != 1:
                        raise ConnectionError(
                            f"Verification failed for Neo4J config '{name}'",
                        )
            except Exception as e:
                await driver.close()
                raise ConnectionError(f"Neo4J config '{name}' failed verification: {e}")

    async def _build_sql_engines(self) -> None:
        schema = "sqlite+aiosqlite:///"
        for name, conf in self.conf.sqlite_confs.items():
            if name in self.sql_engines:
                continue
            db_url = conf.file_path
            if not conf.file_path.startswith(schema):
                db_url = f"sqlite+aiosqlite:///{db_url}"
            engine = create_async_engine(db_url, echo=False, future=True)
            self.sql_engines[name] = engine

    async def _validate_sql_engines(self) -> None:
        for name, engine in self.sql_engines.items():
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1;"))
                    row = result.fetchone()
                    if not row or row[0] != 1:
                        raise ConnectionError(
                            f"Verification failed for SQLite config '{name}'",
                        )
            except Exception as e:
                raise ConnectionError(
                    f"SQLite config '{name}' failed verification: {e}",
                )
