import argparse
import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

script_dir = str(Path(__file__).parent)


def get_base() -> str:
    return open(f"{script_dir}/baseschema.sql", "r").read()


async def delete_data(
    database: str, host: str, port: str, user: str, password: str
):
    d: dict[str, str] = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }
    print(f"Deleteing tables in {d}")
    pool = await asyncpg.create_pool(init=register_vector, **d)
    table_records = await pool.fetch(
        """
        SELECT tablename FROM pg_catalog.pg_tables
        WHERE schemaname = 'public';
        """
    )
    tables = [table_record[0] for table_record in table_records]

    for table in tables:
        print(f"Dropping table: {table}")
        await pool.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')


async def sync_to(
    database: str, host: str, port: str, user: str, password: str
):
    d: dict[str, str] = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }
    print(f"syncing to: {{'host': '{host}', 'port': '{port}', 'user': '{user}', 'database': '{database}', 'password': '****'}}")
    connection = await asyncpg.connect(**d)
    await connection.execute(get_base())
    print("Re-initializing ...")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="memmachine-sync-profile-schema",
        description="sync latest schema to db. By default syncs to the cluster specified by the environment variables",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("POSTGRES_DB"),
        help="the default database name is read from the environment variable POSTGRES_DB",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("POSTGRES_HOST"),
        help="the default host is read from the environment variable POSTGRES_HOST",
    )
    parser.add_argument(
        "--port",
        default=os.getenv("POSTGRES_PORT"),
        help="the default port is read from the environment variable POSTGRES_PORT",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("POSTGRES_USER"),
        help="the default user is read from the environment variable POSTGRES_USER",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("POSTGRES_PASSWORD"),
        help="the defualt password is read from the environement variable POSTGRES_PASS",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="delete and recreate the database with new schema. Use the migration system if you care about data inside the old database.",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


async def main_async(args):
    if args.delete:
        await delete_data(
            args.database, args.host, args.port, args.user, args.password
        )
    await sync_to(
        args.database, args.host, args.port, args.user, args.password
    )


if __name__ == "__main__":
    main()
