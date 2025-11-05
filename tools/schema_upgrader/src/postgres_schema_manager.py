import os

from alembic import command
from alembic.config import Config


def initialize_postgres_database(config: dict[str, str | int]) -> None:
    """
    Setup the database using Alembic migrations.

    Args:
        config: Dictionary containing database connection parameters:
            - host: Database host
            - port: Database port
            - user: Database user
            - password: Database password
            - database: Database name
    """
    # Construct the database URL
    db_url = (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )

    # Get the path to alembic.ini (in tools/schema_upgrader/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir is tools/schema_upgrader/src, so we need to go up one level
    alembic_ini_path = os.path.join(os.path.dirname(script_dir), "alembic.ini")
    alembic_ini_path = os.path.abspath(alembic_ini_path)

    # Create Alembic config
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Run migrations to head (latest version)
    command.upgrade(alembic_cfg, "head")


def setup_postgres_config_from_env():
    config = {
        "host": os.getenv("POSTGRES_HOST", "postgres"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "database": os.getenv("POSTGRES_DB"),
    }

    initialize_postgres_database(config)


def setup_postgres_config_from_args(args):
    config = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
        "database": args.database,
    }

    initialize_postgres_database(config)


def postgres_parser(subparser):
    pg_parser = subparser.add_parser(
        "semantic-pg", help="Setup the semantic store database in postgres"
    )

    pg_sub = pg_parser.add_subparsers(dest="subcommand", required=True)

    pg_init = pg_sub.add_parser(
        "from-env",
        help="Initialize the semantic store database from environment variables",
    )
    pg_init.set_defaults(func=setup_postgres_config_from_env)

    pg_manual = pg_sub.add_parser(
        "from-args",
        help="Initialize the semantic store database from command line arguments",
    )
    pg_manual.add_argument("--host", required=True, help="Database host")
    pg_manual.add_argument(
        "--port", type=int, default=5432, help="Database port (default: 5432)"
    )
    pg_manual.add_argument("--user", required=True, help="Database user")
    pg_manual.add_argument("--password", required=True, help="Database password")
    pg_manual.add_argument("--database", required=True, help="Database name")
    pg_manual.set_defaults(func=setup_postgres_config_from_args)
