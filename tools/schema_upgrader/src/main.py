import argparse

from postgres_schema_manager import postgres_parser


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade schema of a MemMachine database"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    postgres_parser(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        if args.func.__code__.co_argcount > 0:
            args.func(args)
        else:
            args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
