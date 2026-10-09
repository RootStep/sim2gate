"""Command-line entry point. Placeholder until the first checks land."""

import argparse
import sys

from sim2gate import __version__

NOT_READY = (
    "sim2gate {version} is a pre-alpha placeholder; no diagnostics are implemented yet.\n"
    "Follow https://github.com/RootStep/sim2gate for the first release."
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sim2gate",
        description="Training-aware diagnostics for sim-trained robot policies.",
    )
    parser.add_argument("--version", action="version", version=f"sim2gate {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("preflight", help="diagnose a trained policy (not implemented yet)")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    print(NOT_READY.format(version=__version__), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
