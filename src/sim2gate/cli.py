"""Command-line entry point. `preflight` is a placeholder until the first checks land."""

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
    sp = sub.add_parser("signals", help="summarize sim2gate_signals.jsonl files from training")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--last", type=float, default=0.2, help="fraction of iterations at the end to average (default 0.2)")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "signals":
        import json
        from sim2gate.training.summary import summarize
        for f in args.files:
            print(json.dumps(summarize(f, args.last)))
        return 0

    print(NOT_READY.format(version=__version__), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
