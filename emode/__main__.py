"""``python -m emode``: the command line of the client."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .emode import EMode
from .types import EModeError


def _login() -> int:
    """Sign in to EMode on the account page, and say how to where no browser can open."""
    try:
        session = EMode(simulation_name="login", save=False)
    except EModeError as error:
        print(error, file=sys.stderr)
        return 1
    session.close(save=False)
    print("EMode is signed in.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m emode", description="The EMode client.")
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.add_parser("login", help="sign in to EMode on the account page of the browser tab")
    args = parser.parse_args(argv)
    if args.command == "login":
        return _login()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
