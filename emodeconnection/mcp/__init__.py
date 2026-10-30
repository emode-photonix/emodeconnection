"""A local MCP server for EMode, run as ``emode-mcp``.

It needs the optional extra: ``pip install "emodeconnection[mcp]"``. Importing
this package does not import the MCP SDK, so ``main`` can explain a missing
extra instead of failing with a bare ImportError.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the MCP server over stdio until the client disconnects."""
    try:
        from .server import create_server
    except ImportError as err:
        print(
            f"emode-mcp needs the MCP SDK ({err}). "
            'Install it with: pip install "emodeconnection[mcp]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from err
    create_server().run("stdio")
