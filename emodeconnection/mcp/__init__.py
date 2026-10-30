"""A local MCP server for EMode, run as ``emode-mcp``.

It needs the optional extra: ``pip install "emodeconnection[mcp]"``. Importing
this package does not import the MCP SDK, so ``main`` can explain a missing
extra instead of failing with a bare ImportError.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the MCP server over stdio until the client disconnects.

    With ``--schemas``, print the tool definitions as JSON and exit, for
    clients that do not speak MCP.
    """
    parser = argparse.ArgumentParser(
        prog="emode-mcp", description="A local MCP server for EMode."
    )
    parser.add_argument(
        "--schemas",
        action="store_true",
        help="print the tool definitions as JSON and exit",
    )
    args = parser.parse_args(argv)
    try:
        from .server import create_server
    except ImportError as err:
        print(
            f"emode-mcp needs the MCP SDK ({err}). "
            'Install it with: pip install "emodeconnection[mcp]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from err
    server = create_server()
    if args.schemas:
        tools = asyncio.run(server.list_tools())
        definitions = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "output_schema": t.output_schema,
            }
            for t in tools
        ]
        print(json.dumps(definitions, indent=2))
        return
    server.run("stdio")
