"""Entry point of the Desktop extension: run emode-mcp from the bundled libraries."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from emodeconnection.mcp import main  # noqa: E402

main()
