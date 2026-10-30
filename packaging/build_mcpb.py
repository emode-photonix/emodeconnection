"""Build the Claude Desktop extension, ``emode.mcpb``.

    python packaging/build_mcpb.py [output.mcpb]

An .mcpb file is a zip with ``manifest.json`` at its root. This one carries the
emodeconnection package and, unless ``--no-dependencies`` is given, its third
party dependencies installed with pip into ``server/lib``. Those include
compiled wheels, so build the bundle on the platform it will run on. Needs
Python 3.11 or later (tomllib).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
MCPB = Path(__file__).resolve().parent / "mcpb"


def _requirements() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return [*project["dependencies"], *project["optional-dependencies"]["mcp"]]


def _version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def build(output: Path | str, install_dependencies: bool = True) -> Path:
    """Write the bundle to ``output`` and return its path."""
    output = Path(output)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        lib = stage / "server" / "lib"
        lib.mkdir(parents=True)
        manifest = json.loads((MCPB / "manifest.json").read_text())
        manifest["version"] = _version()
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "server" / "main.py").write_text(
            (MCPB / "server" / "main.py").read_text()
        )
        if install_dependencies:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--target",
                    str(lib),
                    *_requirements(),
                ],
                check=True,
            )
        package = lib / "emodeconnection"
        for source in (ROOT / "emodeconnection").rglob("*"):
            if source.is_file() and "__pycache__" not in source.parts:
                target = package / source.relative_to(ROOT / "emodeconnection")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(stage).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default="emode.mcpb")
    parser.add_argument(
        "--no-dependencies",
        action="store_true",
        help="skip pip install of the dependencies",
    )
    args = parser.parse_args()
    print(build(args.output, install_dependencies=not args.no_dependencies))


if __name__ == "__main__":
    main()
