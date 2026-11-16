"""A pathlib.Path argument goes over the wire as its string (emodeconnection#203)."""

from __future__ import annotations

import json
from pathlib import Path

from emode.types import serialize


def test_a_path_argument_serializes_as_a_string(tmp_path: Path) -> None:
    wire = json.loads(json.dumps(serialize({"function": "EM_init", "save_path": tmp_path})))
    assert wire["save_path"] == str(tmp_path)


def test_a_path_inside_a_list_serializes_as_a_string() -> None:
    wire = json.loads(json.dumps(serialize({"paths": [Path("a") / "b", "c"]})))
    assert wire["paths"] == [str(Path("a") / "b"), "c"]
