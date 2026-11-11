"""Simulation .mat files are no longer read (EMode 1.0.9).

`open_file`, `get` and `inspect` read `.eph` files only. A `.mat` name, or a
`.mat` file where the `.eph` was looked for, raises a `FileError` that says so
and points to `.eph`. The client no longer imports scipy.io.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys

import pytest

from emode import get, inspect, open_file
from emode.types import FileError


def _says_mat_was_removed(error: Exception) -> None:
    text = str(error)
    assert ".mat" in text
    assert "1.0.9" in text
    assert ".eph" in text


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_mat_name_raises_file_error(in_tmp):
    (in_tmp / "old.mat").write_bytes(b"not read")
    with pytest.raises(FileError) as caught:
        open_file("old.mat")
    _says_mat_was_removed(caught.value)


def test_only_a_mat_file_raises_file_error(in_tmp):
    (in_tmp / "legacy.mat").write_bytes(b"not read")
    with pytest.raises(FileError) as caught:
        open_file("legacy")
    _says_mat_was_removed(caught.value)


def test_get_and_inspect_say_mat_was_removed(in_tmp):
    (in_tmp / "legacy.mat").write_bytes(b"not read")
    with pytest.raises(FileError):
        get("n_eff", sim="legacy")
    with pytest.raises(FileError):
        inspect(sim="legacy.mat")


def test_a_missing_file_still_raises_file_not_found(in_tmp):
    with pytest.raises(FileNotFoundError):
        open_file("nothing")


def test_an_eph_file_is_still_read(in_tmp):
    data = {"EMode_simulation_file": "closed", "n_eff": [2.4]}
    with open(in_tmp / "sim.eph", "wb") as fl:
        pickle.dump(data, fl)
    assert open_file("sim") == data
    assert open_file("sim.eph") == data
    assert get("n_eff", sim="sim") == [2.4]
    assert inspect(sim="sim") == ["n_eff"]


def test_importing_emode_imports_no_scipy_io():
    code = "import sys, emode; print('scipy.io' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "False"
