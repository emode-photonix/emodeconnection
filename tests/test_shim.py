"""The client's import package is `emode`, and `emodeconnection` is a shim over it (#121).

`pip install emode` installs the package. `pip install emodeconnection` installs
only the shim, which depends on `emode`, re-exports it, and warns once that the
name changed. The wire types must not care which name imported them.
"""

from __future__ import annotations

import importlib
import os
import pickle
import pkgutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Run `python -c` in a clean interpreter that finds this checkout's packages first.
_SHIM_SRC = ROOT / "packaging" / "emodeconnection" / "src"


def _python(code: str) -> subprocess.CompletedProcess[str]:
    env_path = f"{_SHIM_SRC}{os.pathsep}{ROOT}"
    return subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": env_path},
        timeout=120,
    )


def _submodules() -> list[str]:
    emode = importlib.import_module("emode")
    found = pkgutil.walk_packages(emode.__path__, prefix="emode.")
    return sorted(
        m.name.removeprefix("emode.") for m in found if not m.name.endswith("__main__")
    )


def test_emode_is_the_real_package_in_this_checkout():
    import emode

    assert Path(emode.__file__).resolve().is_relative_to(ROOT)
    assert emode.EMode.__module__ == "emode.emode"


def test_importing_emodeconnection_warns_exactly_once():
    result = _python(
        "import emodeconnection\n"
        "import emodeconnection.types\n"
        "import emodeconnection.smatrix\n"
        "import emodeconnection.emodeconnection\n"
        "import emode\n"
    )
    assert result.returncode == 0, result.stderr
    warnings = [
        line for line in result.stderr.splitlines() if "DeprecationWarning" in line
    ]
    assert len(warnings) == 1, result.stderr
    assert "pip install emode" in result.stderr
    assert "import emode" in result.stderr


def test_importing_emode_does_not_warn():
    result = _python("import emode\nimport emode.types\n")
    assert result.returncode == 0, result.stderr
    assert "Warning" not in result.stderr


def test_the_warning_points_at_the_importing_line():
    result = _python("import emodeconnection\n")
    assert "<string>:1" in result.stderr


@pytest.mark.parametrize("name", _submodules())
def test_every_submodule_is_the_same_module_under_both_names(name: str):
    new = importlib.import_module(f"emode.{name}")
    old = importlib.import_module(f"emodeconnection.{name}")
    assert old is new


def test_importing_through_the_shim_leaves_the_module_spec_alone():
    result = _python(
        "import warnings\n"
        'warnings.simplefilter("error")\n'
        'warnings.filterwarnings("ignore", message="The emodeconnection package")\n'
        "import emodeconnection.mcp.server, emodeconnection.types\n"
        "import emode.types, emode.mcp\n"
        'assert emode.types.__spec__.name == "emode.types", emode.types.__spec__\n'
        'assert emode.mcp.__spec__.name == "emode.mcp"\n'
        "import importlib\n"
        "importlib.reload(emode.types)\n"
    )
    assert result.returncode == 0, result.stderr


def test_the_launch_module_keeps_its_old_dotted_name():
    import emode.emode

    old = importlib.import_module("emodeconnection.emodeconnection")
    assert old is emode.emode


def test_every_top_level_name_is_the_same_object():
    import emode
    import emodeconnection

    assert set(emodeconnection.__all__) == set(emode.__all__)
    for name in emode.__all__:
        assert getattr(emodeconnection, name) is getattr(emode, name), name


def test_a_private_name_is_reachable_through_the_shim():
    import emode
    import emodeconnection

    assert emodeconnection.EModeLogin is emode.EModeLogin
    with pytest.raises(AttributeError):
        emodeconnection.not_a_name  # noqa: B018


def test_no_class_is_registered_twice_under_both_names():
    import emode.types
    import emodeconnection.types

    assert emodeconnection.types._TYPE_REGISTRY is emode.types._TYPE_REGISTRY
    classes = list(emode.types._TYPE_REGISTRY.values())
    assert len({id(cls) for cls in classes}) == len(classes)
    for tag, cls in emode.types._TYPE_REGISTRY.items():
        assert cls.__module__.startswith("emode."), tag


def test_a_wire_type_serialized_under_one_name_reconstructs_under_the_other():
    import emode
    import emodeconnection
    from emode.types import wire_dump
    from emodeconnection.types import reconstruct

    original = emode.Bounds(window_width=2.0, window_height=1.0, offset=(0.5, -0.5))
    payload = wire_dump(original)
    rebuilt = reconstruct(payload)
    assert type(rebuilt) is emode.Bounds
    assert type(rebuilt) is emodeconnection.Bounds
    assert rebuilt == original


def test_a_pickle_written_under_the_old_name_loads():
    import emode
    import emode.types

    data = pickle.dumps(emode.types.EModeError("boom"), protocol=0)
    assert b"emode.types" in data
    old = data.replace(b"emode.types", b"emodeconnection.types")
    loaded = pickle.loads(old)  # noqa: S301
    assert type(loaded) is emode.types.EModeError
    assert str(loaded) == "boom"


def test_a_pickle_written_under_the_new_name_round_trips():
    import emode

    bounds = emode.Bounds(window_width=2.0, window_height=1.0)
    loaded = pickle.loads(pickle.dumps(bounds))  # noqa: S301
    assert loaded == bounds


def test_emode_install_and_emode_mcp_come_from_the_emode_distribution():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "emode"
    assert project["scripts"] == {
        "emode-install": "emode.installer:main",
        "emode-mcp": "emode.mcp:main",
    }


def test_the_client_has_no_emode_console_script():
    # The EMode installer's binary has that name.
    for path in (
        ROOT / "pyproject.toml",
        ROOT / "packaging" / "emodeconnection" / "pyproject.toml",
    ):
        scripts = tomllib.loads(path.read_text())["project"].get("scripts", {})
        assert "emode" not in scripts, path


def test_the_shim_distribution_depends_on_emode_and_has_no_scripts():
    project = tomllib.loads(
        (ROOT / "packaging" / "emodeconnection" / "pyproject.toml").read_text()
    )["project"]
    assert project["name"] == "emodeconnection"
    assert any(
        dep.replace(" ", "").startswith("emode>=") for dep in project["dependencies"]
    )
    assert "scripts" not in project
    assert project["optional-dependencies"]["mcp"][0].startswith("emode[mcp]")
    emode = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == emode["version"]


def test_the_placeholder_distribution_is_gone():
    assert not (ROOT / "packaging" / "emode").exists()


def test_the_mcp_server_is_importable_from_either_name():
    import emode.mcp
    import emodeconnection.mcp
    import emodeconnection.mcp.server

    assert emodeconnection.mcp.main is emode.mcp.main
    assert emodeconnection.mcp.server.create_server is emode.mcp.server.create_server


def test_the_mcp_server_lists_its_tools_from_either_name():
    for name in ("emode", "emodeconnection"):
        result = _python(
            f"import sys, {name}.mcp as m\n"
            "sys.argv = ['emode-mcp', '--schemas']\n"
            'm.main(["--schemas"])\n'
        )
        assert result.returncode == 0, result.stderr
        assert "emode_open" in result.stdout


def test_the_extension_entry_point_runs_the_emode_package():
    text = (ROOT / "packaging" / "mcpb" / "server" / "main.py").read_text()
    assert "from emode.mcp import main" in text
