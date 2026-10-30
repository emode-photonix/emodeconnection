"""The remaining plan 8.1 tools, the confirmation gate, large results and --schemas."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mcp")

from test_mcp import FakeEMode, _reset_fake, call, session  # noqa: F401

from emodeconnection import ConvergenceResult, Objective, OptimizationResult, Parameter, PlotImage
from emodeconnection.mcp import main
from emodeconnection.mcp.server import create_server

OPEN = ("emode_open", {})


def run(tool: str, args: dict[str, Any], **server_kw: Any) -> Any:
    """Open a session, call one tool, return its result."""
    return session(OPEN, (tool, args), **server_kw)[1]


def last_call(name: str) -> dict[str, Any]:
    return [c for c in FakeEMode.instances[0].calls if c[0] == name][-1][1]


# settings, shape, material


def test_settings_applies_and_reports_what_it_set() -> None:
    result = run("emode_settings", {"settings": {"wavelength": 1310, "num_modes": 3}})
    assert not result.is_error
    assert last_call("settings") == {"wavelength": 1310, "num_modes": 3}
    assert result.structured_content["data"] == {"wavelength": 1310, "num_modes": 3}


def test_settings_rejects_an_unknown_name() -> None:
    assert run("emode_settings", {"settings": {"wavelenth": 1}}).is_error


def test_shape_draws_one_shape() -> None:
    result = run("emode_shape", {"shape": {"name": "core", "material": "Si", "width": 500, "height": 220}})
    assert not result.is_error
    assert last_call("shape") == {"name": "core", "material": "Si", "width": 500, "height": 220}


def test_material_add_passes_the_equation() -> None:
    result = run(
        "emode_material",
        {"action": "add", "name": "glass", "refractive_index_equation": "1.45+0.001*x", "loss": 2},
    )
    assert not result.is_error
    assert last_call("add_material") == {
        "name": "glass",
        "refractive_index_equation": "1.45+0.001*x",
        "loss": 2,
    }


def test_material_add_needs_an_equation() -> None:
    assert run("emode_material", {"action": "add", "name": "glass"}).is_error


def test_material_index_looks_a_material_up() -> None:
    FakeEMode.returns["refractive_index"] = np.array([3.476])
    result = run("emode_material", {"action": "index", "name": "Si", "wavelength": 1550})
    assert last_call("refractive_index") == {"material": "Si", "wavelength": 1550}
    assert result.structured_content["data"] == [3.476]


# sections and EME


@pytest.mark.parametrize(
    ("kind", "method", "fields"),
    [
        ("straight", "straight_section", {"name": "a", "length": 1000, "profile": "p"}),
        ("taper", "taper_section", {"name": "t", "profile": "p", "profile_end": "q", "length": 5000}),
        ("bend", "bend_section", {"name": "b", "radius": 20000, "sweep_angle": 1.57}),
        ("gds", "gds_section", {"name": "g", "gds_path": "x.gds", "layer_map": {"core": 1}}),
        ("pwd", "pwd_section", {"name": "w", "length": 500, "material": "SiO2", "max_order": [3, 3]}),
    ],
)
def test_section_kinds_call_their_client_method(kind: str, method: str, fields: dict[str, Any]) -> None:
    result = run("emode_section", {"kind": kind, **fields})
    assert not result.is_error
    assert last_call(method) == fields


def test_section_refuses_a_field_its_kind_does_not_take() -> None:
    result = run("emode_section", {"kind": "straight", "name": "a", "radius": 5})
    assert result.is_error
    assert "radius" in result.content[0].text
    assert "straight_section" not in [c[0] for c in FakeEMode.instances[0].calls]


def test_eme_runs_and_summarizes_the_s_matrix() -> None:
    from emodeconnection import SMatrix

    s = SMatrix(
        port_sizes=(1, 1),
        data=np.array([[0.1 + 0j, 0.9j], [0.9j, 0.1 + 0j]]),
        port_names=("left", "right"),
    )
    FakeEMode.returns["get"] = lambda key: s if key == "S_matrix" else None
    result = run("emode_eme", {})
    assert not result.is_error
    assert last_call("EME") == {}
    matrix = result.structured_content["data"]
    assert matrix["port_names"] == ["left", "right"]
    assert matrix["port_sizes"] == [1, 1]
    assert matrix["data"]["real"][0][0] == pytest.approx(0.1)
    assert matrix["data"]["imag"][0][1] == pytest.approx(0.9)


# sweep, with the confirmation gate


def test_a_small_sweep_runs_and_returns_the_result() -> None:
    FakeEMode.returns["sweep"] = {"values": [1500, 1550], "effective_index": [2.5, 2.4]}
    result = run("emode_sweep", {"key": "wavelength", "values": [1500, 1550]})
    assert result.structured_content["status"] == "done"
    assert last_call("sweep") == {"key": "wavelength", "values": [1500, 1550]}
    assert result.structured_content["data"]["effective_index"] == [2.5, 2.4]


def test_a_sweep_over_50_points_asks_first_and_runs_nothing() -> None:
    result = run("emode_sweep", {"key": "wavelength", "start": 1500, "stop": 1600, "num": 50})
    assert not result.is_error
    assert result.structured_content["status"] == "confirmation_required"
    assert "confirm" in result.structured_content["message"]
    assert "sweep" not in [c[0] for c in FakeEMode.instances[0].calls]


def test_a_confirmed_sweep_runs_and_builds_its_values_from_start_stop_num() -> None:
    result = run("emode_sweep", {"key": "wavelength", "start": 1500, "stop": 1600, "num": 50, "confirm": True})
    assert result.structured_content["status"] == "done"
    values = last_call("sweep")["values"]
    assert len(values) == 50
    assert values[0] == 1500
    assert values[-1] == 1600


def test_exactly_20_points_do_not_need_confirmation() -> None:
    result = run("emode_sweep", {"key": "wavelength", "start": 1500, "stop": 1600, "num": 20})
    assert result.structured_content["status"] == "done"


def test_an_interpolated_sweep_is_gated_by_its_nodes_not_its_wavelengths() -> None:
    args = {
        "key": "wavelength",
        "start": 1500,
        "stop": 1600,
        "num": 100,
        "solve_type": "EME",
        "result": "S_matrix",
        "method": "interpolated",
        "num_nodes": 12,
    }
    assert run("emode_sweep", args).structured_content["status"] == "done"
    assert last_call("sweep")["method"] == "interpolated"
    assert last_call("sweep")["num_nodes"] == 12


def test_a_sweep_needs_values_or_a_range_but_not_both() -> None:
    assert run("emode_sweep", {"key": "wavelength"}).is_error
    both = {"key": "wavelength", "values": [1], "start": 1, "stop": 2, "num": 3}
    assert run("emode_sweep", both).is_error


# converge, evaluate, optimize


def _convergence() -> ConvergenceResult:
    return ConvergenceResult(
        setting="resolution",
        target="effective_index",
        tolerance=1e-4,
        status="converged",
        chosen={"x_resolution": 20.0},
    )


def test_converge_returns_the_report_the_solver_made() -> None:
    FakeEMode.returns["converge"] = _convergence()
    result = run("emode_converge", {"setting": "resolution", "max_solves": 8})
    assert last_call("converge") == {"setting": "resolution", "max_solves": 8}
    data = result.structured_content["data"]
    assert data["status"] == "converged"
    assert data["chosen"] == {"x_resolution": 20.0}
    assert "converged" in data["table"]


def test_evaluate_passes_parameters_and_results() -> None:
    FakeEMode.returns["evaluate"] = {"index": 0, "results": {"effective_index": 2.4}}
    result = run(
        "emode_evaluate",
        {"parameters": {"shape, core, width": 450}, "results": ["effective_index"]},
    )
    assert last_call("evaluate") == {
        "parameters": {"shape, core, width": 450},
        "results": ["effective_index"],
    }
    assert result.structured_content["data"]["results"]["effective_index"] == 2.4


OPT = {
    "parameters": [{"key": "shape, core, width", "bounds": [400, 600]}],
    "objective": [{"kind": "effective_index", "goal": "target", "target": 2.4}],
    "max_evaluations": 30,
}


def _optimization() -> OptimizationResult:
    return OptimizationResult(
        x=[480.0],
        parameters={"shape, core, width": 480.0},
        objective=1e-6,
        status="converged",
        evaluations=21,
        goal="minimize",
    )


def test_optimize_asks_before_it_runs() -> None:
    result = run("emode_optimize", OPT)
    assert result.structured_content["status"] == "confirmation_required"
    assert "optimize" not in [c[0] for c in FakeEMode.instances[0].calls]


def test_a_confirmed_optimize_passes_the_public_types() -> None:
    FakeEMode.returns["optimize"] = _optimization()
    result = run("emode_optimize", {**OPT, "confirm": True})
    assert result.structured_content["status"] == "done"
    sent = last_call("optimize")
    assert sent["parameters"] == [Parameter("shape, core, width", bounds=(400, 600))]
    assert sent["objective"] == [Objective("effective_index", goal="target", target=2.4)]
    assert sent["max_evaluations"] == 30
    assert "confirm" not in sent
    data = result.structured_content["data"]
    assert data["parameters"] == {"shape, core, width": 480.0}
    assert data["status"] == "converged"


# get, plot


def test_get_reads_a_value_by_key() -> None:
    FakeEMode.returns["get"] = lambda key: np.array([1.0, 2.0])
    result = run("emode_get", {"key": "effective_index"})
    assert last_call("get") == {"key": "effective_index"}
    assert result.structured_content["data"] == [1.0, 2.0]


def test_get_without_a_key_lists_what_can_be_read() -> None:
    FakeEMode.returns["inspect"] = ["effective_index", "loss_dB_per_m"]
    result = run("emode_get", {})
    assert result.structured_content["data"] == ["effective_index", "loss_dB_per_m"]


def test_a_large_array_is_saved_to_a_file_and_its_path_returned(tmp_path: Path) -> None:
    big = np.arange(5000, dtype=float).reshape(50, 100)
    FakeEMode.returns["get"] = lambda key: big
    result = run("emode_get", {"key": "Ex"}, output_dir=tmp_path)
    ref = result.structured_content["data"]
    assert ref["shape"] == [50, 100]
    assert Path(ref["file"]).parent == tmp_path
    np.testing.assert_array_equal(np.load(ref["file"]), big)
    assert ref["file"] in result.structured_content["files"]


def test_a_small_array_stays_inline(tmp_path: Path) -> None:
    FakeEMode.returns["get"] = lambda key: np.ones((3, 3))
    result = run("emode_get", {"key": "x"}, output_dir=tmp_path)
    assert result.structured_content["data"] == np.ones((3, 3)).tolist()
    assert list(tmp_path.iterdir()) == []


def test_plot_returns_the_png_the_solver_saved(tmp_path: Path) -> None:
    png = b"\x89PNG\r\n\x1a\nfake"

    def plot(**kwargs: Any) -> PlotImage:
        Path(kwargs["file_name"]).write_bytes(png)
        return PlotImage(file=kwargs["file_name"])

    FakeEMode.returns["plot"] = plot
    result = run("emode_plot", {"component": "Ex", "mode": 1}, output_dir=tmp_path)
    assert not result.is_error
    sent = last_call("plot")
    assert sent["component"] == "Ex"
    assert sent["mode"] == 1
    assert Path(sent["file_name"]).parent == tmp_path
    image = next(c for c in result.content if c.type == "image")
    assert image.mime_type == "image/png"
    assert base64.b64decode(image.data) == png


def test_plot_uses_the_png_bytes_when_the_solver_sends_them(tmp_path: Path) -> None:
    png = b"\x89PNG\r\n\x1a\nbytes"
    FakeEMode.returns["plot"] = PlotImage(png=np.frombuffer(png, dtype=np.uint8))
    result = run("emode_plot", {}, output_dir=tmp_path)
    image = next(c for c in result.content if c.type == "image")
    assert base64.b64decode(image.data) == png


# warnings


def test_client_warnings_during_a_call_reach_the_agent() -> None:
    def noisy(**kwargs: Any) -> dict:
        logging.getLogger("emodeconnection.emodeconnection").warning("the client says careful")
        return {}

    FakeEMode.returns["sweep"] = noisy
    result = run("emode_sweep", {"key": "wavelength", "values": [1550]})
    assert "the client says careful" in result.structured_content["warnings"]


# schemas and the public method list


def test_schemas_prints_every_tool_definition_without_starting_a_session(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--schemas"])
    tools = json.loads(capsys.readouterr().out)
    assert sorted(t["name"] for t in tools) == sorted(t.name for t in _tools())
    for tool in tools:
        assert tool["description"]
        assert tool["input_schema"]["type"] == "object"
    assert FakeEMode.instances == []


def _tools() -> list[Any]:
    import asyncio

    from mcp.client import Client

    async def go() -> Any:
        async with Client(create_server(client_factory=FakeEMode)) as client:
            return (await client.list_tools()).tools

    return asyncio.run(go())


CLIENT_METHODS = {
    line.strip()
    for line in (Path(__file__).parent / "fixtures" / "client_methods.txt").read_text().splitlines()
    if line.strip()
}

# Public client methods no tool wraps on purpose, and why.
OMITTED = {
    # session plumbing the tools own
    "init": "emode_open starts the session",
    "open": "emode_open starts the session",
    "version": "not a user action",
    "snapshot": "optimize and evaluate manage snapshots",
    # reached through a wrapped tool
    "report": "emode_solve returns each mode's index and loss",
    "solve": "emode_solve calls FDM or FEM directly",
    "get_shape": "emode_get with a shape_ key",
    "get_grid": "emode_get",
    "get_profile": "emode_get",
    "get_fields": "emode_get with large results saved to a file",
    # later milestones (plans 8.1 and later) and interactive or file-only functions
    "EME_settings": "a later tool",
    "connect_ports": "a later tool",
    "copy_section": "a later tool",
    "check_layout": "a later tool",
    "to_gds": "a later tool",
    "free_space_field": "a later tool",
    "free_space_field_diagnostics": "a later tool",
    "monitor": "a later tool",
    "monitor_coupling": "a later tool",
    "monitor_field": "a later tool",
    "pulse_propagate": "a later tool",
    "chi2_spectrum": "a later tool",
    "create_profile_set": "a later tool",
    "delete_profile": "a later tool",
    "label": "a later tool",
    "label_profile": "a later tool",
    "save": "a later tool",
    "clear_field_data": "a later tool",
    "import_database": "a later tool",
    "d_matrix": "a later tool",
    "plot_S_matrix": "interactive, emode_plot covers field plots",
    "material_explorer": "interactive",
    "heat": "multiphysics, a later tool",
    "electrostatic": "multiphysics, a later tool",
    "mesh": "a later tool",
    "mode_order": "a later tool",
    "confinement": "a later tool",
    "effective_area": "a later tool",
    "group_index": "a later tool",
    "orthogonality": "a later tool",
    "overlap": "a later tool",
    "permittivity": "a later tool",
    "permeability": "a later tool",
    "scattering": "a later tool",
}


def test_every_public_client_method_is_wrapped_or_deliberately_omitted() -> None:
    from emodeconnection.mcp.server import WRAPPED_METHODS

    assert WRAPPED_METHODS <= CLIENT_METHODS, "a tool calls a method the client does not have"
    unaccounted = CLIENT_METHODS - WRAPPED_METHODS - set(OMITTED)
    assert not unaccounted, f"wrap or omit: {sorted(unaccounted)}"
    assert not (set(OMITTED) - CLIENT_METHODS), "omitted a method that does not exist"
