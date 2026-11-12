"""Sweep results of every kind: their wire type, the dicts scripts read, and their plots.

``sweep()`` and ``chi2_spectrum()`` answer with a ``SweepResult``: the dict a
script has always read (``result['values']``, ``result['effective_index']``,
``.keys()``, iteration, ``in``), which also plots itself (``result.plot()``,
``em.plot(result)``). An ``OptimizationResult``'s history and a
``PulseResponse`` plot the same way. Every plot is a ``FigureSpec`` composed
here, drawn by the browser's Plotly renderer and the static matplotlib
renderer alike.
"""
from __future__ import annotations

import base64
import json
import pickle

import numpy as np
import pytest

from emode import compat
from emode.emodeclient import obj_hook
from emode.optimize import Evaluation, OptimizationResult
from emode.pulse import PulseResponse
from emode.smatrix import SMatrix
from emode.sweep import SweepResult
from emode.types import UnreadWireType, reconstruct, serialize, wire_dump
from emode.ui.figure import PlotOptions, compose
from emode.ui.renderers.plotly import to_plotly
from emode.ui.sweep_figure import SweepPanel, sweep_menus, sweep_panel
from emode.ui.views import _select, kind_of, title_of

# ------------------------------------------------------------------ the sweeps


def wavelength_sweep(tracked: bool = True) -> SweepResult:
    """Three modes over four wavelengths, modes 1 and 2 crossing between the second and third.

    Mode A (2.0, 1.97, 1.94, 1.91) and mode B (1.9, 1.92, 1.95, 1.98) swap
    places in the solver's descending order at 1550 nm, which ``mode_order``
    records: the previous point's mode 1 is this point's mode 2.
    """
    n = np.array(
        [
            [2.40, 2.00, 1.90],
            [2.39, 1.97, 1.92],
            [2.38, 1.95, 1.94],
            [2.37, 1.98, 1.91],
        ]
    ) + 1e-6j
    data = {"key": "wavelength", "values": np.array([1500.0, 1525.0, 1550.0, 1575.0]), "effective_index": n}
    if tracked:
        data["mode_order"] = [
            [0.0, [0, 1, 2]],
            [0.01, [0, 1, 2]],
            [0.02, [0, 2, 1]],
            [0.01, [0, 1, 2]],
        ]
    return SweepResult(data=data)


def _smatrix(t: complex, r: complex) -> SMatrix:
    """Two ports of two modes each: mode 0 transmits `t`, mode 1 takes what is left."""
    a = np.sqrt(max(0.0, 1 - abs(t) ** 2 - abs(r) ** 2))
    S = np.zeros((4, 4), complex)
    S[2, 0] = S[0, 2] = t
    S[3, 0] = S[0, 3] = a
    S[0, 0] = r
    S[3, 1] = S[1, 3] = 0.5
    return SMatrix(port_sizes=(2, 2), data=S, port_names=("left", "right"), wavelengths=(1550.0, 1550.0))


def length_sweep() -> SweepResult:
    """The taper example's shape: one S-matrix per taper length."""
    lengths = np.array([10000.0, 20000.0, 30000.0])
    smatrices = [_smatrix(0.9 * np.exp(1j * k), 0.1j) for k in range(3)]
    return SweepResult(data={"key": "length", "values": lengths, "S_matrix": smatrices})


def interpolated_sweep() -> SweepResult:
    wavelengths = np.linspace(1500.0, 1600.0, 11)
    smatrices = [_smatrix(np.cos((w - 1500) / 40) * np.exp(1j * w / 10), 0.0) for w in wavelengths]
    nodes = np.array([1500.0, 1509.5, 1537.0, 1550.0, 1563.0, 1590.5, 1600.0])
    return SweepResult(
        data={
            "key": "wavelength",
            "values": wavelengths,
            "S_matrix": smatrices,
            "num_nodes": 6,
            "interpolation_residual": 2.5e-4,
            "interpolation_residuals": {
                "wavelength": 1563.0,
                "slices": 2.5e-4,
                "junctions": 1e-5,
                "circuit": 3e-3,
            },
            "node_wavelengths": nodes,
        }
    )


def grid_sweep() -> SweepResult:
    widths = [400.0, 500.0, 600.0]
    wavelengths = [1500.0, 1550.0, 1600.0, 1650.0]
    n = [
        [[2.0 + w / 1000 - l / 10000, 1.6 + w / 2000 - l / 20000] for l in wavelengths] for w in widths
    ]
    return SweepResult(
        data={
            "key": ["shape, core, width", "wavelength"],
            "values": [np.array(widths), np.array(wavelengths)],
            "effective_index": n,
        }
    )


def beam_sweep() -> SweepResult:
    angles = np.array([-2.0, 0.0, 2.0, 4.0])
    return SweepResult(
        data={
            "key": "angle",
            "values": angles,
            "monitor_coupling": np.array([0.5, 0.8, 0.6, 0.3]),
            "power_through_plane": np.array([1.0, 1.0, 1.0, 1.0]),
            "overlap": np.array([0.7, 0.9j, 0.77, 0.55 + 0.1j]),
        }
    )


def chi2_spectrum() -> SweepResult:
    w = np.linspace(1540.0, 1560.0, 9)
    return SweepResult(
        data={
            "key": "wavelength",
            "values": w,
            "conversion": np.sinc((w - 1550.0) / 4) ** 2 * 1e-3,
            "solved_wavelengths": np.array([1540.0, 1545.0, 1550.0, 1560.0]),
            "unconverged": [(1555.0, 1560.0, 0.02)],
        }
    )


def optimization() -> OptimizationResult:
    objectives = [0.5, 0.3, 0.4, 0.1, 0.2]
    history = [
        Evaluation(index=i, parameters={"width": 500.0 + 10 * i, "gaps": [100.0 + i, 200.0 - i]}, objective=o)
        for i, o in enumerate(objectives)
    ]
    return OptimizationResult(
        x=[530.0, 103.0, 197.0],
        parameters=history[3].parameters,
        objective=0.1,
        history=history,
        best_index=3,
        evaluations=5,
        goal="minimize",
    )


def pulse() -> PulseResponse:
    c = 299792458.0
    frequency = np.linspace(c / 1700e-9, c / 1400e-9, 64)
    f0 = c / 1550e-9
    shape = np.exp(-(((frequency - f0) / 5e12) ** 2)) * 1e-12
    spectrum = np.stack([shape * s for s in (1.0, 0.9, 0.8)])[:, None, :].astype(complex)
    return PulseResponse(
        z=np.array([0.0, 5e5, 1e6]),
        frequency=frequency,
        spectrum=spectrum,
        modes=[0],
        reference_index=32,
        energy=np.array([[1.0], [0.81], [0.64]]),
        junction_loss=[],
        edge_energy=np.zeros(3),
        domain_walls=[],
        steps=10,
        rejected_steps=0,
        rank_per_slice=[],
        samples_per_slice=[],
    )


SWEEPS = {
    "wavelength": wavelength_sweep,
    "length": length_sweep,
    "interpolated": interpolated_sweep,
    "grid": grid_sweep,
    "beam": beam_sweep,
    "chi2": chi2_spectrum,
}

#: Every kind of sweep-shaped result, each with a panel of its own.
KINDS = {**SWEEPS, "optimize": optimization, "pulse": pulse}


def _round_trip(value):
    """`value` sent as an EMode reply is (arrays as ``__ndarray__``) and read back as the client reads one."""
    payload = json.dumps(serialize(wire_dump(value), ndarray=True))
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _lines(spec, kind: str = "trace") -> list:
    return [line for line in spec.lines if line.kind == kind]


# ------------------------------------------------------------------ the wire


@pytest.mark.parametrize("make", list(SWEEPS.values()), ids=list(SWEEPS))
def test_each_sweep_round_trips_the_wire(make):
    value = make()
    back = _round_trip(value)
    assert type(back) is SweepResult
    assert list(back.keys()) == list(value.keys())
    if value.layout == "grid":
        for got, sent in zip(back["values"], value["values"]):
            np.testing.assert_array_equal(got, sent)
    else:
        np.testing.assert_array_equal(back["values"], value["values"])
    assert back == value
    if "S_matrix" in value:
        assert all(isinstance(s, SMatrix) for s in back["S_matrix"])
        np.testing.assert_array_equal(back["S_matrix"][1].data, value["S_matrix"][1].data)


def test_the_other_kinds_round_trip_the_wire():
    back = _round_trip(optimization())
    assert type(back) is OptimizationResult and len(back.history) == 5
    back = _round_trip(pulse())
    assert type(back) is PulseResponse
    np.testing.assert_array_equal(back.spectrum, pulse().spectrum)


def test_on_the_wire_the_results_sit_at_the_top_level_so_an_older_client_reads_its_dict():
    dumped = serialize(wire_dump(beam_sweep()), ndarray=True)
    assert set(dumped) == {"__data_type__", "key", "values", "monitor_coupling", "power_through_plane", "overlap"}
    assert dumped["__data_type__"] == "SweepResult"
    # a client without the class keeps the reply as the dict it always read
    older = UnreadWireType(json.loads(json.dumps(dumped), object_hook=obj_hook))
    assert older["key"] == "angle"
    np.testing.assert_array_equal(older["monitor_coupling"], beam_sweep()["monitor_coupling"])


def test_the_type_is_versioned():
    assert compat.TYPE_SINCE["SweepResult"] == "1.0.9"


def test_a_saved_sweep_read_with_get_is_a_sweep_result(tmp_path, monkeypatch):
    import emode

    data = dict(beam_sweep())
    (tmp_path / "saved.eph").write_bytes(pickle.dumps({"sweep_data": data}))
    monkeypatch.chdir(tmp_path)
    got = emode.get("sweep_data", simulation_name="saved")
    assert isinstance(got, SweepResult)
    assert got == data


# ------------------------------------------------------------------ the dict scripts read


def test_a_sweep_reads_as_the_dict_scripts_use():
    """The access patterns of the website examples and the user scripts they grew into."""
    result = length_sweep()
    assert result["key"] == "length"
    assert "S_matrix" in result and "effective_index" not in result
    assert list(result) == ["key", "values", "S_matrix"]
    assert list(result.keys()) == ["key", "values", "S_matrix"]
    assert len(result) == 3
    assert dict(result)["key"] == "length"
    assert result.get("missing") is None and result.get("key") == "length"
    assert [k for k, _ in result.items()] == ["key", "values", "S_matrix"]
    transmission = np.array([s.transmission() for s in result["S_matrix"]])
    assert transmission.shape == (3, 2, 2)
    np.testing.assert_allclose(np.min(result["values"]), 10000.0)
    assert result == dict(result)
    # a script may add its own entries, as it could to the dict
    result["transmission"] = transmission
    assert "transmission" in result
    del result["transmission"]
    assert "transmission" not in result
    merged = result | {"note": 1}
    assert merged["note"] == 1 and "note" not in result
    assert pickle.loads(pickle.dumps(result)) == result


def test_index_results_index_as_arrays():
    result = wavelength_sweep()
    assert result["effective_index"].shape == (4, 3)
    assert np.real(result["effective_index"][0, 0]) == pytest.approx(2.4)
    for value in result.values():
        assert value is not None


# ------------------------------------------------------------------ the panels


def test_a_per_mode_result_is_one_trace_per_mode_against_the_key():
    spec = compose(sweep_panel(wavelength_sweep(tracked=False)))
    traces = _lines(spec)
    assert [t.label for t in traces] == ["mode 0", "mode 1", "mode 2"]
    np.testing.assert_array_equal(traces[1].x, [1500.0, 1525.0, 1550.0, 1575.0])
    np.testing.assert_allclose(traces[1].y, [2.00, 1.97, 1.95, 1.98])
    assert spec.axes.xlabel == "Wavelength (nm)"
    assert spec.axes.ylabel == "Effective index"


def test_modes_follow_their_tracking():
    traces = _lines(compose(sweep_panel(wavelength_sweep())))
    np.testing.assert_allclose(traces[1].y, [2.00, 1.97, 1.94, 1.91])
    np.testing.assert_allclose(traces[2].y, [1.90, 1.92, 1.95, 1.98])


def test_an_index_result_offers_its_real_and_imaginary_parts():
    panel = sweep_panel(wavelength_sweep())
    menus = {m["name"]: m for m in sweep_menus(panel)}
    assert menus["component"]["options"] == ["real", "imag"]
    assert menus["component"]["selected"] == "real"
    imag = _lines(compose(_select(panel, {"component": "imag"})))
    np.testing.assert_allclose(imag[0].y, 1e-6)
    assert "mode_order" not in menus["result"]["options"]


def test_values_out_of_order_are_drawn_in_order():
    data = dict(beam_sweep())
    order = [2, 0, 3, 1]
    data["values"] = data["values"][order]
    data["monitor_coupling"] = data["monitor_coupling"][order]
    (trace,) = _lines(compose(sweep_panel(SweepResult(data=data))))
    np.testing.assert_array_equal(trace.x, [-2.0, 0.0, 2.0, 4.0])
    np.testing.assert_allclose(trace.y, [0.5, 0.8, 0.6, 0.3])


def test_the_s_matrix_is_drawn_from_the_first_port_to_the_last_one_trace_per_output_mode():
    """What the taper example drew by hand: ``abs(transmission[:, kk, 0])**2`` for each kk."""
    result = length_sweep()
    spec = compose(sweep_panel(result))
    traces = _lines(spec)
    transmission = np.array([s.transmission() for s in result["S_matrix"]])
    assert [t.label for t in traces] == ["left 0 to right 0", "left 0 to right 1"]
    for kk, trace in enumerate(traces):
        np.testing.assert_allclose(trace.y, np.abs(transmission[:, kk, 0]) ** 2)
    assert spec.axes.xlabel == "length (nm)"
    assert spec.axes.ylabel == "|S|²"


@pytest.mark.parametrize(
    "component, part",
    [
        ("abs^2", lambda s: np.abs(s) ** 2),
        ("dB", lambda s: 10 * np.log10(np.abs(s) ** 2)),
        ("phase", np.angle),
        ("real", np.real),
        ("imag", np.imag),
    ],
)
def test_a_complex_result_in_each_part(component, part):
    result = length_sweep()
    panel = _select(sweep_panel(result), {"component": component})
    trace = _lines(compose(panel))[0]
    entries = np.array([s.data[2, 0] for s in result["S_matrix"]])
    np.testing.assert_allclose(trace.y, part(entries), rtol=1e-6)


def test_the_s_matrix_ports_and_input_mode_are_chosen_in_the_menu():
    panel = _select(sweep_panel(length_sweep()), {"input": "right", "output": "left", "input_mode": "1"})
    traces = _lines(compose(panel))
    assert [t.label for t in traces] == ["right 1 to left 0", "right 1 to left 1"]
    np.testing.assert_allclose(traces[1].y, 0.25)


def test_the_interpolated_sweep_marks_its_nodes_and_the_held_out_residual():
    spec = compose(sweep_panel(interpolated_sweep()))
    (nodes,) = _lines(spec, "node")
    np.testing.assert_array_equal(nodes.x, interpolated_sweep()["node_wavelengths"])
    (held,) = _lines(spec, "held_out")
    assert held.x[0] == 1563.0
    (note,) = [t for t in spec.texts if t.kind == "residual"]
    assert "2.5e-04" in note.text and "1563" in note.text and "6 nodes" in note.text


def test_a_two_key_sweep_is_a_map_with_line_cuts():
    result = grid_sweep()
    spec = compose(sweep_panel(result))
    (raster,) = spec.rasters
    np.testing.assert_array_equal(raster.x, [400.0, 500.0, 600.0])
    np.testing.assert_array_equal(raster.y, [1500.0, 1550.0, 1600.0, 1650.0])
    n = np.real(np.asarray(result["effective_index"]))
    np.testing.assert_allclose(raster.z, n[:, :, 0].T)
    assert spec.axes.xlabel == "core width (nm)" and spec.axes.ylabel == "Wavelength (nm)"
    assert spec.colorbar is not None and spec.colorbar.label == "Effective index, mode 0"

    menus = {m["name"]: m for m in sweep_menus(sweep_panel(result))}
    assert menus["panel"]["options"] == ["map", "against core width", "against wavelength"]
    cuts = compose(_select(sweep_panel(result), {"panel": "against core width", "mode": "1"}))
    traces = _lines(cuts)
    assert [t.label for t in traces] == [f"wavelength = {w:g} nm" for w in (1500, 1550, 1600, 1650)]
    np.testing.assert_allclose(traces[0].y, n[:, 0, 1])
    assert not cuts.rasters


def test_the_beam_sweep_plots_the_coupling_against_the_angle():
    spec = compose(sweep_panel(beam_sweep()))
    (trace,) = _lines(spec)
    np.testing.assert_allclose(trace.y, [0.5, 0.8, 0.6, 0.3])
    assert spec.axes.xlabel == "angle (°)" and spec.axes.ylabel == "Coupling"
    menus = {m["name"]: m for m in sweep_menus(sweep_panel(beam_sweep()))}
    assert menus["result"]["options"] == ["monitor_coupling", "power_through_plane", "overlap"]


def test_several_results_are_traces_of_one_panel():
    spec = compose(sweep_panel(beam_sweep(), result=["monitor_coupling", "power_through_plane"]))
    assert [t.label for t in _lines(spec)] == ["Coupling", "Power through plane"]


def test_the_chi2_spectrum_marks_where_it_solved_and_what_did_not_converge():
    spec = compose(sweep_panel(chi2_spectrum()))
    (trace,) = _lines(spec)
    assert spec.axes.ylabel == "Conversion (signal out / pump in)"
    (solved,) = _lines(spec, "solved")
    np.testing.assert_array_equal(solved.x, [1540.0, 1545.0, 1550.0, 1560.0])
    (bad,) = _lines(spec, "unconverged")
    assert list(bad.x) == [1555.0, 1560.0]


def test_the_optimizer_history_draws_the_objective_and_the_parameters():
    result = optimization()
    spec = compose(sweep_panel(result))
    (trace,) = _lines(spec)
    np.testing.assert_array_equal(trace.x, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(trace.y, [0.5, 0.3, 0.4, 0.1, 0.2])
    (best,) = _lines(spec, "best")
    assert (best.x[0], best.y[0]) == (3, pytest.approx(0.1))
    (running,) = _lines(spec, "running_best")
    np.testing.assert_allclose(running.y, [0.5, 0.3, 0.3, 0.1, 0.1])
    assert spec.axes.xlabel == "Evaluation" and spec.axes.ylabel == "Objective"

    parameters = compose(_select(sweep_panel(result), {"panel": "parameters"}))
    assert [t.label for t in _lines(parameters)] == ["width", "gaps[0]", "gaps[1]"]
    np.testing.assert_allclose(_lines(parameters)[2].y, [200.0, 199.0, 198.0, 197.0, 196.0])


def test_the_pulse_draws_its_spectrum_its_spectrum_against_z_and_its_energy():
    response = pulse()
    spec = compose(sweep_panel(response))
    output = next(t for t in _lines(spec) if t.label == "output")
    order = np.argsort(response.wavelength)
    np.testing.assert_allclose(output.x, response.wavelength[order])
    np.testing.assert_allclose(output.y, response.psd()[order])
    assert spec.axes.xlabel == "Wavelength (nm)"

    along = compose(_select(sweep_panel(response), {"panel": "spectrum against z"}))
    (raster,) = along.rasters
    np.testing.assert_allclose(raster.y, [0.0, 0.5, 1.0])  # mm
    np.testing.assert_allclose(raster.z[1], response.psd(z=5e5)[order], rtol=1e-5)
    assert along.axes.ylabel == "z (mm)"

    energy = compose(_select(sweep_panel(response), {"panel": "energy"}))
    (trace,) = _lines(energy)
    np.testing.assert_allclose(trace.y, [1.0, 0.81, 0.64])


def test_a_plain_sweep_dict_plots_like_the_result():
    data = dict(beam_sweep())
    assert _lines(compose(sweep_panel(data)))[0].y.tolist() == _lines(compose(sweep_panel(beam_sweep())))[0].y.tolist()


def test_a_sweep_with_nothing_to_plot_says_so():
    from emode.types import EModeError

    with pytest.raises(EModeError, match="nothing in this sweep"):
        compose(sweep_panel(SweepResult(data={"key": "wavelength", "values": [1.0, 2.0], "x": [{}, {}]})))


def test_the_view_kind_and_title():
    panel = sweep_panel(length_sweep())
    assert kind_of(panel) == "sweep"
    assert title_of(panel, {}) == "sweep length: S_matrix"


# ------------------------------------------------------------------ both renderers


def _decode(typed) -> np.ndarray:
    if isinstance(typed, list):
        return np.asarray(typed, float)
    return np.frombuffer(base64.b64decode(typed["bdata"]), dtype=np.dtype(typed["dtype"]))


@pytest.mark.parametrize("make", list(KINDS.values()), ids=list(KINDS))
def test_both_renderers_draw_every_line_and_raster_of_each_kind(make):
    pytest.importorskip("matplotlib")
    from emode.ui.renderers import mpl

    panels = [sweep_panel(make())]
    menus = {m["name"]: m for m in sweep_menus(panels[0])}
    for option in menus.get("panel", {}).get("options", [])[1:]:
        panels.append(_select(panels[0], {"panel": option}))
    for panel in panels:
        spec = compose(panel)
        figure = to_plotly(spec)
        kinds = {line.kind for line in spec.lines}
        scatters = [t for t in figure["data"] if t["type"] == "scatter" and t.get("meta") in kinds]
        assert len(scatters) == len(spec.lines)
        for line, trace in zip(spec.lines, scatters):
            np.testing.assert_allclose(_decode(trace["y"]), line.y, rtol=1e-6)
        heatmaps = [t for t in figure["data"] if t["type"] == "heatmap"]
        assert len(heatmaps) == len(spec.rasters)

        fig = mpl.to_figure(spec)
        (ax,) = [a for a in fig.axes if a.get_label() != "<colorbar>"]
        drawn = [l for l in ax.lines if l.get_gid() in {line.kind for line in spec.lines}]
        assert len(drawn) == len(spec.lines)
        for line, artist in zip(spec.lines, drawn):
            np.testing.assert_allclose(np.asarray(artist.get_ydata(), float), line.y, rtol=1e-6)
        assert len(ax.images) == len(spec.rasters)
        labeled = [line.label for line in spec.lines if line.label]
        if labeled:
            legend = ax.get_legend()
            assert legend is not None
            assert [t.get_text() for t in legend.get_texts()] == labeled
            assert [t["name"] for t in figure["data"] if t.get("showlegend")] == labeled


def test_the_figure_options_apply_to_a_sweep():
    spec = compose(sweep_panel(beam_sweep()), PlotOptions(title="t", xlabel="x", ylabel="y", font_size=9))
    assert (spec.title, spec.axes.xlabel, spec.axes.ylabel, spec.font_size) == ("t", "x", "y", 9)


# ------------------------------------------------------------------ plot() and the tab


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.delenv("EMODE_HEADLESS", raising=False)
    monkeypatch.delenv("EMODE_PLOT_INLINE", raising=False)
    monkeypatch.setenv("EMODE_NONINTERACTIVE", "1")


def test_plot_with_a_file_name_draws_a_static_image(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from emode.plot import PlotImage

    monkeypatch.chdir(tmp_path)
    image = length_sweep().plot(file_name="taper", component="dB")
    assert isinstance(image, PlotImage)
    assert (tmp_path / "EM_figures" / "taper.png").read_bytes()[:4] == b"\x89PNG"


@pytest.mark.parametrize("make", [length_sweep, optimization, pulse], ids=["sweep", "optimize", "pulse"])
def test_em_plot_of_a_result_registers_a_sweep_view_without_asking_emode(quiet, make):
    from plot_standin import session

    em, client = session()
    handle = em.plot(make(), title="mine")
    assert handle.id == "w1"
    assert isinstance(handle.view, SweepPanel)
    assert handle.figure.title == "mine"
    assert client.calls("EM_plot") == []


def test_result_plot_uses_the_running_session(quiet):
    from emode.ui.session import register_session

    from plot_standin import session

    em, client = session()
    register_session(em)
    handle = beam_sweep().plot(result="overlap", component="phase")
    assert handle.id == "w1"
    assert handle.view.selected["result"] == "overlap"
    assert handle.view.selected["component"] == "phase"


def test_a_selection_the_result_does_not_have_is_refused():
    from emode.types import ArgumentError

    with pytest.raises(ArgumentError, match="effective_index"):
        sweep_panel(beam_sweep(), result="effective_index")


def test_the_tab_changes_a_sweeps_selection_without_asking_emode():
    from emode.ui.server import UIServer
    from emode.ui.views import ViewRegistry

    registry = ViewRegistry()
    view = registry.add(sweep_panel(length_sweep()), {}, PlotOptions(), (1200, 750))
    fetched = []
    server = UIServer(registry, lambda *a: fetched.append(a)).start()
    try:
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        body = json.dumps({"controls": {"component": "dB", "output": "left"}})
        headers = {"Host": f"127.0.0.1:{server.port}", "X-EMode-Token": server.token, "Content-Type": "application/json"}
        conn.request("POST", f"/api/views/{view.id}/data", body=body, headers=headers)
        reply = json.loads(conn.getresponse().read())
        conn.close()
    finally:
        server.stop()
    assert reply["kind"] == "sweep"
    menus = {m["name"]: m for m in reply["extra"]["menus"]}
    assert menus["component"]["selected"] == "dB"
    assert menus["output"]["selected"] == "left"
    assert {"result", "component", "input", "output", "input_mode"} <= set(menus)
    assert fetched == []
    names = [t["name"] for t in reply["figure"]["data"] if t.get("showlegend")]
    assert names == ["left 0 to left 0", "left 0 to left 1"]


def test_the_app_draws_a_sweeps_menus():
    from pathlib import Path

    import emode.ui

    app = (Path(emode.ui.__file__).parent / "static" / "app.js").read_text()
    assert 'kind === "sweep"' in app and "buildSweepControls" in app


# ------------------------------------------------------------------ what the website examples ask of it


def test_vmin_and_vmax_fix_a_line_panels_y_range():
    spec = compose(sweep_panel(length_sweep(), component="dB"), PlotOptions(vmin=-60.0, vmax=0.0))
    assert spec.axes.yrange == (-60.0, 0.0)


def test_units_um_shows_a_length_key_in_micrometers():
    spec = compose(sweep_panel(length_sweep()), PlotOptions(units="um"))
    np.testing.assert_allclose(_lines(spec)[0].x, [10.0, 20.0, 30.0])
    assert spec.axes.xlabel == "length (μm)"
    # a key that is not a length keeps its unit
    spec = compose(sweep_panel(beam_sweep()), PlotOptions(units="um"))
    assert spec.axes.xlabel == "angle (°)"


def test_a_power_result_is_drawn_linear_or_in_dB():
    panel = sweep_panel(beam_sweep())
    menus = {m["name"]: m for m in sweep_menus(panel)}
    assert menus["component"]["options"] == ["linear", "dB"]
    spec = compose(_select(panel, {"component": "dB"}))
    np.testing.assert_allclose(_lines(spec)[0].y, 10 * np.log10([0.5, 0.8, 0.6, 0.3]))
    assert spec.axes.ylabel == "Coupling (dB)"


def test_an_older_adaptive_scans_modes_by_wavelength_are_one_trace_per_mode():
    """An adaptive scan saved before 1.0.9 holds each per-mode result as (modes, wavelengths)."""
    w = np.linspace(1500.0, 1600.0, 5)
    n = np.vstack([2.4 - w / 1e4, 2.0 - w / 2e4])
    scan = SweepResult(
        data={"key": "wavelength", "values": w, "effective_index": n,
              "solved_wavelengths": np.array([1500.0, 1600.0]), "unconverged": []}
    )
    traces = _lines(compose(sweep_panel(scan)))
    assert [t.label for t in traces] == ["mode 0", "mode 1"]
    np.testing.assert_allclose(traces[1].y, n[1])


def test_one_output_mode_and_ports_by_index():
    panel = sweep_panel(length_sweep(), input=0, output=1, output_mode=0)
    (trace,) = _lines(compose(panel))
    assert trace.label == "left 0 to right 0"
    menus = {m["name"]: m for m in sweep_menus(panel)}
    assert menus["output_mode"]["options"] == ["all", "0", "1"]
    assert len(_lines(compose(_select(panel, {"output_mode": "all"})))) == 2
