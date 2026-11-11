"""Sweep-shaped results as figures: a sweep, an optimizer's history, a pulse.

A ``SweepPanel`` holds the result and what is selected in it (which result,
which part of a complex one, which panel of a grid, which ports of an
S-matrix). ``compose()`` draws it here as a ``FigureSpec`` of traces against
the swept key, or a map, so the browser tab and a static image agree, and the
tab's menus (``sweep_menus``) change the selection without asking EMode.

What each kind draws by default:

- one key: each result against it, a per-mode result as one trace per mode
  (followed by its ``mode_order`` when the sweep tracked the modes), the
  S-matrix from the first port's mode 0 to every mode of the last port,
- two keys: a map of the result, and line cuts against either key,
- an interpolated EME sweep: its nodes and its held-out node marked, and the
  held-out residual written,
- an adaptive scan or a chi(2) spectrum: the wavelengths it solved, and the
  stretches it could not resolve,
- an ``OptimizationResult``: the objective against the evaluation, the best
  so far and the best point, or every parameter against the evaluation,
- a ``PulseResponse``: the spectrum at a record against the launched one, the
  spectrum against z as a map, or each mode's energy against z.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import colormaps
from ..types import ArgumentError, EModeError
from .figure import (
    DEFAULT_SIZE_MM,
    LW,
    AxesSpec,
    ColorbarSpec,
    FigureSpec,
    LineSpec,
    PlotOptions,
    RasterSpec,
    TextSpec,
)

#: The selection a sweep's panel takes, from ``plot()`` and from the tab.
SWEEP_CONTROLS = ("result", "component", "panel", "input", "output", "input_mode", "mode", "record")

#: The parts of a complex amplitude, as the component menu offers them.
AMPLITUDE_COMPONENTS = ["abs^2", "dB", "phase", "real", "imag"]

#: The parts of a complex index (an effective index, a group index).
INDEX_COMPONENTS = ["real", "imag"]

#: Results that are indices: their imaginary part is a loss, never shown as a magnitude.
INDEX_RESULTS = {"effective_index", "n_eff", "group_index", "n_g", "n_eff_tilde"}

#: Real results that are powers or power fractions, drawn linear or in dB (10 log10 of the value).
POWER_RESULTS = {"monitor_coupling", "power_through_plane", "conversion"}

#: The parts of a power result.
POWER_COMPONENTS = ["linear", "dB"]

#: How each known result is labeled on its axis.
RESULT_LABELS = {
    "effective_index": "Effective index",
    "n_eff": "Effective index",
    "group_index": "Group index",
    "n_g": "Group index",
    "GVD": "GVD (ps²/km)",
    "D": "D (ps/(nm km))",
    "loss_dB_per_m": "Loss (dB/m)",
    "TE_fraction": "TE fraction",
    "window_confinement": "Window confinement",
    "conversion": "Conversion (signal out / pump in)",
    "monitor_coupling": "Coupling",
    "power_through_plane": "Power through plane",
    "overlap": "Overlap",
    "S_matrix": "S",
}

#: Swept properties measured in nm, shown with that unit.
NM_KEYS = {
    "wavelength", "wav", "width", "height", "length", "thickness", "radius", "bend_radius",
    "gap", "mask", "etch_depth", "window_width", "window_height", "x_resolution",
    "y_resolution", "mode_field_diameter", "center", "position", "offset", "period", "depth",
    "x", "y", "trapezoid_angle_offset", "pml_depth",
}

#: Swept properties in degrees.
DEGREE_KEYS = {"angle", "rotation", "sidewall_angle", "facet_angle", "trapezoid_angle"}

#: The markers of the sweep's reports, each its own kind: (marker, palette index, legend label).
MARKS = {
    "node": ("triangle-up", 5, "nodes"),
    "held_out": ("diamond", 1, "held-out node"),
    "solved": ("triangle-up", 5, "solves"),
    "best": ("diamond", 1, "best"),
}


@dataclass(frozen=True)
class SweepPanel:
    """A sweep-shaped result and what is selected in it: what the plot and the tab draw."""

    source: Any
    selected: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ the selection


def as_source(value: Any) -> Any:
    """`value` as a result a panel draws: a plain sweep dict becomes a ``SweepResult``."""
    from ..optimize import OptimizationResult
    from ..pulse import PulseResponse
    from ..sweep import SweepResult

    if isinstance(value, (SweepResult, OptimizationResult, PulseResponse)):
        return value
    if isinstance(value, Mapping) and "key" in value and "values" in value:
        return SweepResult(data=dict(value))
    raise TypeError(
        "a sweep plot takes a sweep()'s or chi2_spectrum()'s result, an OptimizationResult "
        f"or a PulseResponse, not {type(value).__name__}"
    )


def is_sweep_shaped(value: Any) -> bool:
    try:
        as_source(value)
    except TypeError:
        return False
    return True


def sweep_panel(source: Any, **selection: Any) -> SweepPanel:
    """`source` with `selection` applied, every choice it leaves out at its default.

    Raises an ``ArgumentError`` naming the choices for a selection `source` does not offer.
    """
    source = as_source(source)
    unknown = [k for k in selection if k not in SWEEP_CONTROLS]
    if unknown:
        raise ArgumentError(
            f"a sweep plot takes no {unknown[0]!r}. It takes {', '.join(SWEEP_CONTROLS)}.", "plot", unknown[0]
        )
    chosen = {k: v for k, v in selection.items() if v is not None}
    return SweepPanel(source, _resolve(source, chosen, strict=set(chosen)))


def select_sweep(panel: SweepPanel, controls: Mapping[str, Any]) -> SweepPanel:
    """`panel` with the tab's `controls` applied. A carried-over choice that no longer applies resets."""
    changed = {k: v for k, v in controls.items() if k in SWEEP_CONTROLS and v is not None}
    merged = {**panel.selected, **changed}
    return SweepPanel(panel.source, _resolve(panel.source, merged, strict=set(changed)))


def sweep_menus(panel: SweepPanel) -> List[Dict[str, Any]]:
    """The tab's menus for `panel`: each control's name, label, choices and selection."""
    return [
        {"name": name, "label": label, "options": options, "selected": _menu_value(panel.selected.get(name))}
        for name, label, options in _offers(panel.source, panel.selected)
        if options
    ]


def _menu_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _resolve(source: Any, raw: Dict[str, Any], strict: set) -> Dict[str, Any]:
    """The full selection: each control's value checked against what `source` offers.

    A value named in `strict` that is not offered raises. Any other one falls
    back to the control's default, as does a control left out.
    """
    selected: Dict[str, Any] = {}
    for _ in range(2):
        # the choices of one control depend on another (a component on the result)
        selected = {}
        for name, _label, options in _offers(source, {**raw, **selected}):
            if not options:
                continue
            value = raw.get(name)
            if name == "result" and isinstance(value, (list, tuple)) and not isinstance(value, str):
                bad = [v for v in value if v not in options]
                if not bad and value:
                    selected[name] = list(value) if len(value) > 1 else value[0]
                    continue
                if name in strict:
                    _refuse(name, bad[0] if bad else value, options)
                selected[name] = options[0]
                continue
            if value is not None:
                value = str(value)
            if value in options:
                selected[name] = value
            elif value is not None and name in strict:
                _refuse(name, value, options)
            else:
                selected[name] = _default(source, name, options, {**raw, **selected})
        raw = {**raw, **selected}
    return selected


def _refuse(name: str, value: Any, options: Sequence[str]) -> None:
    raise ArgumentError(f"{name} is one of {list(options)}, not {value!r}", "plot", name)


def _default(source: Any, name: str, options: Sequence[str], raw: Mapping[str, Any]) -> str:
    from ..pulse import PulseResponse

    if name == "output":
        return options[-1]
    if name == "record" and isinstance(source, PulseResponse):
        return options[-1]
    return options[0]


def _offers(source: Any, raw: Mapping[str, Any]) -> List[Tuple[str, str, List[str]]]:
    """Each control `source` has, as (name, label, choices), given the choices made in `raw`."""
    from ..optimize import OptimizationResult
    from ..pulse import PulseResponse

    if isinstance(source, OptimizationResult):
        return [("panel", "Data", ["objective", "parameters"])]
    if isinstance(source, PulseResponse):
        modes = ["all"] + [str(m) for m in source.modes] if len(source.modes) > 1 else []
        panel = raw.get("panel") or "spectrum"
        return [
            ("panel", "Data", ["spectrum", "spectrum against z", "energy"]),
            ("record", "At", _records(source) if panel == "spectrum" else []),
            ("mode", "Mode", modes if panel != "energy" else []),
        ]
    return _sweep_offers(source, raw)


def _sweep_offers(source: Any, raw: Mapping[str, Any]) -> List[Tuple[str, str, List[str]]]:
    results = plottable_results(source)
    if not results:
        raise EModeError(
            "nothing in this sweep can be plotted: its results "
            f"{source.results} are not numbers, one per swept value"
        )
    result = raw.get("result") or results[0]
    first = result[0] if isinstance(result, (list, tuple)) else result
    if first not in results:
        first = results[0]
    offers: List[Tuple[str, str, List[str]]] = [("result", "Result", results)]
    offers.append(("component", "", _components(source, first)))
    grid = source.layout == "grid"
    if grid:
        offers.append(("panel", "Data", ["map"] + [f"against {_short(k)}" for k in source.swept]))
    if first == "S_matrix":
        labels = _port_labels(source)
        offers.append(("input", "S from", labels))
        offers.append(("output", "to", labels))
        port = labels.index(str(raw["input"])) if raw.get("input") in labels else 0
        sizes = source["S_matrix"][0].port_sizes
        offers.append(("input_mode", "mode", [str(m) for m in range(sizes[port])]))
    elif grid:
        array = _numbers(source, first)
        modes = array.shape[2] if array is not None and array.ndim == 3 else 0
        offers.append(("mode", "Mode", [str(m) for m in range(modes)] if modes > 1 else []))
    return offers


def _records(response: Any) -> List[str]:
    scale, unit = _z_unit(np.asarray(response.z, float))
    return [f"z = {z / scale:g} {unit}" for z in np.asarray(response.z, float)]


# ------------------------------------------------------------------ what a sweep holds


def plottable_results(source: Any) -> List[str]:
    """The results of `source` that are numbers, one per swept value (or per point of a grid)."""
    return [r for r in source.results if r == "S_matrix" and _smatrices(source) or _numbers(source, r) is not None]


def _smatrices(source: Any) -> List[Any]:
    from ..smatrix import SMatrix

    value = source.get("S_matrix")
    if isinstance(value, (list, tuple)) and value and all(isinstance(s, SMatrix) for s in value):
        return list(value)
    return []


def _grid_shape(source: Any) -> Tuple[int, ...]:
    return tuple(len(np.atleast_1d(v)) for v in source["values"])


def _numbers(source: Any, name: str) -> Optional[np.ndarray]:
    """Result `name` as an array of numbers, the swept axes first, or None when it is not one."""
    if name == "S_matrix":
        return None
    try:
        array = np.asarray(source[name])
    except (ValueError, TypeError):
        return None
    if array.dtype.kind not in "biufc":
        return None
    if source.layout == "grid":
        shape = _grid_shape(source)
        if len(shape) != 2 or array.shape[: len(shape)] != shape or array.ndim > len(shape) + 1:
            return None
        return array
    n = len(np.atleast_1d(np.asarray(source["values"])))
    if array.ndim == 2 and array.shape[1] == n and (array.shape[0] != n or "solved_wavelengths" in source):
        # an adaptive scan reports a per-mode result as (modes, wavelengths)
        return array.T
    if array.ndim not in (1, 2) or array.shape[0] != n:
        return None
    return array


def _components(source: Any, name: str) -> List[str]:
    if name == "S_matrix":
        return list(AMPLITUDE_COMPONENTS)
    array = _numbers(source, name)
    if array is None:
        return []
    if array.dtype.kind != "c":
        return list(POWER_COMPONENTS) if name in POWER_RESULTS else []
    return list(INDEX_COMPONENTS) if name in INDEX_RESULTS else list(AMPLITUDE_COMPONENTS)


def _port_labels(source: Any) -> List[str]:
    first = _smatrices(source)[0]
    return [str(n) for n in (first.port_names or [str(i) for i in range(first.num_ports)])]


def tracked(array: np.ndarray, orders: Sequence[Any]) -> np.ndarray:
    """`array` (values by modes) with each column following one mode by `orders`.

    ``orders[i]`` is the ``mode_order`` the sweep reported at value i,
    ``(quality, order)`` with ``order[k]`` the mode at value i that mode k of
    value i - 1 became (None when it matched none). A mode lost on the way is
    NaN from there on.
    """
    out = np.full(array.shape, np.nan, dtype=complex if array.dtype.kind == "c" else float)
    position: List[Optional[int]] = list(range(array.shape[1]))
    for i in range(array.shape[0]):
        if i > 0 and i < len(orders):
            order = list(orders[i][1]) if len(orders[i]) > 1 else list(orders[i])
            position = [order[p] if p is not None and p < len(order) else None for p in position]
        for k, p in enumerate(position):
            if p is not None and p < array.shape[1]:
                out[i, k] = array[i, p]
    return out


def part(values: np.ndarray, component: Optional[str], power: bool = False) -> np.ndarray:
    """The `component` of `values` that is drawn. `power` values are in dB as 10 log10 of themselves."""
    values = np.asarray(values)
    if power and component == "dB":
        with np.errstate(divide="ignore"):
            return 10 * np.log10(np.maximum(np.abs(values), 1e-300))
    if component == "abs^2":
        return np.abs(values) ** 2
    if component == "dB":
        with np.errstate(divide="ignore"):
            return 10 * np.log10(np.maximum(np.abs(values) ** 2, 1e-300))
    if component == "phase":
        return np.angle(values)
    if component == "imag":
        return np.imag(values)
    return np.real(values)


def _component_label(base: str, component: Optional[str], power: bool = False) -> str:
    if power:
        return f"{base} (dB)" if component == "dB" else base
    if component == "abs^2":
        return f"|{base}|²"
    if component == "dB":
        return f"|{base}|² (dB)"
    if component == "phase":
        return f"Phase of {base} (rad)"
    if base == "S" and component in ("real", "imag"):
        return f"{'Re' if component == 'real' else 'Im'} S"
    if component == "imag":
        return f"{base}, imaginary part"
    return base


def _result_label(name: str) -> str:
    return RESULT_LABELS.get(name, name.replace("_", " "))


def _short(key: str) -> str:
    """A swept key as its axis names it: ``'shape, core, width'`` is ``'core width'``."""
    parts = [p.strip() for p in str(key).split(",")]
    if len(parts) >= 3 and parts[0] in ("shape", "section", "setting"):
        return " ".join(parts[-2:]) if parts[0] != "setting" else parts[-1]
    if parts[-1] == "wav":
        return "wavelength"
    return parts[-1]


def _unit(key: str) -> str:
    name = _short(key).split(" ")[-1]
    if name in NM_KEYS:
        return "nm"
    if name in DEGREE_KEYS:
        return "°"
    return ""


def key_scale(key: str, options: PlotOptions) -> Tuple[float, str]:
    """How a key's values are shown: (the divisor, the unit). ``units='um'`` turns nm into μm."""
    unit = _unit(key)
    if unit == "nm" and options.units == "um":
        return 1000.0, "μm"
    return 1.0, unit


def key_label(key: str, unit: Optional[str] = None) -> str:
    """The axis label of a swept key, with its unit (its own by default)."""
    short = _short(key)
    unit = _unit(key) if unit is None else unit
    text = "Wavelength" if short == "wavelength" else short
    return f"{text} ({unit})" if unit else text


def _value_label(key: str, value: float, unit: Optional[str] = None) -> str:
    unit = _unit(key) if unit is None else unit
    return f"{_short(key)} = {value:g}{' ' + unit if unit and unit != '°' else unit}"


# ------------------------------------------------------------------ the figures


def compose_sweep(panel: SweepPanel, options: PlotOptions) -> FigureSpec:
    """The figure of `panel` with `options` applied."""
    from ..optimize import OptimizationResult
    from ..pulse import PulseResponse

    source, selected = panel.source, panel.selected
    if isinstance(source, OptimizationResult):
        return _optimization(source, selected, options)
    if isinstance(source, PulseResponse):
        return _pulse(source, selected, options)
    if not selected:
        selected = _resolve(source, {}, set())
    if source.layout == "grid":
        return _grid(source, selected, options)
    return _line(source, selected, options)


def _color(i: int) -> str:
    palette = colormaps.line_palette()
    return palette[i % len(palette)]


def _trace(x: Any, y: Any, i: int, label: Optional[str] = None, dash: Optional[str] = None) -> LineSpec:
    return LineSpec(
        np.asarray(x, float), np.asarray(y, float), "trace", _color(i), LW * 1.5, dash, label=label
    )


def _mark(kind: str, x: Any, y: float) -> LineSpec:
    marker, color, label = MARKS[kind]
    x = np.asarray(x, float)
    return LineSpec(x, np.full(x.shape, y), kind, _color(color), 0.0, None, 1.0, marker, 4.0, label=label)


def _limits(curves: Sequence[np.ndarray], pad: float = 0.05) -> Tuple[float, float]:
    values = np.concatenate([np.asarray(c, float).ravel() for c in curves]) if curves else np.zeros(0)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi > lo else (abs(hi) or 1.0)
    return lo - pad * span, hi + pad * span


def _figure(
    options: PlotOptions,
    xlabel: str,
    ylabel: str,
    xrange: Tuple[float, float],
    yrange: Tuple[float, float],
    lines: Sequence[LineSpec] = (),
    rasters: Sequence[RasterSpec] = (),
    texts: Sequence[TextSpec] = (),
    colorbar: Optional[ColorbarSpec] = None,
) -> FigureSpec:
    width, height = options.figure_size or DEFAULT_SIZE_MM
    if not rasters:
        # a line panel's scale entries are its y range
        yrange = (
            yrange[0] if options.vmin is None else float(options.vmin),
            yrange[1] if options.vmax is None else float(options.vmax),
        )
    if xrange[0] == xrange[1]:
        xrange = (xrange[0] - 0.5, xrange[1] + 0.5)
    return FigureSpec(
        axes=AxesSpec(
            xlabel=options.xlabel or xlabel,
            ylabel=options.ylabel or ylabel,
            xrange=(float(xrange[0]), float(xrange[1])),
            yrange=(float(yrange[0]), float(yrange[1])),
            grid=not rasters,
        ),
        rasters=tuple(rasters),
        lines=tuple(lines),
        texts=tuple(texts) if options.text_display else (),
        colorbar=colorbar,
        title=options.title,
        width_mm=float(width),
        height_mm=float(height),
        font_size=options.font_size,
        dpi=options.dpi,
    )


def _curves(source: Any, selected: Mapping[str, Any]) -> Tuple[List[Tuple[Optional[str], np.ndarray]], str]:
    """The traces of a one-key sweep's selection, as (label, values in the sweep's order), and the y label."""
    results = selected["result"] if isinstance(selected["result"], list) else [selected["result"]]
    component = selected.get("component")
    curves: List[Tuple[Optional[str], np.ndarray]] = []
    labels = []
    for name in results:
        if name == "S_matrix":
            curves.extend(_smatrix_curves(source, selected, component))
            labels.append(_component_label("S", component))
            continue
        array = _numbers(source, name)
        assert array is not None
        comp = component if component in _components(source, name) else None
        power = name in POWER_RESULTS
        base = _result_label(name)
        label = _component_label(base, comp, power) if comp is not None else base
        labels.append(label)
        if array.ndim == 2:
            if "mode_order" in source and array.shape[1] > 1:
                array = tracked(array, source["mode_order"])
            for k in range(array.shape[1]):
                prefix = f"{label}, " if len(results) > 1 else ""
                curves.append((f"{prefix}mode {k}", part(array[:, k], comp, power)))
        else:
            curves.append((label if len(results) > 1 else None, part(array, comp, power)))
    return curves, ", ".join(dict.fromkeys(labels))


def _smatrix_curves(
    source: Any, selected: Mapping[str, Any], component: Optional[str]
) -> List[Tuple[Optional[str], np.ndarray]]:
    labels = _port_labels(source)
    i_in = labels.index(selected["input"])
    i_out = labels.index(selected["output"])
    m_in = int(selected.get("input_mode") or 0)
    smatrices = _smatrices(source)
    size = max(s.port_sizes[i_out] for s in smatrices)
    entries = np.full((len(smatrices), size), np.nan, dtype=complex)
    for i, s in enumerate(smatrices):
        block = s.block(to_ports=i_out, from_ports=i_in)
        if m_in < block.shape[1]:
            entries[i, : block.shape[0]] = block[:, m_in]
    return [
        (f"{labels[i_in]} {m_in} to {labels[i_out]} {k}", part(entries[:, k], component)) for k in range(size)
    ]


def _line(source: Any, selected: Mapping[str, Any], options: PlotOptions) -> FigureSpec:
    key = source.swept[0]
    scale, unit = key_scale(key, options)
    x = np.asarray(source["values"], float).ravel() / scale
    order = np.argsort(x, kind="stable")
    x = x[order]
    curves, ylabel = _curves(source, selected)
    lines = [_trace(x, y[order], i, label) for i, (label, y) in enumerate(curves)]
    shown = [y for _, y in curves]
    if selected.get("component") == "dB":
        # a null entry is -3000 dB: the axis spans the 60 dB below the peak at most
        peak = max(float(np.nanmax(y)) for y in shown)
        shown = [np.maximum(y, peak - 60.0) for y in shown]
    lo, hi = _limits(shown)
    texts: List[TextSpec] = []
    marks = [
        (kind, [(a / scale, b / scale) for a, b in xs] if kind == "unconverged" else np.asarray(xs, float) / scale)
        for kind, xs in _sweep_marks(source)
    ]
    span = hi - lo
    if source.get("interpolation_residual") is not None:
        hi += 0.12 * span  # room for the residual's line
    if marks:
        lo -= 0.12 * span
        at = lo + 0.05 * span
        for kind, xs in marks:
            if kind == "unconverged":
                lines.append(_unconverged(xs, at))
            else:
                lines.append(_mark(kind, xs, at))
    residual = source.get("interpolation_residual")
    held = (source.get("interpolation_residuals") or {}).get("wavelength")
    if residual is not None and held is not None:
        nodes = source.get("num_nodes")
        text = f"held-out residual {float(residual):.1e} at {float(held):g} nm"
        if nodes is not None:
            text += f", {int(nodes)} nodes"
        texts.append(TextSpec(text, 0.02, 0.98, "axes", "left", "top", "#000000", options.font_size - 1, "residual"))
    return _figure(options, key_label(key, unit), ylabel, (float(x.min()), float(x.max())), (lo, hi), lines, texts=texts)


def _sweep_marks(source: Any) -> List[Tuple[str, Any]]:
    """The sweep's reports drawn on its axis: nodes, the held-out node, solves, unresolved stretches."""
    marks: List[Tuple[str, Any]] = []
    nodes = source.get("node_wavelengths")
    if nodes is not None and len(np.atleast_1d(nodes)):
        marks.append(("node", np.atleast_1d(np.asarray(nodes, float))))
    held = (source.get("interpolation_residuals") or {}).get("wavelength")
    if held is not None:
        marks.append(("held_out", [float(held)]))
    solved = source.get("solved_wavelengths")
    if solved is not None and len(np.atleast_1d(solved)):
        marks.append(("solved", np.atleast_1d(np.asarray(solved, float))))
    unconverged = source.get("unconverged")
    if unconverged:
        marks.append(("unconverged", [(float(u[0]), float(u[1])) for u in unconverged]))
    return marks


def _unconverged(intervals: Sequence[Tuple[float, float]], at: float) -> LineSpec:
    xs: List[float] = []
    for low, high in intervals:
        xs += [low, high, math.nan]
    xs = xs[:-1]
    ys = [math.nan if math.isnan(v) else at for v in xs]
    return LineSpec(np.asarray(xs), np.asarray(ys), "unconverged", _color(1), 3.0, label="unconverged")


def _edges_range(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 1:
        return float(values[0]) - 0.5, float(values[0]) + 0.5
    return float(values[0] - (values[1] - values[0]) / 2), float(values[-1] + (values[-1] - values[-2]) / 2)


def _grid(source: Any, selected: Mapping[str, Any], options: PlotOptions) -> FigureSpec:
    keys = source.swept
    scales = [key_scale(k, options) for k in keys]
    axes = [np.asarray(v, float).ravel() / s for v, (s, _) in zip(source["values"], scales)]
    orders = [np.argsort(a, kind="stable") for a in axes]
    axes = [a[o] for a, o in zip(axes, orders)]
    name = selected["result"][0] if isinstance(selected["result"], list) else selected["result"]
    array = _numbers(source, name)
    assert array is not None
    array = array[np.ix_(orders[0], orders[1])] if array.ndim == 2 else array[orders[0]][:, orders[1]]
    component = selected.get("component") if selected.get("component") in _components(source, name) else None
    base = _result_label(name)
    label = _component_label(base, component) if component else base
    if array.ndim == 3:
        mode = int(selected.get("mode") or 0)
        array = array[:, :, mode]
        label = f"{label}, mode {mode}"
    z = part(array, component)
    panel = selected.get("panel") or "map"
    if panel == "map":
        finite = z[np.isfinite(z)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        vmin = vmin if options.vmin is None else float(options.vmin)
        vmax = vmax if options.vmax is None else float(options.vmax)
        if component == "phase":
            kind = "cyclic"
        elif component in ("real", "imag") and vmin < 0 < vmax:
            kind = "diverging"
        else:
            kind = "sequential"
        cmap = colormaps.resolve(kind, options.colormap)
        raster = RasterSpec(x=axes[0], y=axes[1], z=z.T, colormap=cmap, vmin=vmin, vmax=vmax)
        bar = ColorbarSpec(cmap, vmin, vmax, label=options.colorbar_label or label, tick_format="%.4g")
        return _figure(
            options, key_label(keys[0], scales[0][1]), key_label(keys[1], scales[1][1]),
            _edges_range(axes[0]), _edges_range(axes[1]),
            rasters=[raster], colorbar=bar,
        )
    along = 0 if panel == f"against {_short(keys[0])}" else 1
    other = 1 - along
    curves = z if along == 0 else z.T
    lines = [
        _trace(axes[along], curves[:, j], j, _value_label(keys[other], axes[other][j], scales[other][1]))
        for j in range(curves.shape[1])
    ]
    x = axes[along]
    return _figure(
        options, key_label(keys[along], scales[along][1]), label, (float(x.min()), float(x.max())),
        _limits([curves]), lines,
    )


def _optimization(result: Any, selected: Mapping[str, Any], options: PlotOptions) -> FigureSpec:
    history = list(result.history)
    if not history:
        raise EModeError("this optimization has no history to plot")
    if (selected.get("panel") or "objective") == "parameters":
        lines: List[LineSpec] = []
        curves = []
        x = np.array([e.index for e in history], float)
        for name in history[0].parameters:
            first = history[0].parameters[name]
            count = len(first) if isinstance(first, (list, tuple)) else 0
            for j in range(max(count, 1)):
                y = np.array(
                    [
                        (e.parameters[name][j] if count else e.parameters[name]) if name in e.parameters else math.nan
                        for e in history
                    ],
                    float,
                )
                curves.append(y)
                lines.append(_trace(x, y, len(lines), f"{name}[{j}]" if count else name))
        return _figure(options, "Evaluation", "Parameter value", (float(x.min()), float(x.max())), _limits(curves), lines)
    scored = [e for e in history if e.objective is not None]
    if not scored:
        raise EModeError("no evaluation of this optimization has an objective to plot")
    x = np.array([e.index for e in scored], float)
    y = np.array([e.objective for e in scored], float)
    best_so_far = np.maximum.accumulate(y) if result.goal == "maximize" else np.minimum.accumulate(y)
    lines = [
        _trace(x, y, 0),
        LineSpec(x, best_so_far, "running_best", _color(5), LW * 1.5, "dash", label="best so far"),
    ]
    best = next((e for e in history if e.index == result.best_index and e.objective is not None), None)
    if best is not None:
        lines.append(_mark("best", [best.index], float(best.objective)))
    return _figure(options, "Evaluation", "Objective", (float(x.min()), float(x.max())), _limits([y]), lines)


def _z_unit(z: np.ndarray) -> Tuple[float, str]:
    """The unit z is shown in: mm from 100 μm, else μm."""
    return (1e6, "mm") if z.size and float(np.max(np.abs(z))) >= 1e5 else (1e3, "μm")


def _pulse(response: Any, selected: Mapping[str, Any], options: PlotOptions) -> FigureSpec:
    panel = selected.get("panel") or "spectrum"
    z = np.asarray(response.z, float)
    scale, unit = _z_unit(z)
    mode_choice = selected.get("mode")
    mode = None if mode_choice in (None, "all") else int(mode_choice)
    wavelength = np.asarray(response.wavelength, float)
    order = np.argsort(wavelength)
    w = wavelength[order]
    if panel == "energy":
        energy = np.asarray(response.energy, float)
        lines = [
            _trace(z / scale, energy[:, k], k, f"mode {m}" if len(response.modes) > 1 else None)
            for k, m in enumerate(response.modes)
        ]
        return _figure(options, f"z ({unit})", "Energy (pJ)", (float(z.min() / scale), float(z.max() / scale)), _limits([energy]), lines)
    label = "PSD (dB re 1 pJ/nm)"
    if panel == "spectrum against z":
        rows = np.stack([response.psd(z=zi, mode=mode)[order] for zi in z])
        vmax = float(np.max(rows)) if options.vmax is None else float(options.vmax)
        vmin = vmax - 40.0 if options.vmin is None else float(options.vmin)
        cmap = colormaps.resolve("sequential", options.colormap)
        raster = RasterSpec(x=w, y=z / scale, z=rows, colormap=cmap, vmin=vmin, vmax=vmax)
        bar = ColorbarSpec(cmap, vmin, vmax, label=options.colorbar_label or label, tick_format="%.0f")
        return _figure(
            options, "Wavelength (nm)", f"z ({unit})", (float(w[0]), float(w[-1])),
            _edges_range(z / scale), rasters=[raster], colorbar=bar,
        )
    records = _records(response)
    at = records.index(selected["record"]) if selected.get("record") in records else len(z) - 1
    out = response.psd(z=z[at], mode=mode)[order]
    lines = []
    curves = [out]
    if at != 0:
        launched = response.psd(z=z[0], mode=mode)[order]
        lines.append(_trace(w, launched, 5, "input", "dash"))
        curves.append(launched)
    lines.append(_trace(w, out, 0, "output" if at == len(z) - 1 else records[at]))
    peak = max(float(np.max(c)) for c in curves)
    lo, hi = _limits([np.maximum(c, peak - 60.0) for c in curves])
    return _figure(options, "Wavelength (nm)", label, (float(w[0]), float(w[-1])), (lo, hi), lines)


def sweep_title(panel: SweepPanel) -> str:
    """The panel's name in the plots list."""
    from ..optimize import OptimizationResult
    from ..pulse import PulseResponse

    source, selected = panel.source, panel.selected
    if isinstance(source, OptimizationResult):
        return f"optimization: {selected.get('panel', 'objective')}"
    if isinstance(source, PulseResponse):
        return f"pulse: {selected.get('panel', 'spectrum')}"
    result = selected.get("result")
    result = ", ".join(result) if isinstance(result, list) else result
    return f"sweep {' × '.join(_short(k) for k in source.swept)}: {result}"

