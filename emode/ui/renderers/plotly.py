"""A ``FigureSpec`` as Plotly figure JSON, for the browser tab.

The tab hands the result to ``Plotly.react`` as it is. Nothing is decided here
that the spec does not already say: this is a mapping from the spec's parts to
Plotly's traces and layout. A raster goes as Plotly's typed-array form
(``{dtype, bdata, shape}``), which plotly.js reads natively and which keeps the
JSON at the array's size. Every coordinate is already in display units.
"""
from __future__ import annotations

import base64
import html
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ... import colormaps
from ..figure import FigureSpec, LayerSpec, RasterSpec

#: Plotly's dash names for the spec's.
_DASH = {None: "solid", "dash": "dash", "dot": "dot"}

#: Plotly fill patterns for the spec's hatches.
_HATCH = {"/": "/", "//": "/", "x": "x", "xx": "x"}

#: Points per millimeter, at the browser's 96 px per inch.
PX_PER_MM = 96 / 25.4


# ------------------------------------------------------------ math in labels

#: The bundled plotly has no MathJax, so a label's ``$...$`` math is written in
#: plotly's HTML subset: ``<sub>``, ``<sup>`` and Unicode (emodeconnection#145).
_MATH_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "cdot": "·", "times": "×", "pm": "±", "mp": "∓", "infty": "∞", "approx": "≈", "sim": "~",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "propto": "∝",
    "partial": "∂", "nabla": "∇", "circ": "°", "degree": "°", "prime": "′", "ell": "ℓ",
    "hbar": "ħ", "sqrt": "√", "perp": "⊥", "parallel": "∥", "langle": "⟨", "rangle": "⟩",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "leftrightarrow": "↔", "AA": "Å",
    "vert": "|", "lvert": "|", "rvert": "|", "%": "%", "_": "_", "&": "&amp;", "#": "#",
    "{": "{", "}": "}", "$": "$",
}
#: Font commands: their argument is kept, the font is not.
_MATH_FONTS = {"mathrm", "text", "textrm", "mathbf", "mathit", "mathsf", "rm", "operatorname", "boldsymbol", "mathcal"}
_MATH_SPACES = {",": " ", ";": " ", ":": " ", " ": " ", "!": "", "quad": " ", "qquad": "  "}
#: Sizing commands before a delimiter: dropped, the delimiter kept.
_MATH_SIZES = {"left", "right", "big", "Big", "bigl", "bigr", "Bigl", "Bigr"}


def _math_command(s: str, i: int) -> Tuple[str, int]:
    """The command name after the backslash at ``s[i - 1]``, and the index after it."""
    if i < len(s) and s[i].isalpha():
        j = i
        while j < len(s) and s[j].isalpha():
            j += 1
        name, i = s[i:j], j
        if i < len(s) and s[i] == " ":  # TeX swallows the space that ends a command name
            i += 1
        return name, i
    return (s[i], i + 1) if i < len(s) else ("", i)


def _math_argument(s: str, i: int) -> Tuple[str, int]:
    """One argument of ``^``, ``_`` or a font command: a ``{group}``, a command or one character."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return _math(s, i + 1, closing=True)
    if s[i] == "\\":
        return _math_token(s, i)
    return html.escape(s[i], quote=False), i + 1


def _math_token(s: str, i: int) -> Tuple[str, int]:
    """The HTML of the backslash command at ``s[i]``."""
    name, i = _math_command(s, i + 1)
    if name in _MATH_FONTS:
        return _math_argument(s, i)
    if name in _MATH_SPACES:
        return _MATH_SPACES[name], i
    if name in _MATH_SIZES:
        return "", i
    if name in _MATH_SYMBOLS:
        return _MATH_SYMBOLS[name], i
    return html.escape(name, quote=False), i


def _math(s: str, i: int = 0, closing: bool = False) -> Tuple[str, int]:
    """The HTML of math from ``s[i]`` to the end, or to the ``}`` closing a group."""
    out: List[str] = []
    while i < len(s):
        c = s[i]
        if c == "}" and closing:
            return "".join(out), i + 1
        if c == "{":
            part, i = _math(s, i + 1, closing=True)
            out.append(part)
        elif c in "^_":
            part, i = _math_argument(s, i + 1)
            tag = "sup" if c == "^" else "sub"
            out.append(f"<{tag}>{part}</{tag}>")
        elif c == "\\":
            part, i = _math_token(s, i)
            out.append(part)
        else:
            out.append(html.escape(c, quote=False))
            i += 1
    return "".join(out), i


def math_html(text: Optional[str]) -> Optional[str]:
    """`text` for plotly: its ``$...$`` math as HTML, the rest escaped, and ``\\$`` a dollar sign.

    Covers what labels write: subscripts and superscripts, Greek letters and
    common symbols, ``\\mathrm`` and the other font commands (the font is
    dropped). Anything else in the math shows as its command's name, never as
    a dollar sign. Line breaks are left for the caller.
    """
    if not text or ("$" not in text and "\\" not in text and "<" not in text and "&" not in text and ">" not in text):
        return text
    out: List[str] = []
    plain: List[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and text[i + 1 : i + 2] == "$":
            plain.append("$")
            i += 2
            continue
        if c == "$":
            end = i + 1
            while end < len(text) and not (text[end] == "$" and text[end - 1] != "\\"):
                end += 1
            if end < len(text):
                out.append(html.escape("".join(plain), quote=False))
                plain = []
                out.append(_math(text[i + 1 : end])[0])
                i = end + 1
                continue
        plain.append(c)
        i += 1
    out.append(html.escape("".join(plain), quote=False))
    return "".join(out)


def typed_array(values: Any, dtype: str = "f4") -> Dict[str, Any]:
    """`values` in Plotly's base64 typed-array form."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    out: Dict[str, Any] = {"dtype": dtype, "bdata": base64.b64encode(array.tobytes()).decode()}
    if array.ndim > 1:
        out["shape"] = ",".join(str(n) for n in array.shape)
    return out


def colorscale(name: str) -> Any:
    """The family's stops for one of its names or kinds, or another colormap sampled into a colorscale."""
    if colormaps.kind_of(name) is not None:
        return colormaps.plotly_colorscale(name)
    from matplotlib.colors import to_hex

    cmap = colormaps.get(name)
    return [[i / 16, to_hex(cmap(i / 16))] for i in range(17)]


#: The significant digits a field value is sent with: past the fourth, a colormap
#: shows no difference and the readout needs none.
DIGITS = 4

#: The float32 mantissa bits that carry more than `DIGITS` significant digits
#: (14 of its 23 bits hold 4.2 digits), zeroed so the bytes compress.
_LOW_BITS = np.uint32((1 << (23 - 14)) - 1)


def round_digits(values: Any) -> np.ndarray:
    """`values` as float32 rounded to about `DIGITS` significant digits, NaN kept.

    The mantissa is rounded to its top 14 bits (a relative step of 6e-5), which
    is faster than a decimal rounding and leaves the low bits zero.
    """
    a = np.ascontiguousarray(np.asarray(values, np.float32))
    bits = a.view(np.uint32)
    finite = np.isfinite(a)
    rounded = (bits + (_LOW_BITS >> np.uint32(1)) + np.uint32(1)) & ~_LOW_BITS
    return np.where(finite, rounded, bits).astype(np.uint32).view(np.float32)


def _stride(n: int, room: Optional[int]) -> int:
    """The step that keeps at most `room` of `n` samples, 1 when they fit."""
    if not room or room <= 0 or n <= room:
        return 1
    return int(np.ceil(n / room))


def decimated(r: RasterSpec, pixels: Optional[Tuple[int, int]]) -> RasterSpec:
    """`r` with no more samples than the panel's `pixels` (width, height), None keeping them all.

    Every k-th row and column is kept, which a colormap at about one sample
    per pixel shows as the full raster does, and the contours are traced on
    what is shown. A z panel's joined slices keep their own column edges, so
    they are not decimated (EMode already samples them at the panel's pixels).
    """
    z = np.asarray(r.z)
    if pixels is None or r.x_edges is not None or z.ndim != 2:
        return r
    sx, sy = _stride(z.shape[1], int(pixels[0])), _stride(z.shape[0], int(pixels[1]))
    if sx == 1 and sy == 1:
        return r
    return replace(
        r,
        x=np.asarray(r.x)[::sx],
        y=np.asarray(r.y)[::sy],
        z=z[::sy, ::sx],
        mask=None if r.mask is None else np.asarray(r.mask)[::sy, ::sx],
    )


def _heatmap(r: RasterSpec, showscale: bool, spec: FigureSpec) -> Dict[str, Any]:
    z = np.asarray(r.z, np.float32)
    if r.mask is not None:
        z = np.where(np.asarray(r.mask, bool), np.nan, z).astype(np.float32)
    trace: Dict[str, Any] = {
        "type": "heatmap",
        # the columns' own edges when the image joins slices, else their centers
        "x": typed_array(r.x if r.x_edges is None else r.x_edges),
        "y": typed_array(r.y),
        "z": typed_array(round_digits(z)),
        "colorscale": colorscale(r.colormap),
        "zmin": r.vmin,
        "zmax": r.vmax,
        "zsmooth": "best" if r.interpolation == "bilinear" else False,
        "showscale": showscale,
        "hovertemplate": "%{x:.4g}, %{y:.4g}: %{z:.4f}<extra></extra>",
    }
    if showscale and spec.colorbar is not None:
        trace["colorbar"] = _colorbar(spec)
    if r.target:
        trace["meta"] = r.target
    return trace


#: A contour line's color on each side of zero, as the window drew them.
_CONTOUR_COLORS = ("#000000", "#ffffff")


def _contours(r: RasterSpec) -> List[Dict[str, Any]]:
    """The raster's contour lines: one trace for the levels at or above zero, one for those below.

    The lines are traced here, on the full-resolution field, and sent as
    points with a gap (NaN) between lines. Plotly's own contour trace would
    need the whole field once per level, as its levels are evenly spaced, and
    the browser would trace each one again (24 copies of the field for a
    signed mode). Without contourpy (matplotlib's) the levels go as Plotly
    contours, as before.
    """
    if not r.contour_levels:
        return []
    try:
        from contourpy import LineType, contour_generator
    except ImportError:
        return _plotly_contours(r)
    z = np.asarray(r.z, np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        return []
    bad = ~np.isfinite(z)
    if r.mask is not None:
        bad |= np.asarray(r.mask, bool)
    field = np.ma.array(z, mask=bad) if bad.any() else z
    gen = contour_generator(
        x=np.asarray(r.x, np.float64), y=np.asarray(r.y, np.float64), z=field, line_type=LineType.ChunkCombinedNan
    )
    out = []
    for color, levels in zip(_CONTOUR_COLORS, ([v for v in r.contour_levels if v >= 0], [v for v in r.contour_levels if v < 0])):
        pieces: List[np.ndarray] = []
        for level in levels:
            for chunk in gen.lines(level)[0]:
                if chunk is not None and len(chunk):
                    pieces += [chunk, np.full((1, 2), np.nan)]
        if not pieces:
            continue
        xy = np.concatenate(pieces)
        out.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": typed_array(round_digits(xy[:, 0])),
                "y": typed_array(round_digits(xy[:, 1])),
                "line": {"color": color, "width": 0.5},
                "opacity": 0.7,
                "connectgaps": False,
                "hoverinfo": "skip",
                "showlegend": False,
                "name": "contour",
                "meta": "contour",
            }
        )
    return out


def _plotly_contours(r: RasterSpec) -> List[Dict[str, Any]]:
    """One Plotly contour trace per level, since a Plotly contour's levels are evenly spaced."""
    out = []
    for level in r.contour_levels:
        out.append(
            {
                "type": "contour",
                "x": typed_array(r.x),
                "y": typed_array(r.y),
                "z": typed_array(r.z),
                "contours": {"start": level, "end": level, "size": 1.0, "coloring": "none"},
                "line": {"color": "#000000" if level >= 0 else "#ffffff", "width": 0.5},
                "opacity": 0.7,
                "showscale": False,
                "hoverinfo": "skip",
            }
        )
    return out


def _colorbar(spec: FigureSpec) -> Dict[str, Any]:
    cb = spec.colorbar
    assert cb is not None
    out: Dict[str, Any] = {"thickness": 12, "outlinewidth": 0.5, "tickfont": {"size": spec.font_size + 2}}
    if cb.ticks is not None:
        out["tickvals"] = list(cb.ticks)
        if cb.tick_labels is not None:
            out["ticktext"] = list(cb.tick_labels)
    if cb.label:
        out["title"] = {"text": math_html(cb.label), "side": "right"}
    return out


#: The colorbar's thickness in pixels, and where a single panel's sits in paper fractions.
_BAR_THICKNESS = 12
_BAR_X = 1.02


def _colorbar_lines(
    cb: Any, x: float = _BAR_X, xanchor: str = "left", y: float = 0.5, length: float = 1.0
) -> List[Dict[str, Any]]:
    """A line across the colorbar at each contour level (a Plotly colorbar has no such lines).

    The shapes are anchored at the bar's edge in paper fractions and sized in pixels
    across its thickness, so they follow the bar when the tab resizes.
    """
    span = cb.vmax - cb.vmin
    if not cb.lines or span <= 0:
        return []
    start, end = (0, _BAR_THICKNESS) if xanchor == "left" else (-_BAR_THICKNESS, 0)
    shapes = []
    for level in cb.lines:
        t = (level - cb.vmin) / span
        if not 0.0 <= t <= 1.0:
            continue
        pos = y + (t - 0.5) * length
        shapes.append(
            {
                "type": "line", "name": "colorbar_level", "layer": "above",
                "xref": "paper", "xsizemode": "pixel", "xanchor": x, "x0": start, "x1": end,
                "yref": "paper", "y0": pos, "y1": pos,
                "line": {"color": "#000000", "width": 0.8},
            }
        )
    return shapes


def _closed(x: np.ndarray, y: np.ndarray, holes: Any) -> tuple:
    """A polygon and its holes as one path, each ring closed and separated by a gap."""
    xs: List[Optional[float]] = [float(v) for v in x] + [float(x[0])]
    ys: List[Optional[float]] = [float(v) for v in y] + [float(y[0])]
    for h in holes:
        h = np.asarray(h)
        xs += [None] + [float(v) for v in h[:, 0]] + [float(h[0, 0])]
        ys += [None] + [float(v) for v in h[:, 1]] + [float(h[0, 1])]
    return xs, ys


def _rgba(color: str, alpha: float) -> str:
    """``'#rrggbb'`` (or ``'#rgb'``) with `alpha`, as Plotly's ``rgba(r,g,b,a)``."""
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


def layer_style(layer: LayerSpec) -> Dict[str, Any]:
    """The part of a layer's trace its style sets, which the tab's layer panel restyles in place.

    ``app.js``'s ``layerStyle()`` makes the same from a panel row, so a change in
    the panel redraws in the browser without asking this client for a figure.
    """
    fill = _rgba(layer.color, layer.opacity if layer.filled else 0.0)
    style: Dict[str, Any] = {
        "visible": layer.visible,
        "fillcolor": fill,
        "line": {"color": layer.color, "width": layer.width, "dash": _DASH.get(layer.dash, "solid")},
    }
    if layer.filled:
        style["fillpattern"] = {"shape": layer.hatch, "fgcolor": layer.color, "bgcolor": fill, "fillmode": "replace"}
    return style


def _layer_trace(layer: LayerSpec) -> Dict[str, Any]:
    """One trace for every polygon of a layer, each ring closed and the rings separated by gaps."""
    xs: List[np.ndarray] = []
    for i, ring in enumerate(layer.polygons):
        ring = np.asarray(ring, float)
        if i:
            xs.append(np.full((1, 2), np.nan))
        xs.append(np.vstack([ring, ring[:1]]))
    xy = np.concatenate(xs) if xs else np.zeros((0, 2))
    return {
        "type": "scatter",
        "mode": "lines",
        "x": typed_array(xy[:, 0]),
        "y": typed_array(xy[:, 1]),
        "fill": "toself" if layer.filled else "none",
        "connectgaps": False,
        "hoverinfo": "name",
        "hoveron": "fills",
        "name": math_html(layer.label),
        "meta": f"layer:{layer.key}",
        "showlegend": layer.legend,
        **layer_style(layer),
    }


def _target_text(target: str) -> str:
    """The hover text of a schematic component: what a tap shows."""
    kind, _, name = target.partition(":")
    return f"{name} ({kind}), tap for details"


def to_plotly(spec: FigureSpec, pixels: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """`spec` as ``{'data': [...], 'layout': {...}}`` for ``Plotly.react``.

    `pixels` is the plot's size on screen (width, height): a raster with more
    samples than that is decimated to it. None sends every sample.
    """
    if spec.panels:
        return _panels(spec, pixels)
    data: List[Dict[str, Any]] = []
    for f in spec.fills:
        xs, ys = _closed(np.asarray(f.x), np.asarray(f.y), f.holes)
        trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "fill": "toself",
            "fillcolor": f.color or "rgba(0,0,0,0)",
            "line": {"width": f.width, "color": f.edgecolor or "rgba(0,0,0,0)"},
            "hoverinfo": "skip",
            "showlegend": False,
            "name": f.kind,
        }
        if f.target:
            # a schematic block: a tap on its fill opens its details (#159)
            trace.update({"hoveron": "fills", "hoverinfo": "text", "text": _target_text(f.target),
                          "meta": f"component:{f.target}"})
        if f.hatch:
            trace["fillpattern"] = {"shape": _HATCH.get(f.hatch, "/"), "fgcolor": f.edgecolor or "#000000"}
        data.append(trace)

    for layer in spec.layers:
        data.append(_layer_trace(layer))

    rasters = [decimated(r, pixels) for r in spec.rasters]
    for i, r in enumerate(rasters):
        data.append(_heatmap(r, showscale=(i == 0 and spec.colorbar is not None), spec=spec))
    for r in rasters:
        data.extend(_contours(r))

    for s in spec.segments:
        xy = np.asarray(s.xy)
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        for (x0, y0), (x1, y1) in xy:
            xs += [float(x0), float(x1), None]
            ys += [float(y0), float(y1), None]
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": xs,
                "y": ys,
                "line": {"width": s.width, "color": s.color},
                "opacity": s.alpha,
                "hoverinfo": "skip",
                "showlegend": False,
                "name": s.kind,
            }
        )

    for line in spec.lines:
        trace = {
            "type": "scatter",
            "mode": "markers" if line.marker else "lines",
            "x": typed_array(line.x),
            "y": typed_array(line.y),
            "line": {"width": line.width, "color": line.color, "dash": _DASH.get(line.dash, "solid")},
            "opacity": line.alpha,
            "hoverinfo": "skip",
            "showlegend": line.label is not None,
            "name": math_html(line.label) if line.label is not None else line.kind,
            "meta": line.kind,
        }
        if line.label is not None:
            trace["hovertemplate"] = "%{x:.6g}, %{y:.6g}<extra>%{fullData.name}</extra>"
            del trace["hoverinfo"]
        if line.marker:
            trace["marker"] = {"symbol": line.marker, "size": line.marker_size, "color": line.color}
        if line.target:
            # a port or a connection of the schematic: its points open its details (#159)
            trace.update({"mode": "lines+markers", "marker": {"size": 8, "color": line.color},
                          "hoverinfo": "text", "text": _target_text(line.target),
                          "meta": f"component:{line.target}"})
        if line.axis == "right" and spec.axes.y2range is not None:
            trace["yaxis"] = "y2"
        data.append(trace)

    if spec.colorbar is not None and not spec.rasters:
        # a panel of fills only (the index panel): an invisible trace carries the colorbar
        cb = spec.colorbar
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [None],
                "y": [None],
                "marker": {
                    "colorscale": colorscale(cb.colormap),
                    "cmin": cb.vmin,
                    "cmax": cb.vmax,
                    "color": [cb.vmin],
                    "showscale": True,
                    "colorbar": _colorbar(spec),
                },
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )

    for entry in spec.legend:
        data.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [None],
                "y": [None],
                "name": math_html(entry.label),
                "marker": {"symbol": "square", "size": 10, "color": entry.color or "#ffffff", "line": {"width": 0.5, "color": "#000000"}},
                "showlegend": True,
            }
        )

    layout = _layout(spec)
    if spec.colorbar is not None and spec.rasters:
        shapes = _colorbar_lines(spec.colorbar)
        if shapes:
            layout["shapes"] = shapes
    return {"data": data, "layout": layout}


def _layout(spec: FigureSpec) -> Dict[str, Any]:
    a = spec.axes
    font = {"family": "Arial, Helvetica, sans-serif", "size": spec.font_size + 3}
    yrange = [a.yrange[1], a.yrange[0]] if a.invert_y else list(a.yrange)
    xaxis: Dict[str, Any] = {
        "title": {"text": math_html(a.xlabel)},
        "range": list(a.xrange),
        "ticks": "inside",
        "mirror": "ticks",
        "showline": True,
        "linewidth": 0.5,
        "showgrid": False,
        "zeroline": False,
        "visible": a.visible,
    }
    yaxis: Dict[str, Any] = {
        "title": {"text": math_html(a.ylabel)},
        "range": yrange,
        "ticks": "inside",
        "mirror": "ticks",
        "showline": True,
        "linewidth": 0.5,
        "showgrid": False,
        "zeroline": False,
        "visible": a.visible,
    }
    yaxis2: Optional[Dict[str, Any]] = None
    if a.y2range is not None:
        # the panel's twin: its own range and label on the right, the left axis's ticks only on the left
        yaxis["mirror"] = False
        yaxis2 = {
            "title": {"text": math_html(a.y2label) or ""},
            "range": list(a.y2range),
            "overlaying": "y",
            "side": "right",
            "anchor": "x",
            "ticks": "inside",
            "showline": True,
            "linewidth": 0.5,
            "showgrid": False,
            "zeroline": False,
            "visible": a.visible,
        }
    for axis, color in ((yaxis, a.ylabel_color), (yaxis2, a.y2label_color)):
        if axis is not None and color:
            axis["title"]["font"] = {"color": color}
            axis["tickfont"] = {"color": color}
    if a.equal_aspect:
        yaxis["scaleanchor"] = "x"
        yaxis["scaleratio"] = 1
    if a.grid:
        for axis in (xaxis, yaxis):
            axis.update({"showgrid": True, "gridcolor": "rgba(128,128,128,0.25)", "gridwidth": 0.5})
    if a.integer_ticks:
        for axis in (xaxis, yaxis):
            axis.update({"tickformat": "d", "dtick": 1})
    annotations = []
    for t in spec.texts:
        ref = {"xref": "x domain", "yref": "y domain"} if t.frame == "axes" else {"xref": "x", "yref": "y"}
        annotations.append(
            {
                "text": (math_html(t.text) or "").replace("\n", "<br>"),
                "x": t.x,
                "y": t.y,
                **ref,
                "xanchor": {"left": "left", "right": "right", "center": "center"}[t.ha],
                "yanchor": {"top": "top", "bottom": "bottom", "center": "middle", "baseline": "bottom"}[t.va],
                "showarrow": False,
                "font": {"color": t.color, "size": t.size + 3, "weight": 700 if t.weight == "bold" else 400},
                "name": t.kind,
            }
        )
    glyphs = {"left": "◀", "right": "▶", "up": "▲", "down": "▼"}
    for arrow in spec.arrows:
        annotations.append(
            {
                "text": glyphs[arrow.direction],
                "x": arrow.x,
                "y": arrow.y,
                "xref": "x domain",
                "yref": "y domain",
                "showarrow": False,
                "font": {"color": "#d1495b", "size": 16},
                "bgcolor": "rgba(255,255,255,0.85)",
                "bordercolor": "#d1495b",
                "name": "arrow",
                "captureevents": True,
                "hovertext": f"go to {arrow.target}",
            }
        )
    layout: Dict[str, Any] = {
        "xaxis": xaxis,
        "yaxis": yaxis,
        "font": font,
        "annotations": annotations,
        "showlegend": bool(spec.legend)
        or any(layer.legend for layer in spec.layers)
        or any(line.label is not None for line in spec.lines),
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.02, "yanchor": "bottom"},
        "margin": {"l": 60, "r": 20, "t": 40, "b": 110 if any(t.kind == "footer" for t in spec.texts) else 50},
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": a.background or "#ffffff",
        "dragmode": "zoom",
    }
    if yaxis2 is not None:
        layout["yaxis2"] = yaxis2
        layout["margin"]["r"] = 60
    if spec.title:
        layout["title"] = {"text": math_html(spec.title)}
    return layout


def _panels(spec: FigureSpec, pixels: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """A figure of several panels: each panel's traces on its own pair of axes, on one page.

    Panel k (from 1) draws on ``x{k}``/``y{k}`` (``x``/``y`` for the first),
    whose domains tile the page row-major, the first row on top. A panel's
    colorbar sits at the right edge of its own domain, and its title above it.
    """
    rows, cols = spec.panel_grid
    # the gaps hold a panel's title and tick labels: a tall grid gives each a smaller share
    gap_x, gap_y = 0.1 if cols <= 2 else 0.2 / cols, 0.14 if rows <= 2 else 0.3 / rows
    width, height = (1 - gap_x * (cols - 1)) / cols, (1 - gap_y * (rows - 1)) / rows
    data: List[Dict[str, Any]] = []
    layout: Dict[str, Any] = {}
    annotations: List[Dict[str, Any]] = []
    shapes: List[Dict[str, Any]] = []
    for k, panel in enumerate(spec.panels, start=1):
        row, col = divmod(k - 1, cols)
        x0 = col * (width + gap_x)
        y1 = 1 - row * (height + gap_y)
        xd, yd = [x0, x0 + width], [y1 - height, y1]
        sub = to_plotly(panel, None if pixels is None else (max(1, int(pixels[0] * width)), max(1, int(pixels[1] * height))))
        suffix = "" if k == 1 else str(k)
        # a panel's twin takes an axis numbered past every panel's own
        twin = str(len(spec.panels) + k)
        for trace in sub["data"]:
            on_twin = trace.get("yaxis") == "y2"
            trace["xaxis"], trace["yaxis"] = f"x{suffix}", f"y{twin if on_twin else suffix}"
            if "colorbar" in trace:
                trace["colorbar"].update({"x": max(0.0, xd[1] - 0.02), "xanchor": "right", "y": (yd[0] + yd[1]) / 2, "len": height})
                if panel.colorbar is not None:
                    shapes += _colorbar_lines(
                        panel.colorbar, max(0.0, xd[1] - 0.02), "right", (yd[0] + yd[1]) / 2, height
                    )
            data.append(trace)
        xaxis, yaxis = dict(sub["layout"]["xaxis"]), dict(sub["layout"]["yaxis"])
        xaxis.update({"domain": xd, "anchor": f"y{suffix}"})
        yaxis.update({"domain": yd, "anchor": f"x{suffix}"})
        if "scaleanchor" in yaxis:
            yaxis["scaleanchor"] = f"x{suffix}"
        layout[f"xaxis{suffix}"], layout[f"yaxis{suffix}"] = xaxis, yaxis
        if "yaxis2" in sub["layout"]:
            right = dict(sub["layout"]["yaxis2"])
            right.update({"domain": yd, "anchor": f"x{suffix}", "overlaying": f"y{suffix}"})
            layout[f"yaxis{twin}"] = right
        for note in sub["layout"].get("annotations", []):
            for ref in ("xref", "yref"):
                axis = note.get(ref, "paper")
                if axis != "paper":
                    note[ref] = axis[0] + suffix + axis[1:]
            annotations.append(note)
        if panel.title:
            annotations.append(
                {"text": math_html(panel.title), "x": (xd[0] + xd[1]) / 2, "y": yd[1], "xref": "paper", "yref": "paper",
                 "xanchor": "center", "yanchor": "bottom", "showarrow": False, "name": "panel_title",
                 "font": {"size": spec.font_size + 4}}
            )
    first = _layout(spec.panels[0])
    layout.update(
        {
            "font": first["font"],
            "annotations": annotations,
            **({"shapes": shapes} if shapes else {}),
            "showlegend": any(t.get("showlegend") for t in data),
            "legend": first["legend"],
            "margin": {"l": 60, "r": 60 if any(p.axes.y2range is not None for p in spec.panels) else 20, "t": 60, "b": 50},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "dragmode": "zoom",
        }
    )
    if spec.title:
        layout["title"] = {"text": math_html(spec.title)}
    return {"data": data, "layout": layout}
