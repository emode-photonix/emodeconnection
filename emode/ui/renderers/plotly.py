"""A ``FigureSpec`` as Plotly figure JSON, for the browser tab.

The tab hands the result to ``Plotly.react`` as it is. Nothing is decided here
that the spec does not already say: this is a mapping from the spec's parts to
Plotly's traces and layout. A raster goes as Plotly's typed-array form
(``{dtype, bdata, shape}``), which plotly.js reads natively and which keeps the
JSON at the array's size. Every coordinate is already in display units.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import numpy as np

from ... import colormaps
from ..figure import FigureSpec, LayerSpec, RasterSpec

#: Plotly's dash names for the spec's.
_DASH = {None: "solid", "dash": "dash", "dot": "dot"}

#: Plotly fill patterns for the spec's hatches.
_HATCH = {"/": "/", "//": "/", "x": "x", "xx": "x"}

#: Points per millimeter, at the browser's 96 px per inch.
PX_PER_MM = 96 / 25.4


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


def _heatmap(r: RasterSpec, showscale: bool, spec: FigureSpec) -> Dict[str, Any]:
    z = np.asarray(r.z, np.float32)
    if r.mask is not None:
        z = np.where(np.asarray(r.mask, bool), np.nan, z).astype(np.float32)
    trace: Dict[str, Any] = {
        "type": "heatmap",
        "x": typed_array(r.x),
        "y": typed_array(r.y),
        "z": typed_array(z),
        "colorscale": colorscale(r.colormap),
        "zmin": r.vmin,
        "zmax": r.vmax,
        "zsmooth": "best" if r.interpolation == "bilinear" else False,
        "showscale": showscale,
        "hovertemplate": "%{x:.4g}, %{y:.4g}: %{z:.4f}<extra></extra>",
    }
    if showscale and spec.colorbar is not None:
        trace["colorbar"] = _colorbar(spec)
    return trace


def _contours(r: RasterSpec) -> List[Dict[str, Any]]:
    """One contour trace per level, since a Plotly contour's levels are evenly spaced."""
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
        out["title"] = {"text": cb.label, "side": "right"}
    return out


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
        "name": layer.label,
        "meta": f"layer:{layer.key}",
        "showlegend": layer.legend,
        **layer_style(layer),
    }


def _rgba(color: str, alpha: float) -> str:
    """A ``#rrggbb`` color at `alpha` as Plotly's ``rgba()`` string, other colors as they are."""
    if alpha >= 1.0 or not (color.startswith("#") and len(color) == 7):
        return color
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha:g})"


def to_plotly(spec: FigureSpec) -> Dict[str, Any]:
    """`spec` as ``{'data': [...], 'layout': {...}}`` for ``Plotly.react``."""
    if spec.panels:
        return _panels(spec)
    data: List[Dict[str, Any]] = []
    for f in spec.fills:
        xs, ys = _closed(np.asarray(f.x), np.asarray(f.y), f.holes)
        trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "fill": "toself",
            "fillcolor": _rgba(f.color, f.alpha) if f.color else "rgba(0,0,0,0)",
            "line": {"width": f.width, "color": f.edgecolor or "rgba(0,0,0,0)"},
            "hoverinfo": "skip",
            "showlegend": False,
            "name": f.kind,
        }
        if f.hatch:
            trace["fillpattern"] = {"shape": _HATCH.get(f.hatch, "/"), "fgcolor": f.edgecolor or "#000000"}
        data.append(trace)

    for layer in spec.layers:
        data.append(_layer_trace(layer))

    for i, r in enumerate(spec.rasters):
        data.append(_heatmap(r, showscale=(i == 0 and spec.colorbar is not None), spec=spec))
    for r in spec.rasters:
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
            "name": line.label if line.label is not None else line.kind,
            "meta": line.kind,
        }
        if line.label is not None:
            trace["hovertemplate"] = "%{x:.6g}, %{y:.6g}<extra>%{fullData.name}</extra>"
            del trace["hoverinfo"]
        if line.marker:
            trace["marker"] = {"symbol": line.marker, "size": line.marker_size, "color": line.color}
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
                "name": entry.label,
                "marker": {"symbol": "square", "size": 10, "color": entry.color or "#ffffff", "line": {"width": 0.5, "color": "#000000"}},
                "showlegend": True,
            }
        )

    return {"data": data, "layout": _layout(spec)}


def _layout(spec: FigureSpec) -> Dict[str, Any]:
    a = spec.axes
    font = {"family": "Arial, Helvetica, sans-serif", "size": spec.font_size + 3}
    yrange = [a.yrange[1], a.yrange[0]] if a.invert_y else list(a.yrange)
    xaxis: Dict[str, Any] = {
        "title": {"text": a.xlabel},
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
        "title": {"text": a.ylabel},
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
            "title": {"text": a.y2label or ""},
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
                "text": t.text.replace("\n", "<br>"),
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
        layout["title"] = {"text": spec.title}
    return layout


def _panels(spec: FigureSpec) -> Dict[str, Any]:
    """A figure of several panels: each panel's traces on its own pair of axes, on one page.

    Panel k (from 1) draws on ``x{k}``/``y{k}`` (``x``/``y`` for the first),
    whose domains tile the page row-major, the first row on top. A panel's
    colorbar sits at the right edge of its own domain, and its title above it.
    """
    rows, cols = spec.panel_grid
    gap_x, gap_y = 0.1, 0.14
    width, height = (1 - gap_x * (cols - 1)) / cols, (1 - gap_y * (rows - 1)) / rows
    data: List[Dict[str, Any]] = []
    layout: Dict[str, Any] = {}
    annotations: List[Dict[str, Any]] = []
    for k, panel in enumerate(spec.panels, start=1):
        row, col = divmod(k - 1, cols)
        x0 = col * (width + gap_x)
        y1 = 1 - row * (height + gap_y)
        xd, yd = [x0, x0 + width], [y1 - height, y1]
        sub = to_plotly(panel)
        suffix = "" if k == 1 else str(k)
        # a panel's twin takes an axis numbered past every panel's own
        twin = str(len(spec.panels) + k)
        for trace in sub["data"]:
            on_twin = trace.get("yaxis") == "y2"
            trace["xaxis"], trace["yaxis"] = f"x{suffix}", f"y{twin if on_twin else suffix}"
            if "colorbar" in trace:
                trace["colorbar"].update({"x": max(0.0, xd[1] - 0.02), "xanchor": "right", "y": (yd[0] + yd[1]) / 2, "len": height})
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
                {"text": panel.title, "x": (xd[0] + xd[1]) / 2, "y": yd[1], "xref": "paper", "yref": "paper",
                 "xanchor": "center", "yanchor": "bottom", "showarrow": False, "name": "panel_title",
                 "font": {"size": spec.font_size + 4}}
            )
    first = _layout(spec.panels[0])
    layout.update(
        {
            "font": first["font"],
            "annotations": annotations,
            "showlegend": any(t.get("showlegend") for t in data),
            "legend": first["legend"],
            "margin": {"l": 60, "r": 60 if any(p.axes.y2range is not None for p in spec.panels) else 20, "t": 60, "b": 50},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "dragmode": "zoom",
        }
    )
    if spec.title:
        layout["title"] = {"text": spec.title}
    return {"data": data, "layout": layout}
