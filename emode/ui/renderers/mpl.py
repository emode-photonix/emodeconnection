"""A ``FigureSpec`` as a matplotlib figure, for a static image without a browser.

``plot(file_name=...)``, ``EMODE_HEADLESS=1``, the MCP server's plot tool and
``PlotHandle.png()`` all draw here, from the same spec the browser tab draws,
so a saved file and the tab agree. matplotlib is imported only when a figure is
drawn, never by ``import emode``. A client without it gets an ``EModeError``
naming the command that installs it, and the browser tab never needs it.

The figure is built with ``matplotlib.figure.Figure`` and an Agg canvas, never
``pyplot``, so drawing one opens no window and touches no global state.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...types import EModeError
from ..figure import FigureSpec

#: The resolution of the PNG a headless ``plot()`` returns.
INLINE_DPI = 150

MISSING = "Saving a figure needs matplotlib. Install it with `pip install matplotlib`."

_MARKERS = {
    "triangle-right": ">",
    "triangle-left": "<",
    "triangle-up": "^",
    "circle": "o",
    "square": "s",
    "diamond": "D",
}
_DASHES = {None: "-", "dash": "--", "dot": ":"}


def _matplotlib() -> Any:
    try:
        import matplotlib  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401
    except ImportError as exc:
        raise EModeError(MISSING) from exc
    import matplotlib

    return matplotlib


def _edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges around monotonic centers (the outer edges half a cell out)."""
    c = np.asarray(centers, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[1:] + c[:-1]) / 2
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def to_figure(spec: FigureSpec, fig: Any = None, ax: Any = None) -> Any:
    """`spec` drawn on a new matplotlib ``Figure`` with an Agg canvas (or on `ax` of `fig`)."""
    _matplotlib()
    if spec.panels and ax is None:
        return _panels(spec)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.image import NonUniformImage
    from matplotlib.patches import Patch, PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    from ... import colormaps

    def cmap_of(name: str) -> Any:
        return colormaps.get(name)

    if ax is None:
        fig = Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=spec.dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
    a = spec.axes
    if a.background:
        ax.set_facecolor(a.background)

    from matplotlib.colors import to_rgba

    for f in spec.fills:
        rings = [np.column_stack([f.x, f.y])] + [np.asarray(h) for h in f.holes]
        vertices = np.concatenate([np.vstack([r, r[:1]]) for r in rings])
        codes = np.concatenate(
            [[MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 1) + [MplPath.CLOSEPOLY] for r in rings]
        )
        ax.add_patch(
            PathPatch(
                MplPath(vertices, codes),
                facecolor=to_rgba(f.color, f.alpha) if f.color else "none",
                edgecolor=f.edgecolor or "none",
                lw=f.width,
                hatch=f.hatch,
                gid=f.kind,
            )
        )

    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba

    for layer in spec.layers:
        collection = PolyCollection(
            [np.asarray(p, float) for p in layer.polygons],
            closed=True,
            facecolors=[to_rgba(layer.color, layer.opacity)] if layer.filled else "none",
            edgecolors=layer.color,
            linewidths=layer.width,
            linestyles=_DASHES.get(layer.dash, "-"),
            hatch=layer.hatch or None,
            label=layer.label,
            gid=f"layer:{layer.key}",
        )
        collection.set_visible(layer.visible)
        ax.add_collection(collection)

    for r in spec.rasters:
        z = np.asarray(r.z, float)
        if r.mask is not None:
            # off the panel: NaN, which the colormap leaves transparent
            z = np.where(np.asarray(r.mask, bool), np.nan, z)
        x, y = np.asarray(r.x, float), np.asarray(r.y, float)
        x0, x1 = _edges(x)[[0, -1]]
        y0, y1 = _edges(y)[[0, -1]]
        im = NonUniformImage(
            ax,
            interpolation="bilinear" if r.interpolation == "bilinear" else "nearest",
            extent=(x0, x1, y0, y1),
            cmap=cmap_of(r.colormap),
            norm=Normalize(vmin=r.vmin, vmax=r.vmax),
        )
        im.set_data(x, y, z)
        # A NonUniformImage paints the whole axes, carrying its edge cells out to
        # the panel's edges. A z panel is one image per slice, so without this
        # clip the last slice drawn covered every other one.
        im.set_clip_path(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transData))
        ax.add_image(im)
        if r.contour_levels and min(z.shape) >= 2:
            for level in r.contour_levels:
                ax.contour(
                    x, y, z, levels=[level], colors="k" if level >= 0 else "w",
                    linewidths=0.2, alpha=0.7, linestyles="solid",
                )

    for s in spec.segments:
        ax.add_collection(
            LineCollection(np.asarray(s.xy), colors=s.color, linewidths=s.width, alpha=s.alpha, gid=s.kind)
        )

    twin = ax.twinx() if a.y2range is not None else None
    for line in spec.lines:
        _draw_line(twin if (line.axis == "right" and twin is not None) else ax, line)

    for t in spec.texts:
        kwargs = {"transform": ax.transAxes} if t.frame == "axes" else {}
        ax.text(
            t.x, t.y, t.text, ha=t.ha, va=t.va, color=t.color, fontsize=t.size,
            fontweight=t.weight, gid=t.kind, **kwargs,
        )

    glyphs = {"left": "◀", "right": "▶", "up": "▲", "down": "▼"}
    for arrow in spec.arrows:
        ax.text(
            arrow.x, arrow.y, glyphs[arrow.direction], transform=ax.transAxes,
            ha="center", va="center", fontsize=14, color="#d1495b", gid="arrow",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1495b", "alpha": 0.85},
        )

    ax.set_xlim(*a.xrange)
    ax.set_ylim(*a.yrange)
    if a.invert_y and not ax.yaxis_inverted():
        ax.invert_yaxis()
    if a.equal_aspect:
        ax.set_aspect("equal")
    elif a.box_aspect is not None:
        ax.set_box_aspect(a.box_aspect)
    if a.grid:
        ax.grid(visible=True, which="major", axis="both", linewidth=0.25, color="grey", alpha=0.25)
    if a.integer_ticks:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if not a.visible:
        ax.axis("off")
    else:
        ax.set_xlabel(a.xlabel, fontsize=spec.font_size)
        ax.set_ylabel(a.ylabel, fontsize=spec.font_size)
        ax.tick_params(
            axis="both", direction="in", labelsize=spec.font_size, top=True, right=twin is None, length=3, width=0.5
        )
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_linewidth(0.5)
        if a.ylabel_color:
            ax.yaxis.label.set_color(a.ylabel_color)
            ax.tick_params(axis="y", colors=a.ylabel_color)
    if twin is not None:
        twin.set_ylim(*a.y2range)
        twin.set_ylabel(a.y2label or "", fontsize=spec.font_size)
        twin.tick_params(axis="y", direction="in", labelsize=spec.font_size, length=3, width=0.5)
        for side in ("top", "bottom", "left", "right"):
            twin.spines[side].set_linewidth(0.5)
        if a.y2label_color:
            twin.yaxis.label.set_color(a.y2label_color)
            twin.tick_params(axis="y", colors=a.y2label_color)
        # The legend goes on the twin, and matplotlib keeps a legend off its own
        # axes' lines only, so the left curves stand in there, transparent and
        # mapped onto the right axis's scale.
        (y0, y1), (r0, r1) = a.yrange, a.y2range
        scale = (r1 - r0) / (y1 - y0) if y1 != y0 else 1.0
        for line in spec.lines:
            if line.axis == "left" and line.kind in ("user", "trace"):
                twin.plot(
                    line.x, r0 + (np.asarray(line.y, float) - y0) * scale,
                    alpha=0.0, label="_nolegend_", gid="legend_avoid",
                )
        twin.set_ylim(*a.y2range)
        ax.set_xlim(*a.xrange)

    if spec.colorbar is not None:
        cb = spec.colorbar
        mappable = ScalarMappable(norm=Normalize(vmin=cb.vmin, vmax=cb.vmax), cmap=cmap_of(cb.colormap))
        bar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.03, format=cb.tick_format)
        bar.outline.set_linewidth(0.5)
        bar.ax.tick_params(length=0, labelsize=spec.font_size - 1)
        if cb.ticks is not None:
            bar.set_ticks(list(cb.ticks))
            if cb.tick_labels is not None:
                bar.set_ticklabels(list(cb.tick_labels))
        if cb.label:
            bar.set_label(cb.label, fontsize=spec.font_size)

    handles = [Patch(facecolor=e.color or "white", label=e.label, ec="k", lw=0.5) for e in spec.legend]
    handles += [
        Patch(
            facecolor=to_rgba(layer.color, layer.opacity), edgecolor=layer.color,
            hatch=layer.hatch or None, label=layer.label, lw=0.5,
        )
        for layer in spec.layers
        if layer.legend and layer.visible
    ]
    if handles:
        ax.legend(
            handles=handles, frameon=False, ncol=len(handles) if spec.legend else min(len(handles), 4),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0), fontsize=spec.font_size,
        )
    elif any(line.label is not None for line in spec.lines):
        # the labeled curves, in the order drawn, both axes' in one legend on top
        handles, labels = ax.get_legend_handles_labels()
        if twin is not None:
            more = twin.get_legend_handles_labels()
            handles, labels = handles + more[0], labels + more[1]
        (twin or ax).legend(handles, labels, loc="best", frameon=False, fontsize=spec.font_size - 1)
    if spec.title:
        ax.set_title(spec.title, fontsize=spec.font_size + 1)
    return fig


def _draw_line(ax: Any, line: Any) -> None:
    label = line.label if line.label is not None else "_nolegend_"
    if line.marker:
        ax.plot(
            line.x, line.y, linestyle="none", marker=_MARKERS.get(line.marker, "o"),
            color=line.color, markersize=line.marker_size, gid=line.kind, label=label,
        )
    else:
        ax.plot(
            line.x, line.y, color=line.color, linewidth=line.width,
            linestyle=_DASHES.get(line.dash, "-"), alpha=line.alpha, gid=line.kind,
            solid_capstyle="butt", label=label,
        )


def _panels(spec: FigureSpec) -> Any:
    """A figure of several panels: one axes per panel on a grid, laid out to fit."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4), dpi=spec.dpi, layout="constrained")
    FigureCanvasAgg(fig)
    rows, cols = spec.panel_grid
    grid = fig.subplots(rows, cols, squeeze=False)
    for panel, ax in zip(spec.panels, grid.ravel()):
        to_figure(panel, fig, ax)
    if spec.title:
        fig.suptitle(spec.title, fontsize=spec.font_size + 1)
    return fig


def render(spec: FigureSpec, file_type: str = "png", dpi: Optional[int] = None) -> bytes:
    """`spec` as the bytes of a PNG, PDF or SVG file, cropped to the drawing."""
    fig = to_figure(spec)
    buf = io.BytesIO()
    metadata = {"Software": "EMode, https://emodephotonix.com"} if file_type == "png" else None
    if file_type == "pdf":
        metadata = {"Creator": "EMode, https://emodephotonix.com"}
    fig.savefig(
        buf, format=file_type, dpi=dpi or spec.dpi, bbox_inches="tight", pad_inches=0.1, metadata=metadata
    )
    return buf.getvalue()


def save(spec: FigureSpec, path: Path, file_type: str) -> Path:
    """Write `spec` to `path` as `file_type` at the spec's dpi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render(spec, file_type))
    return path
