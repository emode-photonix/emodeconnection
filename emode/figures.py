"""``Figure``: a figure of one or several panels, drawn by EMode's plotting.

A panel holds a sweep's plot (or an optimizer's history, or a pulse), the
script's own curves (a fit, a theory, a quantity it computed), reference lines
and shaded bands. The figure is drawn in the session's browser tab, or saved
as a static image, as every other EMode plot is::

    fig = emode.Figure(rows=2)
    top = fig.panel(0)
    top.sweep(result, component='dB')                  # the sweep's traces and labels
    top.line(wavelengths, theory, style='--', label='theory')
    top.set(ylim=(-40, 0))
    fig.panel(1).line(x, y, style='o-', label='measured')
    fig.plot(file_name='spectrum')                     # EM_figures/spectrum.png

A panel's ``twin()`` is a second y axis on its right, for a curve in other units
on the same x axis::

    p = fig.panel(0)
    p.line(z, pump_mW, label='pump')
    p.set(xlabel='Position (mm)', ylabel='Pump (mW)')
    right = p.twin()
    right.line(z, signal_uW, style='--', label='signal')
    right.set(ylabel='Signal (uW)')

A panel's axes can be logarithmic with ``set(xscale='log', yscale='log')``,
and ``secondary_x()`` draws a second x axis on top whose ticks are a function of
x, the same wavelengths as frequencies for example::

    p.set(xlabel='Wavelength (nm)', xscale='log')
    p.secondary_x(lambda nm: 299792.458 / nm, label='Frequency (THz)')

A panel's ``style`` is a short string as in matplotlib: ``'-'`` (solid),
``'--'`` (dashed), ``':'`` (dotted), a marker (``'o'``, ``'s'``, ``'^'``,
``'D'``, ``'>'``, ``'<'``), or both (``'o-'``). ``color`` is an index into
EMode's line palette or a color (``'#3634A3'``, ``'gray'``).
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .types import ArgumentError

Range = Optional[Tuple[float, float]]

#: The scales an axis can have.
SCALES = ("linear", "log")


def _check_scale(function: str, name: str, value: Optional[str]) -> None:
    if value is not None and value not in SCALES:
        raise ArgumentError(f"{name} must be 'linear' or 'log', not {value!r}", function, name)


class TwinAxis:
    """A panel's second y axis, on its right: curves in other units against the same x."""

    def __init__(self) -> None:
        self.items: List[Tuple[str, dict]] = []
        self.settings: dict = {}

    def line(
        self,
        x: Sequence[float],
        y: Sequence[float],
        style: str = "-",
        label: Optional[str] = None,
        color: Union[int, str, None] = None,
        width: Optional[float] = None,
    ) -> "TwinAxis":
        """A curve through (`x`, `y`), read against the right axis."""
        self.items.append(("line", dict(x=x, y=y, style=style, label=label, color=color, width=width)))
        return self

    def hline(self, y: float, style: str = "--", label: Optional[str] = None, color: Union[int, str, None] = None) -> "TwinAxis":
        """A horizontal line at `y` on the right axis, across the panel."""
        self.items.append(("hline", dict(at=y, style=style, label=label, color=color)))
        return self

    def set(self, ylabel: Optional[str] = None, ylim: Range = None, yscale: Optional[str] = None) -> "TwinAxis":
        """The right axis's label, range and scale (``'linear'`` or ``'log'``). The x axis is the panel's own."""
        _check_scale("TwinAxis.set", "yscale", yscale)
        for name, value in dict(ylabel=ylabel, ylim=ylim, yscale=yscale).items():
            if value is not None:
                self.settings[name] = value
        return self


class Panel:
    """One panel of a ``Figure``: what is drawn on it, and its labels and ranges."""

    def __init__(self) -> None:
        self.source: Any = None
        self.selection: dict = {}
        self.items: List[Tuple[str, dict]] = []
        self.settings: dict = {}
        self.right: Optional[TwinAxis] = None
        self.secondary: Optional[dict] = None

    def twin(self) -> TwinAxis:
        """The panel's second y axis, on its right, made on the first call."""
        if self.right is None:
            self.right = TwinAxis()
        return self.right

    def secondary_x(
        self,
        function: Callable[[Any], Any],
        label: Optional[str] = None,
        inverse: Optional[Callable[[Any], Any]] = None,
        ticks: Optional[Sequence[float]] = None,
    ) -> "Panel":
        """A second x axis on top whose values are ``function(x)``, for the same data in other units.

        ``function`` takes and returns arrays, for example wavelength in nm to
        frequency in THz, ``lambda x: 299792.458 / x``. Its ticks are round
        values of the second axis, placed where `inverse` (found from
        ``function`` when it is monotonic and none is given) maps them back
        to x. ``ticks`` fixes them.
        """
        if not callable(function) or (inverse is not None and not callable(inverse)):
            raise ArgumentError("function and inverse must be callable", "Panel.secondary_x", "function")
        self.secondary = dict(function=function, label=label, inverse=inverse, ticks=ticks)
        return self

    def sweep(self, result: Any, **selection: Any) -> "Panel":
        """Draw `result` here as its own plot draws it: a sweep's result, an ``OptimizationResult``
        or a ``PulseResponse``, with the selection ``plot()`` takes (``result``, ``component``,
        ``panel``, ``input``, ``output``, ``input_mode``, ``output_mode``, ``mode``, ``record``)."""
        from .ui.sweep_figure import sweep_panel

        sweep_panel(result, **selection)  # refuses a selection the result does not offer, now
        self.source, self.selection = result, dict(selection)
        return self

    def line(
        self,
        x: Sequence[float],
        y: Sequence[float],
        style: str = "-",
        label: Optional[str] = None,
        color: Union[int, str, None] = None,
        width: Optional[float] = None,
    ) -> "Panel":
        """A curve of the script's own through (`x`, `y`)."""
        self.items.append(("line", dict(x=x, y=y, style=style, label=label, color=color, width=width)))
        return self

    def vline(self, x: float, style: str = "--", label: Optional[str] = None, color: Union[int, str, None] = None) -> "Panel":
        """A vertical line at `x` across the panel."""
        self.items.append(("vline", dict(at=x, style=style, label=label, color=color)))
        return self

    def hline(self, y: float, style: str = "--", label: Optional[str] = None, color: Union[int, str, None] = None) -> "Panel":
        """A horizontal line at `y` across the panel."""
        self.items.append(("hline", dict(at=y, style=style, label=label, color=color)))
        return self

    def vspan(self, x0: float, x1: float, label: Optional[str] = None, color: Union[int, str, None] = None) -> "Panel":
        """A shaded band from `x0` to `x1` across the panel."""
        self.items.append(("vspan", dict(x0=x0, x1=x1, label=label, color=color)))
        return self

    def set(
        self,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        xlim: Range = None,
        ylim: Range = None,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
    ) -> "Panel":
        """The panel's axis labels, title, ranges and scales (``'linear'`` or ``'log'``).

        What is not given keeps its default. A log axis needs positive data and ranges.
        """
        _check_scale("Panel.set", "xscale", xscale)
        _check_scale("Panel.set", "yscale", yscale)
        for name, value in dict(
            xlabel=xlabel, ylabel=ylabel, title=title, xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale
        ).items():
            if value is not None:
                self.settings[name] = value
        return self


class Figure:
    """A figure of `rows` by `cols` panels, drawn in the browser tab or saved as an image.

    ``panel(i)`` is the i-th panel in row-major order, or ``panel((row, col))``.
    ``figure_size`` is ``(width, height)`` in mm, by default 80 mm wide and
    56.6 mm high per panel. ``plot()`` shows or saves it as ``SweepResult.plot``
    does, and ``em.plot(figure)`` is the same.
    """

    def __init__(
        self,
        rows: int = 1,
        cols: int = 1,
        title: Optional[str] = None,
        figure_size: Range = None,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ArgumentError("a figure has at least one row and one column", "Figure", "rows")
        self.rows, self.cols = int(rows), int(cols)
        self.title = title
        self.figure_size = figure_size
        self.panels = [Panel() for _ in range(self.rows * self.cols)]

    def panel(self, index: Union[int, Tuple[int, int]] = 0) -> Panel:
        if isinstance(index, tuple):
            row, col = index
            index = row * self.cols + col if 0 <= col < self.cols else -1
        if not 0 <= index < len(self.panels):
            raise ArgumentError(
                f"this figure has {len(self.panels)} panels ({self.rows} x {self.cols}), no panel {index}",
                "Figure.panel",
                "index",
            )
        return self.panels[index]

    def plot(self, em: Any = None, **arguments: Any) -> Any:
        """Show the figure in an EMode session's browser tab, or save it with ``file_name``."""
        from .ui.session import show_data

        return show_data(self, em=em, **arguments)

    show = plot
