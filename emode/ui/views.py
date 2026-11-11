"""The views of one session: what each ``plot()`` fetched, and what the tab has seen.

Each ``plot()`` fetches its panel from EMode at the call and registers it here
under an id (``p1``, ``p2``, ...) with the call's arguments as its initial
controls. The tab loads a view from here, so it needs no solver round trip,
and acknowledges it (``fetched``), which is what ``close()`` waits for. A
control change in the tab that needs new data replaces the view's data.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..plotting import MaterialView, MenuState, PlotView, Schematic, SMatrixView
from .figure import DATA_ARGUMENTS, FarFieldPanel, PlotOptions
from .sweep_figure import SWEEP_CONTROLS, SweepPanel, select_sweep, sweep_title


def _select(data: Any, controls: Dict[str, Any]) -> Any:
    """`data` with the client-side selection of `controls` applied, without asking EMode."""
    if isinstance(data, SMatrixView):
        selected = dict(data.menu.selected)
        for name, key in (("input", "input"), ("output", "output"), ("plot_function", "component")):
            value = controls.get(name)
            if value is None:
                continue
            if key != "component":
                value = _port_label(data, value)
            selected[key] = str(value)
        return data.model_copy(update={"menu": data.menu.model_copy(update={"selected": selected})})
    if isinstance(data, MaterialView) and controls.get("component") is not None:
        selected = {**data.menu.selected, "component": str(controls["component"])}
        return data.model_copy(update={"menu": data.menu.model_copy(update={"selected": selected})})
    if isinstance(data, SweepPanel):
        return select_sweep(data, controls)
    if isinstance(data, FarFieldPanel) and controls.get("data") is not None:
        panel = str(controls["data"])
        return FarFieldPanel(data.farfield, panel if panel in data.panels else "intensity")
    return data


def _port_label(view: SMatrixView, value: Any) -> str:
    """A port given as an index, a label or a port's name, as its menu label."""
    labels = list(view.port_labels)
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        if 0 <= index < len(labels):
            return labels[index]
    if value in labels:
        return str(value)
    names = list(view.smatrix.port_names or ())
    if value in names:
        return labels[names.index(value)]
    from ..types import ArgumentError

    raise ArgumentError(f"no port {value!r} in this S-matrix, which has {labels}", "plot_S_matrix", "input")


@dataclass(frozen=True)
class Kind:
    """A kind of view: its id prefix, the arguments EMode computes and those this client applies."""

    name: str
    prefix: str
    data_arguments: tuple
    local_arguments: tuple = ()


KINDS: Dict[str, Kind] = {
    "plot": Kind("plot", "p", DATA_ARGUMENTS),
    "schematic": Kind("schematic", "s", DATA_ARGUMENTS),
    "smatrix": Kind("smatrix", "x", (), ("input", "output", "plot_function")),
    "material": Kind("material", "m", ("material", "wavelengths", "step", "x"), ("component",)),
    "farfield": Kind("farfield", "f", (), ("data",)),
    "sweep": Kind("sweep", "w", (), SWEEP_CONTROLS),
    "figure": Kind("figure", "c", ()),
}


def kind_of(data: Any) -> str:
    if isinstance(data, Schematic):
        return "schematic"
    if isinstance(data, SMatrixView):
        return "smatrix"
    if isinstance(data, MaterialView):
        return "material"
    if isinstance(data, FarFieldPanel):
        return "farfield"
    if isinstance(data, SweepPanel):
        return "sweep"
    from ..figures import Figure

    if isinstance(data, Figure):
        return "figure"
    return "plot"


def menu_of(data: Any) -> MenuState:
    """The controls a view offers, as ``MenuState``."""
    if isinstance(data, FarFieldPanel):
        return MenuState(data=data.panels, selected={"data": data.panel})
    if isinstance(data, SweepPanel):
        # the menus themselves go in the reply's extra (sweep_menus): their names vary by result
        selected = {k: ", ".join(v) if isinstance(v, list) else str(v) for k, v in data.selected.items()}
        return MenuState(selected=selected)
    return getattr(data, "menu", None) or MenuState()


@dataclass
class View:
    """One plot of the session. Mutable: the tab changes its controls and data."""

    id: str
    kind: str
    title: str
    data: Any
    arguments: Dict[str, Any]
    options: PlotOptions
    pixels: tuple
    created: float = field(default_factory=time.time)
    fetched: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "created": self.created,
            "fetched": self.fetched,
        }


def title_of(data: Any, arguments: Dict[str, Any]) -> str:
    """A view's name in the plots list: its plane and what it shows."""
    if isinstance(data, Schematic):
        return "schematic"
    if isinstance(data, SMatrixView):
        s = data.menu.selected
        return f"S-matrix {s.get('input', '')} to {s.get('output', '')} {s.get('component', '')}".strip()
    if isinstance(data, MaterialView):
        return f"material {data.material} {data.menu.selected.get('component', '')}".strip()
    if isinstance(data, FarFieldPanel):
        return f"far field {data.farfield.wavelength:g} nm {data.panel}"
    if isinstance(data, SweepPanel):
        return sweep_title(data)
    from ..figures import Figure

    if isinstance(data, Figure):
        return data.title or f"figure of {len(data.panels)} panels"
    selected = data.menu.selected
    what = selected.get("data") or arguments.get("component") or ""
    how = selected.get("component") or ""
    return " ".join(part for part in (data.plane, what, how) if part)


class ViewRegistry:
    """The session's views, newest last, safe to read from the tab's thread."""

    def __init__(self) -> None:
        self._views: Dict[str, View] = {}
        self._lock = threading.Lock()
        self._count = 0

    def add(
        self,
        data: Any,
        arguments: Dict[str, Any],
        options: PlotOptions,
        pixels: tuple,
    ) -> View:
        kind = kind_of(data)
        with self._lock:
            self._count += 1
            view = View(
                id=f"{KINDS[kind].prefix}{self._count}",
                kind=kind,
                title=title_of(data, arguments),
                data=data,
                arguments=dict(arguments),
                options=options,
                pixels=tuple(pixels),
            )
            self._views[view.id] = view
            return view

    def get(self, view_id: str) -> Optional[View]:
        with self._lock:
            return self._views.get(view_id)

    def list(self) -> List[View]:
        with self._lock:
            return list(self._views.values())

    def mark_fetched(self, view_id: str) -> bool:
        with self._lock:
            view = self._views.get(view_id)
            if view is None:
                return False
            view.fetched = True
            return True

    def all_fetched(self) -> bool:
        with self._lock:
            return all(v.fetched for v in self._views.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._views)
