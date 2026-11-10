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

from ..plotting import PlotView, Schematic
from .figure import PlotOptions


@dataclass
class View:
    """One plot of the session. Mutable: the tab changes its controls and data."""

    id: str
    kind: str
    title: str
    data: Union[PlotView, Schematic]
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


def title_of(data: Union[PlotView, Schematic], arguments: Dict[str, Any]) -> str:
    """A view's name in the plots list: its plane and what it shows."""
    if isinstance(data, Schematic):
        return "schematic"
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
        data: Union[PlotView, Schematic],
        arguments: Dict[str, Any],
        options: PlotOptions,
        pixels: tuple,
    ) -> View:
        with self._lock:
            self._count += 1
            prefix = "s" if isinstance(data, Schematic) else "p"
            view = View(
                id=f"{prefix}{self._count}",
                kind="schematic" if isinstance(data, Schematic) else "plot",
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
