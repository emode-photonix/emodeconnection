"""The views of one session: what each ``plot()`` fetched, and what the tab has seen.

Each ``plot()`` fetches its panel from EMode at the call and registers it here
under an id (``p1``, ``p2``, ...) with the call's arguments as its initial
controls. The tab loads a view from here, so it needs no solver round trip,
and acknowledges it (``fetched``), which is what ``close()`` waits for. A
control change in the tab that needs new data replaces the view's data.

Every view is listed in the sidebar as a saved view (emode#1703): an ``Entry``
holding its recipe (a ``SavedView``) and, for a script's call, the view whose
data that call fetched. A recipe the simulation file holds is listed with no
data, and opens on the workbench.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

from ..plotting import MaterialView, MenuState, SavedView, Schematic, SMatrixView
from .figure import DATA_ARGUMENTS, FarFieldPanel, MonitorPanel, PlotOptions
from .gds import LOCAL_ARGUMENTS as GDS_ARGUMENTS
from .gds import GDSPanel
from .gds import apply as apply_gds
from .gds import title as gds_title
from .sweep_figure import SWEEP_CONTROLS, SweepPanel, select_sweep, sweep_title


def _select(data: Any, controls: Dict[str, Any]) -> Any:
    """`data` with the client-side selection of `controls` applied, without asking EMode."""
    if isinstance(data, GDSPanel):
        return apply_gds(data, controls)
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
    if isinstance(data, MonitorPanel):
        return data.with_selection(controls.get("component"), controls.get("plot_function"))
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
    "monitor": Kind("monitor", "n", (), ("component", "plot_function")),
    "gds": Kind("gds", "g", DATA_ARGUMENTS, GDS_ARGUMENTS),
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
    if isinstance(data, MonitorPanel):
        return "monitor"
    if isinstance(data, GDSPanel):
        return "gds"
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
    if isinstance(data, MonitorPanel):
        return MenuState(
            data=list(data.DATA), components=list(data.PARTS), selected={"data": data.data, "component": data.part}
        )
    from .modes import ModesPanel, offers_all_modes, with_all_modes

    if isinstance(data, ModesPanel):
        return data.menu
    if offers_all_modes(data):
        return with_all_modes(data.menu)
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
    #: The session's generation (``EMode.generation``) when `data` was fetched (#182).
    generation: int = 0

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
    if isinstance(data, MonitorPanel):
        return f"monitor {data.field.name} {data.data} {data.part}"
    if isinstance(data, GDSPanel):
        return gds_title(data)
    if isinstance(data, SweepPanel):
        return sweep_title(data)
    from ..figures import Figure

    if isinstance(data, Figure):
        return data.title or f"figure of {len(data.panels)} panels"
    from .modes import ModesPanel

    selected = data.menu.selected
    what = selected.get("data") or arguments.get("component") or ""
    how = selected.get("component") or ""
    grid = "all modes" if isinstance(data, ModesPanel) else ""
    return " ".join(part for part in (data.plane, what, how, grid) if part)


#: The workbench's id: the tab's default plot, pinned first in the sidebar (emode#1701).
BENCH = "bench"

#: How many of the workbench's datasets keep their data, the most recently shown.
CACHE_SIZE = 8


@dataclass
class Workbench:
    """The tab's default plot (emode#1701): the dataset of the inventory it shows, and what it held of the others.

    Each dataset the workbench has shown keeps its ``View`` here, its recipe
    (``arguments``, ``options``) and, for the `CACHE_SIZE` shown last, its
    data, so switching back to one asks EMode for nothing. Mutable: the tab's
    controls change it.
    """

    category: str = "field"
    dataset: Optional[str] = None
    held: "OrderedDict[str, View]" = field(default_factory=OrderedDict)
    #: The dataset each category showed last, which choosing the category shows again.
    last: Dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    #: The saved view (an ``Entry`` id) the workbench was loaded from or last saved as (emode#1703).
    origin: Optional[str] = None
    #: The workbench's recipe (``recipe_key``) when it was loaded or saved: "modified" when it differs.
    origin_key: Optional[str] = None

    @property
    def view(self) -> Optional[View]:
        return self.held.get(self.dataset) if self.dataset is not None else None

    def show(self, dataset: str, category: str, view: View) -> None:
        """Show `view` as `dataset`, holding the data of the `CACHE_SIZE` datasets shown last."""
        self.dataset, self.category = dataset, category
        self.last[category] = dataset
        self.held[dataset] = view
        self.held.move_to_end(dataset)
        with_data = [k for k, v in self.held.items() if v.data is not None]
        for key in with_data[: max(0, len(with_data) - CACHE_SIZE)]:
            self.held[key].data = None

    def summary(self) -> Dict[str, Any]:
        return {"id": BENCH, "kind": "bench", "title": "Workbench", "created": self.created, "fetched": True}

    def key(self) -> Optional[str]:
        """What the workbench shows, as a ``recipe_key``: None when it shows nothing."""
        view = self.view
        if view is None:
            return None
        return recipe_key(view.kind, view.arguments, changed_options(view.options), self.dataset)

    def recipe(self, name: str, tags: Optional[List[str]] = None) -> Optional[SavedView]:
        """What the workbench shows, as a saved view named `name` with `tags`."""
        view = self.view
        if view is None:
            return None
        return SavedView(
            name=name,
            kind=view.kind,
            arguments=plain(view.arguments),
            options=changed_options(view.options),
            dataset=self.dataset,
            tags=list(tags or []),
            created=now_iso(),
        )

    def modified(self) -> bool:
        """Whether the workbench's settings differ from the saved view it came from."""
        return self.origin is not None and self.key() != self.origin_key


def plain(value: Any) -> Any:
    """`value` as JSON reads it back: tuples as lists, numpy scalars and arrays as numbers and lists.

    A recipe is compared with one read back from EMode, which went through JSON,
    so both are put in that form first.
    """

    def default(v: Any) -> Any:
        import numpy as np

        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            return v.item()
        if isinstance(v, (set, frozenset)):
            return sorted(v)
        if isinstance(v, complex):
            return [v.real, v.imag]
        return str(v)

    return json.loads(json.dumps(value, default=default))


def changed_options(options: PlotOptions) -> Dict[str, Any]:
    """The figure options that differ from ``PlotOptions``' defaults, as plain data."""
    defaults = PlotOptions()
    out = {f.name: getattr(options, f.name) for f in fields(PlotOptions)}
    return plain({k: v for k, v in out.items() if v != getattr(defaults, k)})


def options_of(recipe: Dict[str, Any]) -> PlotOptions:
    """A recipe's ``options`` as ``PlotOptions``, a name this client does not know left out."""
    known = {f.name for f in fields(PlotOptions)}
    values = {k: v for k, v in recipe.items() if k in known}
    if isinstance(values.get("figure_size"), list):
        values["figure_size"] = tuple(values["figure_size"])
    return PlotOptions(**values)


def recipe_key(kind: str, arguments: Dict[str, Any], options: Dict[str, Any], dataset: Optional[str]) -> str:
    """What makes two recipes the same plot: their kind, arguments, options and dataset."""
    return json.dumps([kind, plain(arguments), plain(options), dataset], sort_keys=True)


def key_of(view: SavedView) -> str:
    return recipe_key(view.kind, view.arguments, view.options, view.dataset)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def clean_tags(tags: Any) -> List[str]:
    """`tags` as a saved view keeps them: a list, or text separated by commas, trimmed, without blanks or repeats."""
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = tags.split(",")
    if not isinstance(tags, (list, tuple)):
        from ..types import ArgumentError

        raise ArgumentError("tags are a list of strings, or one string separated by commas", "save_view", "tags")
    return list(dict.fromkeys(t for t in (str(tag).strip() for tag in tags) if t))


@dataclass
class Entry:
    """One saved view of the sidebar: its recipe and, for a script's call, the view holding its data.

    ``session_only`` marks data this client holds and no recipe can fetch
    again (a far field, a figure, a sweep's result, an S-matrix the script
    built): listed for the session, never posted to EMode. Mutable: renaming
    and saving change it.
    """

    id: str
    saved: SavedView
    view: Optional[View] = None
    session_only: bool = False
    created: float = field(default_factory=time.time)

    def summary(self) -> Dict[str, Any]:
        view = self.view
        return {
            "id": self.id,
            "kind": view.kind if view is not None else self.saved.kind,
            "title": self.saved.name,
            "created": view.created if view is not None else self.created,
            "fetched": view.fetched if view is not None else True,
            "script": view is not None and not self.session_only,
            "session_only": self.session_only,
            "tags": list(self.saved.tags),
            # a script's view shows its own data, a recipe opens on the workbench
            "opens": "view" if view is not None else "bench",
        }


class ViewRegistry:
    """The session's views, newest last, safe to read from the tab's thread, and the workbench.

    Each view is listed as a saved view (``Entry``), in the order it was first
    listed: the simulation file's recipes, then the script's views and the
    workbench's saves.
    """

    def __init__(self) -> None:
        self._views: Dict[str, View] = {}
        self._entries: "OrderedDict[str, Entry]" = OrderedDict()
        self._lock = threading.RLock()
        self._count = 0
        self.bench = Workbench()
        #: Whether the simulation file's saved views have been listed (once a session).
        self.file_listed = False

    # ---------------------------------------------------------------- views with data

    def _new_view(
        self, data: Any, arguments: Dict[str, Any], options: PlotOptions, pixels: tuple, generation: int
    ) -> View:
        kind = kind_of(data)
        self._count += 1
        return View(
            id=f"{KINDS[kind].prefix}{self._count}",
            kind=kind,
            title=title_of(data, arguments),
            data=data,
            arguments=dict(arguments),
            options=options,
            pixels=tuple(pixels),
            generation=generation,
        )

    def add(
        self,
        data: Any,
        arguments: Dict[str, Any],
        options: PlotOptions,
        pixels: tuple,
        generation: int = 0,
        session_only: bool = False,
    ) -> View:
        """Register `data` as a new view, listed under its title made unique. Nothing is deduplicated."""
        with self._lock:
            view = self._new_view(data, arguments, options, pixels, generation)
            self._views[view.id] = view
            recipe = self._recipe(view, self.unique_name(view.title))
            self._entries[view.id] = Entry(view.id, recipe, view, session_only)
            return view

    def register(
        self,
        data: Any,
        arguments: Dict[str, Any],
        options: PlotOptions,
        pixels: tuple,
        generation: int = 0,
        name: Optional[str] = None,
        session_only: bool = False,
        dataset: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[View, Optional[SavedView]]:
        """Register a script call's `data` as a saved view: ``(view, the recipe to post or None)``.

        A view named `name` replaces the saved view of that name. Without a
        name, a recipe the sidebar already lists (from this run or from the
        file) is shown again rather than listed twice, and a new one is named
        by its title, numbered when another view has that title. Nothing is
        posted for a recipe EMode already holds, or for `session_only` data.
        `dataset` is the workbench data entry a replayed saved view names.

        `tags` (already cleaned, ``clean_tags``) are the saved view's tags. None
        keeps the tags of a saved view this call replaces or shows again, and a
        list, ``[]`` included, replaces them (emode#1704).
        """
        with self._lock:
            kind = kind_of(data)
            key = recipe_key(kind, arguments, changed_options(options), dataset)
            entry: Optional[Entry] = None
            if name is not None:
                entry = self.by_name(name)
            elif not session_only:
                entry = next(
                    (e for e in self._entries.values() if not e.session_only and key_of(e.saved) == key), None
                )
            if entry is None or entry.session_only != session_only:
                if entry is not None:
                    self._drop(entry)
                view = self._new_view(data, arguments, options, pixels, generation)
                self._views[view.id] = view
                recipe = self._recipe(
                    view, name if name is not None else self.unique_name(view.title), dataset, tags
                )
                self._entries[view.id] = Entry(view.id, recipe, view, session_only)
                return view, None if session_only else recipe
            same = key_of(entry.saved) == key
            view = entry.view
            if view is None:
                view = View(entry.id, kind, title_of(data, arguments), data, dict(arguments), options, tuple(pixels))
                self._views[entry.id] = view
                entry.view = view
            view.kind, view.data, view.arguments, view.options = kind, data, dict(arguments), options
            view.title, view.pixels, view.generation = title_of(data, arguments), tuple(pixels), generation
            view.fetched = False
            kept = list(entry.saved.tags) if tags is None else list(tags)
            if same and kept == list(entry.saved.tags):
                return view, None
            # a new recipe under the same name keeps the saved view's tags unless the call gives them
            entry.saved = self._recipe(view, entry.saved.name, dataset, kept)
            return view, None if session_only else entry.saved

    @staticmethod
    def _recipe(
        view: View, name: str, dataset: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> SavedView:
        return SavedView(
            name=name,
            kind=view.kind,
            arguments=plain(view.arguments),
            options=changed_options(view.options),
            dataset=dataset,
            tags=list(tags or []),
            created=now_iso(),
        )

    def get(self, view_id: str) -> Optional[View]:
        """The view `view_id`: a plot's, or what the workbench shows for ``'bench'``."""
        if view_id == BENCH:
            return self.bench.view
        with self._lock:
            return self._views.get(view_id)

    # ---------------------------------------------------------------- the saved views

    def entries(self) -> List[Entry]:
        with self._lock:
            return list(self._entries.values())

    def entry(self, entry_id: str) -> Optional[Entry]:
        with self._lock:
            return self._entries.get(entry_id)

    def by_name(self, name: str) -> Optional[Entry]:
        with self._lock:
            return next((e for e in self._entries.values() if e.saved.name == name), None)

    def unique_name(self, title: str) -> str:
        """`title`, or ``title (2)``, ``(3)``... when another saved view has it."""
        with self._lock:
            taken = {e.saved.name for e in self._entries.values()}
        if title not in taken:
            return title
        n = 2
        while f"{title} ({n})" in taken:
            n += 1
        return f"{title} ({n})"

    def list_file(self, views: List[SavedView]) -> None:
        """List the simulation file's saved views, in its order, each opening on the workbench."""
        with self._lock:
            self.file_listed = True
            for saved in views:
                if self.by_name(saved.name) is not None:
                    continue
                self._count += 1
                entry_id = f"v{self._count}"
                self._entries[entry_id] = Entry(entry_id, saved)

    def save(self, saved: SavedView, entry_id: Optional[str] = None, view: Optional[View] = None) -> Entry:
        """Keep `saved` under `entry_id` (its recipe replaced), or as a new entry; `view` is the data it now shows."""
        with self._lock:
            entry = self._entries.get(entry_id) if entry_id is not None else None
            if entry is None:
                self._count += 1
                entry = Entry(f"v{self._count}", saved)
                self._entries[entry.id] = entry
            entry.saved = saved
            if view is not None and entry.view is not None:
                self._views[entry.id] = view
                entry.view = view
            return entry

    def rename(self, entry_id: str, name: str, tags: Optional[List[str]] = None) -> Entry:
        """Give the entry `entry_id` the name `name` and, unless None, the `tags`."""
        with self._lock:
            entry = self._entries[entry_id]
            update: Dict[str, Any] = {"name": name}
            if tags is not None:
                update["tags"] = list(tags)
            entry.saved = entry.saved.model_copy(update=update)
            return entry

    def remove(self, entry_id: str) -> Optional[Entry]:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is not None:
                self._drop(entry)
            if self.bench.origin == entry_id:
                self.bench.origin = None
            return entry

    def _drop(self, entry: Entry) -> None:
        self._entries.pop(entry.id, None)
        view = self._views.pop(entry.id, None)
        if view is not None:
            # close() waits for the tab to load every view: none can open this one now
            view.fetched = True

    def summaries(self) -> List[Dict[str, Any]]:
        """The sidebar's entries: the workbench first, then each saved view, oldest first."""
        return [self.bench.summary(), *(e.summary() for e in self.entries())]

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
