"""The session's local web server: the app tab's files and its API, on loopback only.

One ``UIServer`` per ``EMode`` session, started at the first ``plot()`` and
stopped by ``close()``. It is the standard library's ``ThreadingHTTPServer``
bound to ``127.0.0.1`` on an ephemeral port (``EMODE_UI_PORT`` names a fixed
one, for a port forwarded over SSH), serving on a daemon thread.

What keeps another program or page out:

- A 256-bit token per session, in the tab's URL (``?token=``) when the app
  loads and in the ``X-EMode-Token`` header of every API request, the event
  stream included. It is held in memory only, written to no file, no log and no
  response, and rotated by ``stop()``. A request without it is answered 403
  with an empty body. The app's static files (the page's script, its style
  sheet and the plotting library) are this package's own files and carry no
  session data, so they are served without it: a ``<script>`` tag cannot send
  a header, and putting the token in the page would put it in a response.
- The ``Host`` header must be this server's loopback address (or the host of
  ``EMODE_UI_URL``), which defeats DNS rebinding.
- A request whose ``Origin`` is another site's is refused before any handler
  runs, a preflight (``OPTIONS``) is always refused, and no
  ``Access-Control-Allow-Origin`` is ever sent, so no other page can read a
  reply.
- The page is served with a ``Content-Security-Policy`` allowing only this
  server's own files, API replies are never cached, and no path outside the
  package's ``ui/pages`` and ``ui/static`` directories is ever served.

The tab is told what happens over server-sent events (``/api/events``):
``view-added`` when ``plot()`` registers a view, ``session-closed`` when the
session ends. It posts ``/api/ping`` every two seconds, which is how the
client knows the tab is alive.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import secrets
import threading
import time
from dataclasses import fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .. import colormaps
from ..types import EModeError
from .figure import PLOT_RATIO, PlotOptions, compose, effective_contours, figure_aspect
from .gds import LOCAL_ARGUMENTS as GDS_ARGUMENTS
from .gds import GDSPanel, layer_rows, state_arguments
from .renderers.plotly import to_plotly
from ..plotting import PlotView, Schematic
from .cut import CutPanel, cut_arguments, held_xy, offers_cut
from .cut import frozen as frozen_cut
from .modes import ALL_MODES, ModesPanel, columns_for, frozen_arguments, mode_labels, offers_all_modes
from .views import BENCH, KINDS, View, ViewRegistry, kind_of, menu_of, options_of, plain, title_of

UI_DIR = Path(__file__).parent
PAGES = UI_DIR / "pages"
STATIC = UI_DIR / "static"

#: Seconds without a ping after which the tab counts as gone.
HEARTBEAT_TIMEOUT = 5.0

#: The page's content security policy: only this server's own files.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors *"
)

#: How far the tab's panel may differ from the size a view was fetched at, as a
#: fraction, before a z panel is fetched again at the tab's size.
REFETCH_FRACTION = 0.25

FIGURE_OPTIONS = tuple(f.name for f in fields(PlotOptions))


def _ui_url() -> Optional[str]:
    url = os.environ.get("EMODE_UI_URL")
    return url.rstrip("/") if url else None


class UIServer:
    """The local server of one session. ``fetch`` asks EMode for a view's data again."""

    def __init__(
        self,
        registry: ViewRegistry,
        fetch: Callable[[View, Dict[str, Any], Tuple[int, int]], Any],
        port: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
        session_info: Optional[Callable[[], Dict[str, Any]]] = None,
        details: Optional[Callable[[str], Dict[str, Any]]] = None,
        login: Optional[Callable[..., Any]] = None,
        emode_version: Optional[Callable[[], Optional[str]]] = None,
        generation: Optional[Callable[[], int]] = None,
        inventory: Optional[Callable[[], Any]] = None,
        fetch_dataset: Optional[Callable[[str, Dict[str, Any], Tuple[int, int]], Any]] = None,
        post_saved: Optional[Callable[[Any], None]] = None,
        delete_saved: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.registry = registry
        #: ``post_saved(view)``: keep a ``SavedView`` in the simulation file (emode#1703)
        self.post_saved = post_saved
        #: ``delete_saved(name)``: remove a saved view from the simulation file
        self.delete_saved = delete_saved
        #: ``inventory()``: EMode's ``PlotInventory``, what the workbench lists (emode#1701)
        self.inventory = inventory
        #: ``fetch_dataset(kind, arguments, pixels)``: one of the inventory's datasets, for the workbench
        self.fetch_dataset = fetch_dataset
        self._inventory_cache: Optional[Tuple[float, int, Any]] = None
        self._bench_lock = threading.RLock()
        #: The session's generation now (``EMode.generation``): a view fetched before it is stale (#182)
        self.generation = generation
        #: ``login(function, **kwargs)``: one of EMode's login calls, for the account page
        self.login = login
        self.emode_version = emode_version
        self.fetch = fetch
        self.session_info = session_info
        self.details = details
        self.clock = clock
        self.token = secrets.token_urlsafe(32)
        self._requested_port = (
            port if port is not None else int(os.environ.get("EMODE_UI_PORT") or 0)
        )
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._subscribers: List["queue.Queue[Optional[Tuple[str, Any]]]"] = []
        self._lock = threading.Lock()
        self.last_ping: Optional[float] = None
        self.opened_at: Optional[float] = None
        self.tab_closed = False
        self.session_ended = False
        self._login_state: Optional[str] = None

    # ---------------------------------------------------------------- lifecycle

    @property
    def running(self) -> bool:
        return self._httpd is not None

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise EModeError("the plot server is not running")
        return int(self._httpd.server_address[1])

    def start(self) -> "UIServer":
        if self._httpd is not None:
            return self
        httpd = ThreadingHTTPServer(("127.0.0.1", self._requested_port), _Handler)
        httpd.daemon_threads = True
        httpd.ui = self  # type: ignore[attr-defined]
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="emode-ui",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Tell the tab the session ended, stop serving, free the port and rotate the token."""
        httpd = self._httpd
        if httpd is None:
            return
        self.publish("session-closed", {})
        with self._lock:
            for q in self._subscribers:
                q.put(None)
            self._subscribers.clear()
        httpd.shutdown()
        httpd.server_close()
        self._httpd = None
        self.token = secrets.token_urlsafe(32)

    # ---------------------------------------------------------------- addresses

    def base_url(self) -> str:
        return _ui_url() or f"http://127.0.0.1:{self.port}"

    def url(
        self, view_id: Optional[str] = None, embed: bool = False, page: Optional[str] = None
    ) -> str:
        """The app's address with this session's token, opened on `view_id`, or on `page` (``'account'``)."""
        query = f"?token={self.token}" + ("&embed=1" if embed else "")
        if page is not None:
            fragment = f"#/{page}"
        else:
            fragment = f"#/plots/{view_id}" if view_id else "#/plots"
        return f"{self.base_url()}/{query}{fragment}"

    def allowed_hosts(self) -> set:
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        url = _ui_url()
        if url:
            hosts.add(urlsplit(url).netloc)
        return hosts

    def allowed_origins(self) -> set:
        origins = {f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"}
        url = _ui_url()
        if url:
            parts = urlsplit(url)
            origins.add(f"{parts.scheme}://{parts.netloc}")
        return origins

    # ---------------------------------------------------------------- the tab

    def tab_alive(self) -> bool:
        """Whether the tab has pinged within the last `HEARTBEAT_TIMEOUT` seconds."""
        if self.tab_closed or self.last_ping is None:
            return False
        return self.clock() - self.last_ping < HEARTBEAT_TIMEOUT

    def ping(self) -> None:
        self.last_ping = self.clock()
        self.tab_closed = False

    def publish(self, event: str, data: Any) -> None:
        """Send `event` to every connected tab."""
        with self._lock:
            for q in self._subscribers:
                q.put((event, data))

    def subscribe(self) -> "queue.Queue[Optional[Tuple[str, Any]]]":
        q: "queue.Queue[Optional[Tuple[str, Any]]]" = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Any) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    # ---------------------------------------------------------------- the account page

    def account(self, method: str, action: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Answer the account page's `action` through EMode's login calls, None for no such route.

        A change of the sign-in state is told to the tab as ``login-changed``.
        """
        from .account import device_signin_offered, route

        if self.login is None:
            return None
        reply = route(self.login, method, action, body)
        if reply is not None and action == "status":
            reply.setdefault("device", device_signin_offered())
            if self.emode_version is not None and "version" not in reply:
                try:
                    reply["version"] = self.emode_version()
                except EModeError:
                    pass
        state = (reply or {}).get("state")
        if state is not None and state != self._login_state:
            self._login_state = state
            self.publish("login-changed", {"state": state})
        return reply

    # ---------------------------------------------------------------- the session page

    def session_page(self) -> Dict[str, Any]:
        """The session page's rows: EMode's ``SessionInfo`` (while the session runs) and this client's."""
        from .session import client_rows

        emode: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        if not self.session_ended and self.session_info is not None:
            try:
                emode = self.session_info()
            except EModeError as exc:
                error = str(exc)
        from .session import DOCS_URL

        client = client_rows(self)
        # the documentation of the EMode this session runs, once it has said which
        version = (emode or {}).get("version")
        client["docs"] = f"{DOCS_URL}?version={version}" if version else DOCS_URL
        return {
            "emode": emode,
            "error": error,
            "session_ended": self.session_ended,
            "client": client,
        }


    # ---------------------------------------------------------------- the workbench (emode#1701)

    def current_inventory(self) -> Any:
        """EMode's ``PlotInventory``, asked again after `INVENTORY_SECONDS` or a change of the session."""
        if self.inventory is None:
            raise EModeError("this session cannot ask EMode what it can plot")
        now = self.generation() if self.generation is not None else 0
        cached = self._inventory_cache
        if cached is not None and (self.session_ended or (
            cached[1] == now and self.clock() - cached[0] < INVENTORY_SECONDS
        )):
            return cached[2]
        if self.session_ended:
            raise EModeError("This EMode session has ended, so it cannot be asked what it can plot.")
        value = self.inventory()
        self._inventory_cache = (self.clock(), now, value)
        return value

    def inventory_reply(self) -> Dict[str, Any]:
        return self.current_inventory().model_dump(mode="json")

    def bench_reply(
        self,
        controls: Dict[str, Any],
        pixels: Optional[Tuple[int, int]],
        width: Optional[float] = None,
    ) -> Dict[str, Any]:
        """The workbench's figure and menus after the tab's `controls`.

        ``category`` (``'field'`` or ``'trend'``) shows the data that category
        showed last, or its first available. ``dataset`` (a ``DatasetEntry``'s
        id) shows that data, fetched with its arguments unless the workbench
        holds it. An unavailable one is refused with the inventory's reason and
        nothing is asked of EMode. The other controls are the shown data's own
        and go through ``view_reply`` as a plot's do, except when the data
        changes: then they were the old data's, and are dropped.
        """
        with self._bench_lock:
            bench = self.registry.bench
            controls = dict(controls)
            saved = controls.pop("saved", None)
            category = controls.pop("category", None)
            dataset = controls.pop("dataset", None)
            if saved is not None:
                # a saved view, or a script's view (Edit in workbench), loaded into the workbench (emode#1703)
                self.bench_open_saved(str(saved), pixels)
                controls, category, dataset = {}, None, bench.dataset
            inventory = self.current_inventory()
            entries = {d.id: d for d in inventory.datasets}
            if category not in (None, "field", "trend"):
                return self._bench_error(f"the workbench shows field or trend data, not {category!r}", inventory)
            if dataset is None and category is not None and category != bench.category:
                remembered = bench.last.get(category)
                dataset = remembered if remembered in entries else _first(inventory, category)
            if dataset is None:
                dataset = bench.dataset if bench.dataset in entries else _first(inventory, None)
            entry = entries.get(dataset) if dataset is not None else None
            if entry is None:
                held = f"no {dataset!r}" if dataset else "nothing it can plot yet"
                return self._bench_error(f"This session holds {held}.", inventory, category)
            if not entry.available:
                return self._bench_error(entry.reason or "not available", inventory, category or entry.category)
            if dataset != bench.dataset or bench.view is None:
                self._bench_show(entry, pixels)
                controls = {}
            else:
                view = bench.view
                if self._bench_stale(entry, view):
                    bench.show(entry.id, entry.category, self._bench_load(entry, view, pixels))
            reply = self.view_reply(bench.view, controls, pixels, width)
            reply["bench"] = {**_bench_menus(inventory, bench.category, bench.dataset), **self._origin()}
            return reply

    # ---------------------------------------------------------------- saved views (emode#1703)

    def _origin(self) -> Dict[str, Any]:
        """The saved view the workbench came from, and whether its settings have changed since."""
        bench = self.registry.bench
        entry = self.registry.entry(bench.origin) if bench.origin is not None else None
        if entry is None:
            return {"origin": None, "modified": False}
        return {"origin": {"id": entry.id, "name": entry.saved.name}, "modified": bench.modified()}

    def _known_inventory(self) -> Optional[Any]:
        """The inventory, or the last one asked for once the session has ended (None if never)."""
        if self.session_ended:
            return self._inventory_cache[2] if self._inventory_cache is not None else None
        return self.current_inventory()

    def bench_open_saved(self, entry_id: str, pixels: Optional[Tuple[int, int]]) -> None:
        """Load the saved view `entry_id` into the workbench, which then counts it as its origin.

        A script's view brings its own data (Edit in workbench), so nothing is
        fetched. A recipe is fetched as the inventory's entry for it says, from
        the data the workbench still holds when that is the same.
        """
        with self._bench_lock:
            entry = self.registry.entry(entry_id)
            if entry is None:
                raise EModeError(f"no saved view {entry_id}")
            if entry.session_only:
                raise EModeError(
                    "This plot shows data the script holds, which EMode cannot fetch again, "
                    "so it cannot be edited on the workbench."
                )
            bench, saved, view = self.registry.bench, entry.saved, entry.view
            if view is None and self.session_ended:
                raise EModeError(
                    f"This EMode session has ended, so the saved view {saved.name!r} cannot be fetched. "
                    "The plots the script showed still open."
                )
            inventory = self._known_inventory()
            entries = {d.id: d for d in inventory.datasets} if inventory is not None else {}
            dataset = saved.dataset if saved.dataset in entries else None
            if dataset is None:
                source = view if view is not None else saved
                dataset = dataset_for(source.kind, source.arguments, entries)
            target = entries.get(dataset) if dataset is not None else None
            if target is None:
                raise EModeError(f"The workbench lists no data that the saved view {saved.name!r} shows.")
            if view is not None and view.data is not None:
                bench.show(target.id, target.category, replace(view, id=BENCH, arguments=dict(view.arguments), fetched=True))
            else:
                if not target.available:
                    raise EModeError(target.reason or f"{target.label} is not available")
                held = bench.held.get(target.id)
                if (
                    held is not None
                    and held.data is not None
                    and plain(held.arguments) == plain(saved.arguments)
                    and not self._bench_stale(target, held)
                ):
                    held.options = options_of(saved.options)
                    bench.show(target.id, target.category, held)
                else:
                    recipe = View(
                        id=BENCH,
                        kind=saved.kind if saved.kind in KINDS else target.kind,
                        title=saved.name,
                        data=None,
                        arguments=dict(saved.arguments),
                        options=options_of(saved.options),
                        pixels=tuple(pixels or (1200, 750)),
                    )
                    bench.show(target.id, target.category, self._bench_load(target, recipe, pixels))
            bench.origin = entry.id
            bench.origin_key = bench.key()

    def saved_action(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """The tab's Save, Save as, Rename and Remove, each kept in the simulation file before the sidebar changes.

        ``save`` writes the workbench back to the saved view it came from,
        ``save_as`` keeps it under a new ``name``, ``rename`` gives the entry
        ``id`` the ``name``, and ``remove`` drops it. A view of the script's own
        data is renamed and removed for this session only.
        """
        action = body.get("action")
        with self._bench_lock:
            bench = self.registry.bench
            if action == "save":
                entry = self.registry.entry(bench.origin) if bench.origin is not None else None
                if entry is None:
                    raise EModeError("This workbench did not come from a saved view: use Save as, with a name.")
                recipe = bench.recipe(entry.saved.name)
                if recipe is None or bench.view is None:
                    raise EModeError("The workbench shows nothing to save yet.")
                self._post(recipe)
                shown = None
                if entry.view is not None:
                    # the script's entry shows what was saved from now on
                    shown = replace(bench.view, id=entry.id, arguments=dict(bench.view.arguments), fetched=True)
                self.registry.save(recipe, entry.id, shown)
                bench.origin_key = bench.key()
            elif action == "save_as":
                name = self._new_name(body.get("name"))
                recipe = bench.recipe(name)
                if recipe is None:
                    raise EModeError("The workbench shows nothing to save yet.")
                self._post(recipe)
                entry = self.registry.save(recipe)
                bench.origin, bench.origin_key = entry.id, bench.key()
            elif action in ("rename", "remove"):
                entry = self.registry.entry(str(body.get("id")))
                if entry is None:
                    raise EModeError(f"no saved view {body.get('id')}")
                if action == "rename":
                    name = self._new_name(body.get("name"), entry.id)
                    if not entry.session_only:
                        self._delete(entry.saved.name)
                        self._post(entry.saved.model_copy(update={"name": name}))
                    self.registry.rename(entry.id, name)
                else:
                    if not entry.session_only:
                        self._delete(entry.saved.name)
                    self.registry.remove(entry.id)
            else:
                raise EModeError(f"no saved-view action {action!r}")
        self.publish("saved-changed", {"id": entry.id})
        return {"ok": True, "id": entry.id, "views": self.registry.summaries()}

    def _new_name(self, name: Any, entry_id: Optional[str] = None) -> str:
        """`name` as a saved view's name: not blank, and no other saved view's."""
        text = str(name or "").strip()
        if not text:
            raise EModeError("A saved view needs a name.")
        other = self.registry.by_name(text)
        if other is not None and other.id != entry_id:
            raise EModeError(f"A saved view named {text!r} exists already: choose another name, or open it and Save.")
        return text

    def _post(self, recipe: Any) -> None:
        if self.post_saved is not None:
            self.post_saved(recipe)

    def _delete(self, name: str) -> None:
        if self.delete_saved is not None:
            self.delete_saved(name)

    def bench_open(self, name: str) -> Dict[str, Any]:
        """The View menu's schematic or GDS view on the workbench: it switches the data, keeping the field (#146)."""
        with self._bench_lock:
            view = self.registry.bench.view
            if view is None:
                raise EModeError("the workbench shows nothing yet")
            offered = menu_of(view.data).views or []
            if name not in ("schematic", "gds") or name not in offered:
                raise EModeError(f"this plot does not offer the {name!r} view")
            entry = {d.id: d for d in self.current_inventory().datasets}.get(name)
            if entry is None or not entry.available:
                raise EModeError(entry.reason if entry is not None and entry.reason else f"no {name} view")
            self._bench_show(entry, view.pixels)
            return {"id": BENCH}

    def _bench_show(self, entry: Any, pixels: Optional[Tuple[int, int]]) -> None:
        bench = self.registry.bench
        view = bench.held.get(entry.id)
        if view is None or view.data is None or self._bench_stale(entry, view):
            view = self._bench_load(entry, view, pixels)
        bench.show(entry.id, entry.category, view)

    def _bench_stale(self, entry: Any, view: View) -> bool:
        """Data the workbench holds that EMode may have changed since: ``view_reply`` refreshes a plot's itself."""
        if view.kind in REFRESHED_KINDS or self.generation is None or self.session_ended:
            return False
        return view.generation != self.generation()

    def _bench_load(self, entry: Any, previous: Optional[View], pixels: Optional[Tuple[int, int]]) -> View:
        """`entry`'s data fetched with its arguments and the recipe `previous` kept, as the workbench's view."""
        from .views import _select

        if self.fetch_dataset is None:
            raise EModeError("this session cannot fetch the workbench's data")
        if self.session_ended:
            raise EModeError("This plot's EMode session has ended, so its data cannot be fetched.")
        recipe = previous.arguments if previous is not None else {}
        kept = KINDS[previous.kind].data_arguments if previous is not None else ()
        arguments = {**entry.arguments, **{k: v for k, v in recipe.items() if k in kept}}
        size = tuple(pixels or (previous.pixels if previous is not None else (1200, 750)))
        data = self.fetch_dataset(entry.kind, arguments, size)
        kind = KINDS[kind_of(data)]
        local = {k: v for k, v in recipe.items() if k in kind.local_arguments}
        if local:
            data = _select(data, local)
        if isinstance(data, GDSPanel):
            arguments = state_arguments(arguments, data)
        options = previous.options if previous is not None else PlotOptions()
        own = {k: v for k, v in {**arguments, **local}.items() if k in kind.data_arguments or k in kind.local_arguments}
        return View(
            id=BENCH,
            kind=kind.name,
            title=title_of(data, own),
            data=data,
            arguments=own,
            options=options,
            pixels=size,
            fetched=True,
            generation=self.generation() if self.generation is not None else 0,
        )

    def _bench_error(self, message: str, inventory: Any, category: Optional[str] = None) -> Dict[str, Any]:
        bench = self.registry.bench
        return {
            "id": BENCH,
            "error": message,
            "bench": {**_bench_menus(inventory, category or bench.category, bench.dataset), **self._origin()},
        }

    # ---------------------------------------------------------------- a view's data

    def view_reply(
        self,
        view: View,
        controls: Dict[str, Any],
        pixels: Optional[Tuple[int, int]],
        width: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply the tab's `controls` to `view`, fetching new data only when they need it.

        `pixels` is the plot's size in device pixels and `width` its width in
        CSS pixels, which sets the All modes grid's columns. A raster with more
        samples than the plot has pixels is sent decimated to them.
        """
        from .views import _select

        kind = KINDS[view.kind]
        now = self._stale(view)
        # the x-y field at a z panel's cut is drawn below it (#180): the controls apply to the z panel
        cut = view.data if isinstance(view.data, CutPanel) else None
        if cut is not None:
            view.data = cut.zpanel
        controls, grid, want_grid = self._modes_controls(view, controls)
        notice = None
        if now is not None:
            # the session has changed since this view was fetched: its menus are fetched again,
            # with its own arguments, and the panels held beside it with them
            cut = grid = None
        data_args = {k: controls[k] for k in kind.data_arguments if k in controls}
        local_args = {k: controls[k] for k in kind.local_arguments if k in controls}
        figure_args = {k: controls[k] for k in FIGURE_OPTIONS if k in controls}
        changed = {k: v for k, v in data_args.items() if view.arguments.get(k) != v}
        resize = pixels is not None and _needs_refetch(view, pixels)
        if changed or resize or now is not None:
            if self.session_ended:
                raise EModeError(
                    "This plot's EMode session has ended, so its data cannot change."
                )
            arguments = {**view.arguments, **data_args}
            sent = {k: v for k, v in arguments.items() if k not in kind.local_arguments}
            try:
                data = self.fetch(view, sent, pixels or view.pixels)
            except EModeError as exc:
                if changed or resize:
                    raise
                # only the refresh failed (a profile it names was deleted, say): show the view as it was
                data = None
                notice = (
                    f"This plot could not be brought up to date with the session, so it shows "
                    f"the session as it was when plotted: {exc}"
                )
            if data is not None:
                local = {
                    k: v for k, v in view.arguments.items() if k in kind.local_arguments
                }
                view.data = _select(data, local) if local else data
                view.arguments = arguments
                view.pixels = tuple(pixels or view.pixels)
                view.generation = now if now is not None else view.generation
        if local_args:
            view.data = _select(view.data, local_args)
            view.arguments = {**view.arguments, **local_args}
        if isinstance(view.data, GDSPanel):
            view.arguments = state_arguments(view.arguments, view.data)
        if figure_args:
            options = replace(view.options, **_clean_options(figure_args))
            # a figure option out of its bounds (an aspect ratio past 0.001 to 1000, #189) is refused
            PlotOptions.from_arguments(_options_dict(options))
            view.options = options
        if want_grid and offers_all_modes(view.data):
            view.data = self._modes_grid(view, grid, width)
        if offers_cut(view.data):
            view.data = self._with_cut(view, cut)
        view.title = title_of(view.data, view.arguments)
        spec = compose(view.data, view.options)
        aspect = figure_aspect(spec)
        if isinstance(view.data, ModesPanel):
            # small panels: one sample per CSS pixel is all they show
            css = float(width or (pixels or view.pixels)[0] / 2)
            shown: Optional[Tuple[int, int]] = (int(css), int(css * aspect))
        else:
            shown = tuple(pixels) if pixels else None
        return {
            "id": view.id,
            "kind": view.kind,
            "title": view.title,
            "figure": to_plotly(spec, shown),
            # the plot box's height over its width: the tab sizes the plot to its width (#160)
            "aspect": aspect,
            "menu": menu_of(view.data).model_dump(mode="json"),
            "controls": {
                **view.arguments,
                **_options_dict(view.options),
                **_drawn_contours(view),
                **({"all_modes": isinstance(view.data, ModesPanel)} if offers_all_modes(_single(view.data)) else {}),
            },
            "extra": _extra(view.data),
            **({"notice": notice} if notice else {}),
        }

    def _stale(self, view: View) -> Optional[int]:
        """The session's generation when `view` was fetched before it and can be fetched again, else None.

        Only a view EMode fetched is (a plot, its schematic or GDS view), and only
        while the session lives: after it ends a view is shown as it was.
        """
        if self.generation is None or self.session_ended or view.kind not in REFRESHED_KINDS:
            return None
        now = self.generation()
        return now if now != view.generation else None

    def _modes_controls(
        self, view: View, controls: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[ModesPanel], bool]:
        """The tab's controls with the All modes choice taken out: (controls, the grid shown, grid wanted).

        ``mode='All modes'`` (or ``all_modes=True``) asks for the grid, and any
        other mode for the single view. A mode the grid holds is shown from it,
        with no EMode call, when the plot's other data arguments are the grid's.
        `view.data` is left as the single view the controls then apply to.
        """
        controls = dict(controls)
        grid = view.data if isinstance(view.data, ModesPanel) else None
        if grid is not None:
            view.data = grid.view_of(grid.shown) or grid.views[0]
        if controls.get("mode") == ALL_MODES:
            controls.pop("mode")
            return controls, grid, True
        if "all_modes" in controls:
            return controls, grid, bool(controls.pop("all_modes"))
        if "mode" not in controls:
            return controls, grid, grid is not None
        if grid is not None:
            kind = KINDS[view.kind]
            asked = {**view.arguments, **{k: controls[k] for k in kind.data_arguments if k in controls}}
            panel = grid.view_of(str(controls["mode"]))
            if panel is not None and frozen_arguments(asked) == grid.arguments:
                view.data = panel
                view.arguments = {**view.arguments, "mode": controls["mode"]}
        return controls, grid, False

    def _modes_grid(self, view: View, grid: Optional[ModesPanel], width: Optional[float]) -> ModesPanel:
        """The All modes grid of `view`'s single view: `grid` again when it is still the plot's, else fetched."""
        single = view.data
        labels = mode_labels(single)
        key = frozen_arguments(view.arguments)
        shown = str(single.menu.selected.get("mode") or labels[0])
        columns = columns_for(width, len(labels))
        if grid is not None and grid.arguments == key and grid.labels == labels:
            return replace(grid, columns=columns, shown=shown)
        if self.session_ended:
            raise EModeError("This plot's EMode session has ended, so its other modes cannot be fetched.")
        views = []
        for label in labels:
            if label == shown:
                views.append(single)
                continue
            arguments = {**view.arguments, "mode": label}
            views.append(self.fetch(view, arguments, view.pixels))
        return ModesPanel(tuple(views), labels, columns, key, shown)

    def _with_cut(self, view: View, previous: Optional[CutPanel]) -> Any:
        """`view`'s z panel with the device's x-y field at its cut, fetched only when its arguments changed.

        A session that has ended keeps the x-y field it has. When EMode cannot
        draw the x-y plane there, the z panel is drawn alone, as before.
        """
        zpanel = view.data
        data_arguments = KINDS[view.kind].data_arguments
        arguments = cut_arguments(zpanel, {k: v for k, v in view.arguments.items() if k in data_arguments})
        xy = held_xy(previous, arguments)
        if xy is None:
            if self.session_ended:
                return zpanel
            try:
                xy = self.fetch(view, arguments, view.pixels)
            except EModeError:
                return zpanel
            if not isinstance(xy, PlotView) or xy.plane != "x-y":
                return zpanel
        return CutPanel(zpanel, xy, frozen_cut(arguments))

    def open_view(self, view: View, name: str) -> View:
        """Open the schematic or the GDS view of `view`'s device as a plot of its own (#146).

        The View menu offers what ``MenuState.views`` lists. Layout and
        simulation change the plot in place (``view`` is a data argument), the
        schematic and the GDS view are other kinds of plot, so each opens beside it.
        """
        offered = menu_of(view.data).views or []
        if name not in ("schematic", "gds") or name not in offered:
            raise EModeError(f"this plot does not offer the {name!r} view")
        plot_arguments = KINDS["plot"].data_arguments
        arguments = {k: v for k, v in view.arguments.items() if k in plot_arguments}
        arguments["view"] = name
        data = self.fetch(replace(view, kind=name), arguments, view.pixels)
        if isinstance(data, GDSPanel):
            arguments = state_arguments(arguments, data)
        generation = self.generation() if self.generation is not None else 0
        opened = self.registry.add(data, arguments, view.options, view.pixels, generation)
        self.publish("view-added", opened.summary())
        return opened

    def component_details(self, view: View, component: str) -> Dict[str, Any]:
        """The settings and parameters of a component of `view`'s schematic (#159).

        Only an id the schematic carries is asked for, so the tab cannot ask
        EMode about anything else.
        """
        if not isinstance(view.data, Schematic):
            raise EModeError(f"view {view.id} is not a schematic, so it has no components")
        if component not in dict(schematic_components(view.data)):
            raise EModeError(f"this schematic has no component {component!r}")
        if self.details is None:
            raise EModeError("this session cannot ask EMode for a component's details")
        if self.session_ended:
            raise EModeError("This plot's EMode session has ended, so it cannot be asked for details.")
        return self.details(component)

    def layers(self, view: View, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the layer panel's `changes` to a GDS layer view and keep them for the session.

        The tab has already restyled its figure, so the answer is the panel's
        rows (for a change it cannot make alone, such as a ``.lyp`` file), not a figure.
        """
        if not isinstance(view.data, GDSPanel):
            raise EModeError(
                f"view {view.id} is not a GDS layer view, so it has no layers"
            )
        local = {k: changes[k] for k in GDS_ARGUMENTS if changes.get(k) is not None}
        view.data = view.data.with_changes(local)
        view.arguments = state_arguments(view.arguments, view.data)
        return {
            "id": view.id,
            "layers": layer_rows(view.data),
            "shapes": view.data.shapes,
            "orientation": view.data.orientation,
        }

    def export(self, view: View, request: Dict[str, Any]) -> Tuple[bytes, str, str]:
        """`view` drawn as a file: its bytes, its type and its name.

        `request` holds ``file_type`` (``'png'``, ``'pdf'`` or ``'svg'``),
        ``file_name``, and any figure option (``figure_size`` in mm, ``dpi``,
        ``title``, ``xlabel``, ``ylabel``, ``units``, ``font_size``,
        ``colorbar_label``), applied over the view's own for this file only.
        """
        from .renderers import mpl
        from .session import compose_file_name

        file_type = str(request.get("file_type") or "png").lower().lstrip(".")
        if file_type not in EXPORT_TYPES:
            raise EModeError(
                f"file_type must be one of {EXPORT_TYPES}, got {file_type!r}"
            )
        if file_type == "pdf":
            _require_matplotlib_for_pdf()
        figure_args = {
            k: request[k] for k in FIGURE_OPTIONS if request.get(k) is not None
        }
        options = (
            replace(view.options, **_clean_options(figure_args))
            if figure_args
            else view.options
        )
        PlotOptions.from_arguments(_options_dict(options))
        spec = compose(view.data, options)
        name = compose_file_name(
            str(request.get("file_name") or f"EM_{view.id}"), file_type
        )
        return mpl.render(spec, file_type), file_type, name

    def save(self, view: View, request: Dict[str, Any]) -> Path:
        """Write `view` as a file under the working directory, ``EM_figures/`` by default.

        The name is relative to the working directory and may not leave it.
        A bare name goes in ``EM_figures/``, as ``plot(file_name=)`` saves.
        A PNG or an SVG comes as the tab drew it (``image``, a data URL from
        ``Plotly.toImage``), which is written as is, with no matplotlib (#164).
        """
        if request.get("image"):
            body, file_type, name = _tab_image(view, request)
        else:
            body, file_type, name = self.export(view, request)
        path = _save_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path.resolve()


#: What the tab shows when a PDF is asked for and matplotlib is not installed (#164).
NO_PDF = (
    "Saving as PDF needs matplotlib. Install it with: pip install matplotlib. "
    "PNG and SVG work without it."
)


def _require_matplotlib_for_pdf() -> None:
    """Refuse a PDF with the tab's warning when matplotlib is missing, rather than failing."""
    try:
        import matplotlib  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401
    except ImportError as exc:
        raise EModeError(NO_PDF) from exc


#: The file types the export panel saves through matplotlib.
EXPORT_TYPES = ("png", "pdf", "svg")

#: The content type of each exported file.
CONTENT_TYPES = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}


#: The data URL prefix of each image the tab draws itself (#164).
TAB_IMAGES = {"png": "image/png", "svg": "image/svg+xml"}


def _tab_image(view: View, request: Dict[str, Any]) -> Tuple[bytes, str, str]:
    """The tab's own PNG or SVG from its data URL: its bytes, its type and its file name."""
    import base64
    from urllib.parse import unquote

    from .session import compose_file_name

    file_type = str(request.get("file_type") or "png").lower().lstrip(".")
    if file_type not in TAB_IMAGES:
        raise EModeError(f"the tab draws only {tuple(TAB_IMAGES)} itself, so a {file_type} is not sent as an image")
    head, _, data = str(request["image"]).partition(",")
    if not head.startswith(f"data:{TAB_IMAGES[file_type]}"):
        raise EModeError(f"the image sent is not a {file_type}")
    try:
        body = base64.b64decode(data, validate=True) if head.endswith(";base64") else unquote(data).encode()
    except ValueError as exc:
        raise EModeError(f"the image sent is not a readable {file_type}") from exc
    name = compose_file_name(str(request.get("file_name") or f"EM_{view.id}"), file_type)
    return body, file_type, name


def _save_path(name: str) -> Path:
    """Where the export panel saves `name`: inside the working directory, ``EM_figures/`` by default."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise EModeError(
            f"save a figure under the working directory: {name!r} is not a relative file name there"
        )
    if path.parent == Path("."):
        path = Path("EM_figures") / path
    return Path.cwd() / path


def _extra(data: Any) -> Dict[str, Any]:
    """What a view's own controls need beyond its menu (the material explorer's entries)."""
    from ..plotting import MaterialView
    from .gds import extra as gds_extra
    from .sweep_figure import SweepPanel, sweep_menus

    if isinstance(data, GDSPanel):
        return gds_extra(data)
    if isinstance(data, Schematic):
        return {"components": [list(c) for c in schematic_components(data)]}
    if isinstance(data, SweepPanel):
        return {"menus": sweep_menus(data)}
    if isinstance(data, MaterialView):
        return {
            "requested": list(data.requested),
            "step": data.step,
            "x_fraction": data.x_fraction,
            "materials": list(data.materials),
            "reference": data.reference,
        }
    return {}


def schematic_components(view: Schematic) -> List[Tuple[str, str]]:
    """The schematic's components a click can open (#159): ``(id, label)``, sections, ports, junctions."""
    out: List[Tuple[str, str]] = []
    for b in view.blocks:
        if b.id:
            out.append((b.id, f"section {b.name}"))
    for s in view.stubs:
        if s.id and s.external:
            out.append((s.id, f"port {s.label}"))
    for c in view.connection_ids:
        if c and c not in dict(out):
            a, _, b = c.partition(":")[2].partition("|")
            out.append((c, f"junction {a} to {b}"))
    return out


#: Seconds the workbench's inventory is kept before EMode is asked again (a change of the session asks at once).
INVENTORY_SECONDS = 2.0


def _first(inventory: Any, category: Optional[str]) -> Optional[str]:
    """The first available dataset of `category`, else its first, or with no category the first available field, else trend."""
    order = [category] if category else ["field", "trend"]
    for wanted in order:
        for d in inventory.datasets:
            if d.category == wanted and d.available:
                return d.id
    if category:
        return next((d.id for d in inventory.datasets if d.category == category), None)
    return None


def dataset_for(kind: str, arguments: Dict[str, Any], entries: Dict[str, Any]) -> Optional[str]:
    """The workbench's data entry (an id of `entries`, the inventory's datasets) a plot of `kind` shows.

    A field plot is its free-space field, its profile, the device, or else the
    first field that draws. The schematic, the GDS layers and the S-matrix are
    their own entries, a material its own index or the database.
    """
    if kind == "plot":
        excitation = arguments.get("excitation")
        profile = arguments.get("profile")
        candidates = [
            f"excitation:{excitation}" if excitation else None,
            f"profile:{profile}" if profile and profile != "EME" else None,
            "device",
        ]
        for candidate in candidates:
            if candidate in entries and entries[candidate].available:
                return candidate
        return next((d.id for d in entries.values() if d.kind == "plot" and d.available), None)
    if kind in ("schematic", "gds", "smatrix"):
        return kind if kind in entries else None
    if kind == "material":
        material = arguments.get("material")
        return f"material:{material}" if f"material:{material}" in entries else ("material" if "material" in entries else None)
    if kind == "monitor":
        name = f"monitor:{arguments.get('name')}"
        return name if name in entries else None
    return None


def _bench_menus(inventory: Any, category: str, dataset: Optional[str]) -> Dict[str, Any]:
    """The workbench's own menus: Field or Trend, and the data of the category, each with why it is greyed."""
    fields = {"id", "kind", "label", "available", "reason", "summary"}
    return {
        "categories": ["field", "trend"],
        "category": category,
        "dataset": dataset,
        "datasets": [d.model_dump(mode="json", include=fields) for d in inventory.datasets if d.category == category],
    }


#: The kinds of view fetched again when the session has changed since (#182). The others
#: hold data the client made or was given (a sweep, a far field, a figure, an S-matrix
#: the script holds), which EMode cannot answer again.
REFRESHED_KINDS = ("plot", "schematic", "gds")


def _pixels(value: Any) -> Optional[Tuple[int, int]]:
    """The tab's plot size in device pixels, a side of 0 or less never passed on to EMode.

    A phone's plot has no height before its first reply sizes it, and EMode
    samples a panel at the pixels it is given: 0 tall came back one row
    (emode#1701). A missing height is the width times the plot box's default
    aspect, and a missing width is no size at all.
    """
    if not value:
        return None
    width, height = (int(p) for p in value)
    if width <= 0:
        return None
    return width, height if height > 0 else max(1, round(width / PLOT_RATIO))


def _needs_refetch(view: View, pixels: Tuple[int, int]) -> bool:
    """A z panel sampled for a panel more than a quarter wider or narrower than the tab's."""
    data = view.data
    if getattr(data, "plane", None) not in ("z-x", "z-y"):
        return False
    if view.arguments.get("resolution") not in (None, "auto", "Auto"):
        return False
    old = max(1, int(view.pixels[0]))
    return abs(int(pixels[0]) - old) / old > REFETCH_FRACTION


def _clean_options(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """The tab's figure options as ``PlotOptions`` fields: an empty entry is the option's default.

    A size with an empty side is no size (the default), and a number the tab
    sent as text is a number.
    """
    defaults = PlotOptions()
    out = dict(arguments)
    for name, value in arguments.items():
        if value is None or value == "":
            out[name] = getattr(defaults, name)
    size = out.get("figure_size")
    if size is not None:
        try:
            width, height = (float(v) for v in size)
        except (TypeError, ValueError):
            out["figure_size"] = None
        else:
            out["figure_size"] = (width, height) if width > 0 and height > 0 else None
    for name in ("font_size", "dpi", "vmin", "vmax", "xmin", "xmax", "ymin", "ymax", "aspect_ratio"):
        if isinstance(out.get(name), str):
            try:
                out[name] = float(out[name])
            except ValueError:
                out[name] = getattr(defaults, name)
    if out.get("dpi") is not None:
        out["dpi"] = int(out["dpi"])
    return out


def _single(data: Any) -> Any:
    """The single view of `data`: the shown mode's panel of an All modes grid, a cut's z panel, else `data`."""
    if isinstance(data, ModesPanel):
        return data.view_of(data.shown) or data.views[0]
    if isinstance(data, CutPanel):
        return data.zpanel
    return data


def _drawn_contours(view: View) -> Dict[str, Any]:
    """The Contours control as drawn: the default None resolved, so the checkbox shows the truth."""
    drawn = effective_contours(_single(view.data), view.options)
    return {} if drawn is None else {"contours": drawn}


def _options_dict(options: PlotOptions) -> Dict[str, Any]:
    return {f: getattr(options, f) for f in FIGURE_OPTIONS}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class _Handler(BaseHTTPRequestHandler):
    server_version = "EMode"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    @property
    def ui(self) -> UIServer:
        return self.server.ui  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Nothing is logged: a request line can carry the token."""

    # ---------------------------------------------------------------- guards

    def _forbid(self) -> None:
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _origin_ok(self) -> bool:
        if self.headers.get("Host") not in self.ui.allowed_hosts():
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in self.ui.allowed_origins()

    def _token_ok(self, from_query: bool = False) -> bool:
        if from_query:
            given = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
        else:
            given = self.headers.get("X-EMode-Token", "")
        return bool(given) and secrets.compare_digest(given, self.ui.token)

    # ---------------------------------------------------------------- replies

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, default=_json_default).encode()
        self._send(status, body, "application/json", {"Cache-Control": "no-store"})

    def _drain(self) -> None:
        """Read the request's body off the connection, once, before any reply.

        The server keeps connections alive (HTTP/1.1), so a body left unread
        would be read as the start of the next request (emodeconnection#156).
        """
        if getattr(self, "_raw_body", None) is not None:
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        self._raw_body = self.rfile.read(length) if length > 0 else b""

    def _body(self) -> Dict[str, Any]:
        self._drain()
        if not self._raw_body:
            return {}
        try:
            data = json.loads(self._raw_body)
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    # ---------------------------------------------------------------- methods

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._raw_body = None
        self._drain()
        self._forbid()

    def do_PUT(self) -> None:  # noqa: N802
        self._raw_body = None
        self._drain()
        self._forbid()

    def do_DELETE(self) -> None:  # noqa: N802
        self._raw_body = None
        self._drain()
        self._forbid()

    def do_GET(self) -> None:  # noqa: N802
        self._raw_body = None
        self._drain()
        if not self._origin_ok():
            return self._forbid()
        path = urlsplit(self.path).path
        if path.startswith("/static/"):
            return self._static(path[len("/static/") :])
        if path in ("/", "/index.html"):
            if not self._token_ok(from_query=True):
                return self._forbid()
            body = (PAGES / "app.html").read_bytes()
            return self._send(
                200,
                body,
                "text/html; charset=utf-8",
                {"Content-Security-Policy": CSP, "Cache-Control": "no-store"},
            )
        if not path.startswith("/api/") or not self._token_ok():
            return self._forbid()
        if path == "/api/events":
            return self._events()
        if path == "/api/views":
            return self._json(200, self.ui.registry.summaries())
        if path == "/api/inventory":
            try:
                return self._json(200, self.ui.inventory_reply())
            except EModeError as exc:
                return self._json(200, {"error": str(exc)})
        if path == "/api/colormaps":
            return self._json(200, colormaps.all_stops())
        if path == "/api/status":
            return self._json(
                200,
                {
                    "session_ended": self.ui.session_ended,
                    "views": len(self.ui.registry),
                },
            )
        if path == "/api/session":
            return self._json(200, self.ui.session_page())
        if path.startswith("/api/login/"):
            return self._login("GET", path)
        return self._json(404, {"error": "no such route"})

    def do_POST(self) -> None:  # noqa: N802
        self._raw_body = None
        self._drain()
        if not self._origin_ok() or not self._token_ok():
            return self._forbid()
        path = urlsplit(self.path).path
        if path == "/api/ping":
            self.ui.ping()
            return self._json(200, {"ok": True, "session_ended": self.ui.session_ended})
        if path == "/api/closed":
            self.ui.tab_closed = True
            return self._json(200, {"ok": True})
        if path.startswith("/api/login/"):
            return self._login("POST", path)
        if path == "/api/saved":
            try:
                return self._json(200, self.ui.saved_action(self._body()))
            except EModeError as exc:
                return self._json(200, {"error": str(exc)})
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:3] == ["api", "views", BENCH] and parts[3] in ("data", "open", "fetched"):
            return self._bench(parts[3])
        if len(parts) == 4 and parts[:2] == ["api", "views"]:
            view = self.ui.registry.get(parts[2])
            if view is None:
                return self._json(404, {"error": f"no view {parts[2]}"})
            if parts[3] == "fetched":
                self.ui.registry.mark_fetched(view.id)
                return self._json(200, {"ok": True})
            if parts[3] in ("save", "export"):
                return self._export(view, parts[3])
            if parts[3] == "open":
                try:
                    opened = self.ui.open_view(view, str(self._body().get("view")))
                except EModeError as exc:
                    return self._json(200, {"id": view.id, "error": str(exc)})
                return self._json(200, {"id": opened.id})
            if parts[3] == "details":
                try:
                    reply = self.ui.component_details(view, str(self._body().get("component")))
                except EModeError as exc:
                    return self._json(200, {"id": view.id, "error": str(exc)})
                return self._json(200, reply)
            if parts[3] == "layers":
                try:
                    return self._json(200, self.ui.layers(view, self._body()))
                except EModeError as exc:
                    return self._json(200, {"id": view.id, "error": str(exc)})
            if parts[3] == "data":
                body = self._body()
                controls = body.get("controls") or {}
                pixels = _pixels(body.get("pixels"))
                width = body.get("width")
                width = float(width) if isinstance(width, (int, float)) and width > 0 else None
                try:
                    reply = self.ui.view_reply(view, controls, pixels, width)
                except EModeError as exc:
                    return self._json(200, {"id": view.id, "error": str(exc)})
                return self._json(200, reply)
        return self._json(404, {"error": "no such route"})

    # ---------------------------------------------------------------- routes

    def _bench(self, action: str) -> None:
        """The workbench's routes: its data, the View menu's switch, and the tab's acknowledgment."""
        body = self._body()
        if action == "fetched":
            return self._json(200, {"ok": True})
        try:
            if action == "open":
                return self._json(200, self.ui.bench_open(str(body.get("view"))))
            pixels = _pixels(body.get("pixels"))
            width = body.get("width")
            width = float(width) if isinstance(width, (int, float)) and width > 0 else None
            return self._json(200, self.ui.bench_reply(body.get("controls") or {}, pixels, width))
        except EModeError as exc:
            return self._json(200, {"id": BENCH, "error": str(exc)})

    def _login(self, method: str, path: str) -> None:
        """The account page's routes: a read is a GET, a change a POST whose body may hold a password."""
        action = path[len("/api/login/") :]
        body = self._body() if method == "POST" else {}
        reply = self.ui.account(method, action, body)
        body.clear()
        self._raw_body = b""
        if reply is None:
            return self._json(404, {"error": "no such route"})
        return self._json(200, reply)

    def _export(self, view: View, action: str) -> None:
        """Save `view` under the working directory, or send it back as a file to download."""
        request = self._body()
        try:
            if action == "save":
                path = self.ui.save(view, request)
                return self._json(200, {"id": view.id, "path": str(path)})
            body, file_type, name = self.ui.export(view, request)
        except EModeError as exc:
            return self._json(200, {"id": view.id, "error": str(exc)})
        safe = (
            "".join(c for c in Path(name).name if c.isalnum() or c in "._- ")
            or f"EM_{view.id}.{file_type}"
        )
        self._send(
            200,
            body,
            CONTENT_TYPES[file_type],
            {
                "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="{safe}"',
            },
        )

    def _static(self, name: str) -> None:
        root = STATIC.resolve()
        target = (STATIC / name).resolve()
        if root not in target.parents or not target.is_file():
            return self._forbid()
        content_type = (
            mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        )
        if target.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self._send(
            200, target.read_bytes(), content_type, {"Cache-Control": "no-cache"}
        )

    def _events(self) -> None:
        """Stream server-sent events until the tab goes or the session stops."""
        q = self.ui.subscribe()
        self.ui.ping()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    item = q.get(timeout=10.0)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    break
                event, data = item
                payload = json.dumps(data, default=_json_default)
                self.wfile.write(f"event: {event}\ndata: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.ui.unsubscribe(q)
            self.close_connection = True
