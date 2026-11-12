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
from .figure import PlotOptions, compose
from .gds import LOCAL_ARGUMENTS as GDS_ARGUMENTS
from .gds import GDSPanel, layer_rows, state_arguments
from .renderers.plotly import to_plotly
from .views import KINDS, View, ViewRegistry, menu_of, title_of

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
    ) -> None:
        self.registry = registry
        self.fetch = fetch
        self.session_info = session_info
        self.clock = clock
        self.token = secrets.token_urlsafe(32)
        self._requested_port = port if port is not None else int(os.environ.get("EMODE_UI_PORT") or 0)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._subscribers: List["queue.Queue[Optional[Tuple[str, Any]]]"] = []
        self._lock = threading.Lock()
        self.last_ping: Optional[float] = None
        self.opened_at: Optional[float] = None
        self.tab_closed = False
        self.session_ended = False

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
            target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, name="emode-ui", daemon=True
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

    def url(self, view_id: Optional[str] = None, embed: bool = False) -> str:
        """The app's address with this session's token, opened on `view_id` when given."""
        query = f"?token={self.token}" + ("&embed=1" if embed else "")
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

    # ---------------------------------------------------------------- a view's data

    def view_reply(self, view: View, controls: Dict[str, Any], pixels: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        """Apply the tab's `controls` to `view`, fetching new data only when they need it."""
        from .views import _select

        kind = KINDS[view.kind]
        data_args = {k: controls[k] for k in kind.data_arguments if k in controls}
        local_args = {k: controls[k] for k in kind.local_arguments if k in controls}
        figure_args = {k: controls[k] for k in FIGURE_OPTIONS if k in controls}
        changed = {k: v for k, v in data_args.items() if view.arguments.get(k) != v}
        resize = pixels is not None and _needs_refetch(view, pixels)
        if changed or resize:
            if self.session_ended:
                raise EModeError("This plot's EMode session has ended, so its data cannot change.")
            arguments = {**view.arguments, **data_args}
            sent = {k: v for k, v in arguments.items() if k not in kind.local_arguments}
            data = self.fetch(view, sent, pixels or view.pixels)
            local = {k: v for k, v in view.arguments.items() if k in kind.local_arguments}
            view.data = _select(data, local) if local else data
            view.arguments = arguments
            view.pixels = tuple(pixels or view.pixels)
        if local_args:
            view.data = _select(view.data, local_args)
            view.arguments = {**view.arguments, **local_args}
        if isinstance(view.data, GDSPanel):
            view.arguments = state_arguments(view.arguments, view.data)
        if figure_args:
            view.options = replace(view.options, **_clean_options(figure_args))
        view.title = title_of(view.data, view.arguments)
        spec = compose(view.data, view.options)
        return {
            "id": view.id,
            "kind": view.kind,
            "title": view.title,
            "figure": to_plotly(spec),
            "menu": menu_of(view.data).model_dump(mode="json"),
            "controls": {**view.arguments, **_options_dict(view.options)},
            "extra": _extra(view.data),
        }

    def layers(self, view: View, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the layer panel's `changes` to a GDS layer view and keep them for the session.

        The tab has already restyled its figure, so the answer is the panel's
        rows (for a change it cannot make alone, such as a ``.lyp`` file), not a figure.
        """
        if not isinstance(view.data, GDSPanel):
            raise EModeError(f"view {view.id} is not a GDS layer view, so it has no layers")
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
            raise EModeError(f"file_type must be one of {EXPORT_TYPES}, got {file_type!r}")
        figure_args = {k: request[k] for k in FIGURE_OPTIONS if request.get(k) is not None}
        options = replace(view.options, **_clean_options(figure_args)) if figure_args else view.options
        PlotOptions.from_arguments(_options_dict(options))
        spec = compose(view.data, options)
        name = compose_file_name(str(request.get("file_name") or f"EM_{view.id}"), file_type)
        return mpl.render(spec, file_type), file_type, name

    def save(self, view: View, request: Dict[str, Any]) -> Path:
        """Write `view` as a file under the working directory, ``EM_figures/`` by default.

        The name is relative to the working directory and may not leave it.
        A bare name goes in ``EM_figures/``, as ``plot(file_name=)`` saves.
        """
        body, file_type, name = self.export(view, request)
        path = _save_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return path.resolve()


#: The file types the export panel saves through matplotlib.
EXPORT_TYPES = ("png", "pdf", "svg")

#: The content type of each exported file.
CONTENT_TYPES = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}


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

    if isinstance(data, GDSPanel):
        return gds_extra(data)
    if isinstance(data, MaterialView):
        return {
            "requested": list(data.requested),
            "step": data.step,
            "x_fraction": data.x_fraction,
            "materials": list(data.materials),
            "reference": data.reference,
        }
    return {}


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
    for name in ("font_size", "dpi", "vmin", "vmax"):
        if isinstance(out.get(name), str):
            try:
                out[name] = float(out[name])
            except ValueError:
                out[name] = getattr(defaults, name)
    if out.get("dpi") is not None:
        out["dpi"] = int(out["dpi"])
    return out


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

    def _send(self, status: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None) -> None:
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

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(length))
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    # ---------------------------------------------------------------- methods

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._forbid()

    def do_PUT(self) -> None:  # noqa: N802
        self._forbid()

    def do_DELETE(self) -> None:  # noqa: N802
        self._forbid()

    def do_GET(self) -> None:  # noqa: N802
        if not self._origin_ok():
            return self._forbid()
        path = urlsplit(self.path).path
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path in ("/", "/index.html"):
            if not self._token_ok(from_query=True):
                return self._forbid()
            body = (PAGES / "app.html").read_bytes()
            return self._send(
                200, body, "text/html; charset=utf-8",
                {"Content-Security-Policy": CSP, "Cache-Control": "no-store"},
            )
        if not path.startswith("/api/") or not self._token_ok():
            return self._forbid()
        if path == "/api/events":
            return self._events()
        if path == "/api/views":
            return self._json(200, [v.summary() for v in self.ui.registry.list()])
        if path == "/api/colormaps":
            return self._json(200, colormaps.all_stops())
        if path == "/api/status":
            return self._json(200, {"session_ended": self.ui.session_ended, "views": len(self.ui.registry)})
        if path == "/api/session":
            return self._json(200, self.ui.session_page())
        return self._json(404, {"error": "no such route"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_ok() or not self._token_ok():
            return self._forbid()
        path = urlsplit(self.path).path
        if path == "/api/ping":
            self.ui.ping()
            return self._json(200, {"ok": True, "session_ended": self.ui.session_ended})
        if path == "/api/closed":
            self.ui.tab_closed = True
            return self._json(200, {"ok": True})
        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "views"]:
            view = self.ui.registry.get(parts[2])
            if view is None:
                return self._json(404, {"error": f"no view {parts[2]}"})
            if parts[3] == "fetched":
                self.ui.registry.mark_fetched(view.id)
                return self._json(200, {"ok": True})
            if parts[3] in ("save", "export"):
                return self._export(view, parts[3])
            if parts[3] == "layers":
                try:
                    return self._json(200, self.ui.layers(view, self._body()))
                except EModeError as exc:
                    return self._json(200, {"id": view.id, "error": str(exc)})
            if parts[3] == "data":
                body = self._body()
                controls = body.get("controls") or {}
                pixels = body.get("pixels")
                pixels = tuple(int(p) for p in pixels) if pixels else None
                try:
                    reply = self.ui.view_reply(view, controls, pixels)
                except EModeError as exc:
                    return self._json(200, {"id": view.id, "error": str(exc)})
                return self._json(200, reply)
        return self._json(404, {"error": "no such route"})

    # ---------------------------------------------------------------- routes

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
        safe = "".join(c for c in Path(name).name if c.isalnum() or c in "._- ") or f"EM_{view.id}.{file_type}"
        self._send(
            200, body, CONTENT_TYPES[file_type],
            {"Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="{safe}"'},
        )

    def _static(self, name: str) -> None:
        root = STATIC.resolve()
        target = (STATIC / name).resolve()
        if root not in target.parents or not target.is_file():
            return self._forbid()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self._send(200, target.read_bytes(), content_type, {"Cache-Control": "no-cache"})

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
