"""``plot()`` of an ``EMode`` session: fetch the panel, register it, show it in the one tab.

``plot()`` returns at once with a ``PlotHandle``. The panel's data is fetched
from EMode at the call (``EM_plot``), so the tab loads it from this client and
never waits on the solver, and a script that ends right after ``plot()`` still
shows its plot: ``close()`` keeps serving until the tab has fetched every view,
for at most ``EMODE_UI_EXIT_WAIT`` seconds (15 by default).

The session has one browser tab. The first ``plot()`` opens it, and later ones
tell it over the event stream while it is alive (it pings every two seconds),
or open it again after it has been gone five seconds. ``plot(block=True)``
waits until the tab closes.

Where no browser is wanted the panel is drawn here with matplotlib and
returned as a ``PlotImage``: with ``file_name`` (saved under ``EM_figures/``),
with ``EMODE_HEADLESS=1``, on Linux without a display, and with
``inline='png'`` (or ``EMODE_PLOT_INLINE=png``) in a notebook.
``EMODE_NONINTERACTIVE=1`` fetches and registers the panel and opens nothing.
"""
from __future__ import annotations

import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np

from ..plot import PlotImage
from ..plotting import PlotView, Schematic
from ..types import EModeError
from .figure import FigureSpec, PlotOptions, check_figure_arguments, compose, split_arguments
from .server import UIServer
from .views import View

if TYPE_CHECKING:
    from ..emode import EMode

#: The panel size a view is fetched at before the tab says its own, in device pixels.
DEFAULT_PIXELS: Tuple[int, int] = (1200, 750)

#: What a 1.0.9 EMode says in its ``version()`` answer under ``'plot'``: it answers
#: ``EM_plot`` with a view to draw rather than drawing a window.
PLOT_VIEW = "view"

#: Seconds ``close()`` waits for the tab to fetch its views, by default.
EXIT_WAIT = 15.0

#: Seconds ``plot(block=True)`` waits for the tab to connect before giving up.
CONNECT_WAIT = 30.0

# The clock and the pause, which the tests replace.
clock = time.monotonic
sleep = time.sleep


def headless_requested(file_name: Optional[str]) -> bool:
    """Whether a plot is drawn here as an image rather than in the browser tab."""
    if file_name:
        return True
    if os.environ.get("EMODE_HEADLESS") == "1":
        return True
    if sys.platform.startswith("linux"):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


def noninteractive() -> bool:
    """``EMODE_NONINTERACTIVE=1``: every plot returns at once, opens nothing and waits for nothing."""
    return bool(os.environ.get("EMODE_NONINTERACTIVE"))


def compose_file_name(file_name: str, file_type: str) -> str:
    """``file_name.file_type``, keeping an extension the name already has."""
    stem = file_name.rstrip(".")
    extension = file_type.lstrip(".")
    if not extension or stem.lower().endswith(f".{extension.lower()}"):
        return stem
    return f"{stem}.{extension}"


def check_plot_support(em: "EMode") -> None:
    """Refuse ``plot()`` before sending it to an EMode that draws its own window."""
    info = em._emode_info()
    if isinstance(info, dict) and info.get("plot") == PLOT_VIEW:
        return
    version = info.get("emode") if isinstance(info, dict) else None
    this = version or "older than 1.0.7"
    raise EModeError(
        f"plot() needs EMode 1.0.9 or newer, and this EMode is {this}. Update it with emode-install."
    )


def fetch_view(em: "EMode", arguments: Dict[str, Any], pixels: Tuple[int, int]) -> Union[PlotView, Schematic]:
    """Ask EMode for the panel of `arguments` at a panel of `pixels`."""
    data = em.call("EM_plot", **arguments, pixels=[int(pixels[0]), int(pixels[1])])
    if not isinstance(data, (PlotView, Schematic)):
        raise EModeError(
            "EMode answered plot() with something this client cannot draw. "
            "Update the client with pip install -U emode."
        )
    return data


def ensure_server(em: "EMode") -> UIServer:
    """The session's local server, started on first use."""
    server = em.__dict__.get("_ui")
    if server is None or not server.running:

        def refetch(view: View, arguments: Dict[str, Any], pixels: Tuple[int, int]) -> Any:
            if not em.running:
                raise EModeError("This plot's EMode session has ended, so its data cannot change.")
            return fetch_view(em, arguments, pixels)

        server = UIServer(em._views, refetch, clock=clock).start()
        em._ui = server
    return server


def headless_image(
    data: Union[PlotView, Schematic],
    options: PlotOptions,
    file_name: Optional[str],
    file_type: str,
    inline: bool = True,
) -> PlotImage:
    """The panel drawn with matplotlib: saved when `file_name` is given, and as a 150 dpi PNG.

    The PNG is left out (``png=None``) when a file was saved and `inline` is
    False, as ``PlotImage`` has always said: it is sent when no file was asked
    for, and always to a notebook, which shows it.
    """
    from .renderers import mpl

    spec = compose(data, options)
    saved = None
    if file_name:
        name = compose_file_name(file_name, file_type or "png")
        extension = Path(name).suffix.lstrip(".").lower() or "png"
        path = Path("EM_figures") / name
        mpl.save(spec, path, extension)
        saved = str(path.resolve())
    if saved is not None and not inline:
        return PlotImage(file=saved, png=None, dpi=mpl.INLINE_DPI)
    png = mpl.render(spec, "png", dpi=mpl.INLINE_DPI)
    return PlotImage(file=saved, png=np.frombuffer(png, dtype=np.uint8), dpi=mpl.INLINE_DPI)


def open_tab(em: "EMode", server: UIServer, view: View) -> None:
    """Show `view` in the session's tab: tell a live tab, or open one.

    A tab opened less than `CONNECT_WAIT` seconds ago that has not pinged yet
    is still starting (a cold browser), so it is told rather than opened again.
    """
    server.publish("view-added", view.summary())
    if server.tab_alive():
        return
    starting = server.opened_at is not None and server.last_ping is None
    if starting and clock() - server.opened_at < CONNECT_WAIT:
        return
    em._ui_opened = True
    server.opened_at = clock()
    server.last_ping = None
    webbrowser.open(server.url(view.id))


def plot(
    em: "EMode",
    block: bool = False,
    inline: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: str = "png",
    **arguments: Any,
) -> Union["PlotHandle", PlotImage, None]:
    check_figure_arguments(arguments)
    data_args, figure_args = split_arguments(arguments)
    options = PlotOptions.from_arguments(figure_args)
    if inline not in (None, "png"):
        raise EModeError("inline is 'png' or None")
    inline = inline or os.environ.get("EMODE_PLOT_INLINE") or None
    check_plot_support(em)

    if headless_requested(file_name) or inline == "png":
        notebook = em.in_ipython()
        data = fetch_view(em, data_args, DEFAULT_PIXELS)
        image = headless_image(data, options, file_name, file_type, inline=notebook)
        if notebook:
            from IPython.display import display  # type: ignore

            display(image)
            return None
        return image

    data = fetch_view(em, data_args, DEFAULT_PIXELS)
    view = em._views.add(data, data_args, options, DEFAULT_PIXELS)
    handle = PlotHandle(em, view)
    if noninteractive():
        return handle
    server = ensure_server(em)
    if em.in_ipython():
        # the cell shows the handle's frame, pinned to this view
        return handle
    open_tab(em, server, view)
    if block:
        wait_for_tab_to_close(server)
    return handle


def wait_for_tab_to_close(server: UIServer) -> None:
    """Wait until the tab has connected and then gone (closed, or silent for five seconds)."""
    start = clock()
    try:
        while server.last_ping is None and not server.tab_closed:
            if clock() - start > CONNECT_WAIT:
                return
            sleep(0.1)
        while server.tab_alive():
            sleep(0.1)
    except KeyboardInterrupt:
        return


def exit_wait(em: "EMode") -> None:
    """At ``close()``: serve until the tab has every view, then stop the server."""
    server = em.__dict__.get("_ui")
    if server is None or not server.running:
        return
    server.session_ended = True
    try:
        if em.__dict__.get("_ui_opened") and not noninteractive() and not em.__dict__.get("_ipython"):
            limit = float(os.environ.get("EMODE_UI_EXIT_WAIT") or EXIT_WAIT)
            if not em._views.all_fetched() and not server.tab_closed:
                print("EMode: waiting for the plot tab to load, Ctrl-C to skip", flush=True)
                start = clock()
                try:
                    while (
                        not em._views.all_fetched()
                        and not server.tab_closed
                        and clock() - start < limit
                    ):
                        sleep(0.1)
                except KeyboardInterrupt:
                    pass
    finally:
        server.stop()


class PlotHandle:
    """What ``plot()`` returns: the view, its figure, and ways to show or save it.

    - ``id``: the view's id in the session's plots list.
    - ``view``: the ``PlotView`` (or ``Schematic``) EMode returned, the arrays the plot shows.
    - ``figure``: the ``FigureSpec`` drawn from it with the call's options.
    - ``url``: the app tab's address, opened on this view.
    - ``png(dpi=150)``, ``save(file_name, file_type='png')``: a static image, drawn with matplotlib.
    - ``open()``: open or reopen the tab on this view.
    """

    def __init__(self, em: "EMode", view: View) -> None:
        self._em = em
        self._view = view

    @property
    def id(self) -> str:
        return self._view.id

    @property
    def view(self) -> Union[PlotView, Schematic]:
        return self._view.data

    @property
    def figure(self) -> FigureSpec:
        return compose(self._view.data, self._view.options)

    @property
    def url(self) -> Optional[str]:
        server = self._em.__dict__.get("_ui")
        if server is None or not server.running:
            return None
        return _public_url(server, self.id, embed=False)

    def png(self, dpi: int = 150) -> bytes:
        from .renderers import mpl

        return mpl.render(self.figure, "png", dpi=dpi)

    def save(self, file_name: str, file_type: str = "png") -> Path:
        from .renderers import mpl

        name = compose_file_name(file_name, file_type)
        path = Path(name)
        if not path.is_absolute() and path.parent == Path("."):
            path = Path("EM_figures") / path
        return mpl.save(self.figure, path, path.suffix.lstrip(".").lower() or "png").resolve()

    def open(self) -> None:
        server = ensure_server(self._em)
        self._em._ui_opened = True
        server.opened_at = clock()
        webbrowser.open(server.url(self.id))

    def _repr_html_(self) -> str:
        server = ensure_server(self._em)
        url = _public_url(server, self.id, embed=True)
        return (
            f'<iframe src="{url}" width="900" height="600" style="border:0"></iframe>'
            f'<div style="font-size:11px">EMode plot {self.id}: <a href="{_public_url(server, self.id, False)}" '
            f'target="_blank" rel="noopener noreferrer">open in a tab</a></div>'
        )

    def __repr__(self) -> str:
        return f"<PlotHandle {self.id}: {self._view.title}>"


def _public_url(server: UIServer, view_id: str, embed: bool) -> str:
    """The app's address as the browser reaches it: through Colab's proxy on Colab."""
    url = server.url(view_id, embed=embed)
    if "google.colab" in sys.modules and not os.environ.get("EMODE_UI_URL"):
        try:
            from google.colab.output import eval_js  # type: ignore

            base = eval_js(f"google.colab.kernel.proxyPort({server.port})").rstrip("/")
            return base + url[len(server.base_url()):]
        except Exception:  # noqa: BLE001  (no proxy: the loopback address is all there is)
            return url
    return url
