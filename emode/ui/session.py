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
import weakref
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..plot import PlotImage
from ..plotting import MaterialView, MenuState, PlotView, Schematic, SessionInfo, SMatrixView
from ..types import EModeError
from ..gds import GDSView
from .figure import (
    DATA_ARGUMENTS,
    FarFieldPanel,
    FigureSpec,
    PlotOptions,
    check_figure_arguments,
    compose,
    split_arguments,
)
from .gds import LOCAL_ARGUMENTS as GDS_ARGUMENTS
from .gds import GDSPanel, state_arguments
from .server import UIServer
from .views import KINDS, View, _select

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


def check_plot_support(em: "EMode", kind: str = "plot") -> None:
    """Refuse a plot call before sending it to an EMode that draws its own window.

    ``plot()`` needs ``'plot': 'view'`` in EMode's ``version()`` answer
    (1.0.9). ``plot_S_matrix()``, ``material_explorer()`` and the session page
    also need their kind listed under ``'views'``.
    """
    info = em._emode_info()
    function = {
        "smatrix": "plot_S_matrix()",
        "material": "material_explorer()",
        "session": "The session page",
        "details": "A schematic component's details",
    }
    if isinstance(info, dict) and info.get("plot") == PLOT_VIEW:
        if kind in ("plot", "schematic") or kind in (info.get("views") or ()):
            return
    version = info.get("emode") if isinstance(info, dict) else None
    this = version or "older than 1.0.7"
    raise EModeError(
        f"{function.get(kind, 'plot()')} needs EMode 1.0.9 or newer, and this EMode is {this}. "
        "Update it with emode-install."
    )


#: The EMode function each kind of view is fetched with.
FUNCTIONS = {
    "plot": "EM_plot",
    "schematic": "EM_plot",
    "gds": "EM_plot",
    "smatrix": "EM_plot_S_matrix",
    "material": "EM_material_explorer",
}

#: What each function answers with.
ANSWERS = {
    "EM_plot": (PlotView, Schematic, GDSView),
    "EM_plot_S_matrix": (SMatrixView,),
    "EM_material_explorer": (MaterialView,),
}


def fetch_view(
    em: "EMode", arguments: Dict[str, Any], pixels: Tuple[int, int], kind: str = "plot"
) -> Any:
    """Ask EMode for the data of a view of `kind` with `arguments`, at a panel of `pixels`."""
    function = FUNCTIONS[kind]
    data = em.call(function, **arguments, pixels=[int(pixels[0]), int(pixels[1])])
    if not isinstance(data, ANSWERS[function]):
        raise EModeError(
            "EMode answered plot() with something this client cannot draw. "
            "Update the client with pip install -U emode."
        )
    if isinstance(data, GDSView):
        data = GDSPanel.from_view(data)
    return data


def fetch_details(em: "EMode", component: str) -> Dict[str, Any]:
    """Ask EMode for every setting and parameter of a schematic component (#159), as JSON for the tab."""
    from ..plotting import ComponentDetails

    if not em.running:
        raise EModeError("This plot's EMode session has ended, so it cannot be asked for details.")
    check_plot_support(em, "details")
    details = em.call("EM_component_details", component=component)
    if not isinstance(details, ComponentDetails):
        raise EModeError("EMode answered with something this client cannot read. Update it with pip install -U emode.")
    return details.model_dump(mode="json")


def client_rows(server: Optional[UIServer] = None) -> Dict[str, Any]:
    """This client's rows of the session page: its version, Python, its process and the tab."""
    from ..types import client_version

    rows: Dict[str, Any] = {
        "version": client_version(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pid": os.getpid(),
        "ui_port": server.port if server is not None and server.running else None,
        "views": len(server.registry) if server is not None else 0,
    }
    try:
        import psutil  # type: ignore
    except ImportError:
        rows["memory"] = "install psutil (pip install psutil) to show this client's memory and CPU"
        return rows
    process = psutil.Process()
    rows["rss_bytes"] = int(process.memory_info().rss)
    rows["cpu_percent"] = float(process.cpu_percent(interval=None))
    rows["threads"] = int(process.num_threads())
    return rows


#: The documentation site the tab links to.
DOCS_URL = "https://docs.emodephotonix.com/"


def ensure_server(em: "EMode") -> UIServer:
    """The session's local server, started on first use."""
    server = em.__dict__.get("_ui")
    if server is None or not server.running:

        def refetch(view: View, arguments: Dict[str, Any], pixels: Tuple[int, int]) -> Any:
            if not em.running:
                raise EModeError("This plot's EMode session has ended, so its data cannot change.")
            return fetch_view(em, _sent(view.kind, arguments), pixels, view.kind)

        def session_info() -> Dict[str, Any]:
            if not em.running:
                raise EModeError("This EMode session has ended.")
            check_plot_support(em, "session")
            info = em.call("EM_session_info")
            if not isinstance(info, SessionInfo):
                raise EModeError("EMode answered session_info() with something this client cannot read.")
            return info.model_dump(mode="json")

        server = UIServer(
            em._views,
            refetch,
            clock=clock,
            session_info=session_info,
            details=lambda component: fetch_details(em, component),
        ).start()
        em._ui = server
    return server


def headless_image(
    data: Any,
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


def _sent(kind: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """The arguments of a view EMode computes: the client-side selection stays here."""
    local = KINDS[kind].local_arguments if kind in ("material", "farfield", "gds") else ()
    return {k: v for k, v in arguments.items() if k not in local}


def plot(
    em: "EMode",
    block: bool = False,
    inline: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: str = "png",
    **arguments: Any,
) -> Union["PlotHandle", PlotImage, None]:
    if arguments.get("view") == "gds":
        # the GDS layer view: its layer panel's arguments stay in this client
        data_args, figure_args = _split(arguments, DATA_ARGUMENTS + GDS_ARGUMENTS, "plot")
        return show_call(em, "gds", data_args, figure_args, block, inline, file_name, file_type)
    arguments = _physical_space(arguments)
    check_figure_arguments(arguments)
    data_args, figure_args = split_arguments(arguments)
    return show_call(em, "plot", data_args, figure_args, block, inline, file_name, file_type)


def _physical_space(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """`arguments` with the old ``physical_space`` turned into the view it means (emode#1649).

    The view sets the space: ``view='layout'`` is physical space and
    ``view='simulation'`` the simulation space. An explicit ``view`` wins.
    """
    if "physical_space" not in arguments:
        return arguments
    import warnings

    out = dict(arguments)
    physical = out.pop("physical_space")
    warnings.warn(
        "plot(physical_space=...) is deprecated: the view sets the space. "
        "Use view='layout' for physical space or view='simulation' for the simulation space.",
        DeprecationWarning,
        stacklevel=3,
    )
    if out.get("view") is None and physical is not None:
        out["view"] = "layout" if physical else "simulation"
    return out


def show_call(
    em: "EMode",
    kind: str,
    data_args: Dict[str, Any],
    figure_args: Dict[str, Any],
    block: bool = False,
    inline: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: str = "png",
) -> Union["PlotHandle", PlotImage, None]:
    """Fetch a view of `kind` from EMode and show it: in the tab, or as a static image."""
    options = PlotOptions.from_arguments(figure_args)
    if inline not in (None, "png"):
        raise EModeError("inline is 'png' or None")
    check_plot_support(em, kind)
    sent = _sent(kind, data_args)
    data = fetch_view(em, sent, DEFAULT_PIXELS, kind)
    local = {k: v for k, v in data_args.items() if k not in sent}
    if local:
        data = _select(data, local)
    if isinstance(data, GDSPanel):
        # the view keeps the panel's resolved state, which a fetch again reapplies
        data_args = state_arguments(data_args, data)
    return present(em, data, data_args, options, block, inline, file_name, file_type)


def present(
    em: "EMode",
    data: Any,
    arguments: Dict[str, Any],
    options: PlotOptions,
    block: bool = False,
    inline: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: str = "png",
) -> Union["PlotHandle", PlotImage, None]:
    """Show `data` (a view's data the client holds): register it with the tab, or draw it here."""
    inline = inline or os.environ.get("EMODE_PLOT_INLINE") or None
    if headless_requested(file_name) or inline == "png":
        notebook = em.in_ipython()
        image = headless_image(data, options, file_name, file_type, inline=notebook)
        if notebook:
            from IPython.display import display  # type: ignore

            display(image)
            return None
        return image

    view = em._views.add(data, arguments, options, DEFAULT_PIXELS)
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


def plot_S_matrix(
    em: "EMode",
    block: bool = False,
    inline: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: str = "png",
    **arguments: Any,
) -> Union["PlotHandle", PlotImage, None]:
    """``EMode.plot_S_matrix``: the S-matrix's block from `input` to `output`, as a view."""
    data_args, figure_args = _split(arguments, ("input", "output", "plot_function"), "plot_S_matrix")
    return show_call(em, "smatrix", data_args, figure_args, block, inline, file_name, file_type)


def material_explorer(
    em: "EMode",
    block: bool = False,
    inline: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: str = "png",
    **arguments: Any,
) -> Union["PlotHandle", PlotImage, None]:
    """``EMode.material_explorer``: a material's refractive index against wavelength, as a view."""
    data_args, figure_args = _split(
        arguments, ("material", "wavelengths", "step", "x", "component"), "material_explorer"
    )
    return show_call(em, "material", data_args, figure_args, block, inline, file_name, file_type)


def _split(arguments: Dict[str, Any], data: Sequence[str], function: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """`arguments` split into the view's data arguments and the figure options, refusing the rest."""
    from ..types import ArgumentError

    allowed = set(PlotOptions.__dataclass_fields__)
    for name in arguments:
        if name not in allowed and name not in data:
            raise ArgumentError(
                f"{function}() takes no argument {name!r}. It takes "
                f"{', '.join(sorted(set(data) | allowed))}.",
                function,
                name,
            )
    data_args = {k: v for k, v in arguments.items() if k in data and v is not None}
    figure_args = {k: v for k, v in arguments.items() if k in allowed}
    return data_args, figure_args


def show_data(
    value: Any,
    /,
    em: Optional["EMode"] = None,
    block: bool = False,
    inline: Optional[str] = None,
    file_name: Optional[str] = None,
    file_type: str = "png",
    **arguments: Any,
) -> Union["PlotHandle", PlotImage, None]:
    """``FarField.show()``, ``SMatrix.show()`` and a sweep's ``plot()``: data this client holds, shown with no EMode call.

    `em` is the session whose tab shows it, by default the newest session of
    this process that is still running.
    """
    from ..free_space import FarField
    from ..smatrix import SMatrix
    from .sweep_figure import SWEEP_CONTROLS, is_sweep_shaped, sweep_panel

    from ..figures import Figure

    em = em or current_session()
    if isinstance(value, Figure):
        data_args, figure_args = _split(arguments, (), "Figure.plot")
        view_data: Any = value
    elif is_sweep_shaped(value):
        data_args, figure_args = _split(arguments, SWEEP_CONTROLS, "plot")
        view_data = sweep_panel(value, **data_args)
    elif isinstance(value, FarField):
        data_args, figure_args = _split(arguments, ("data",), "FarField.show")
        view_data = _select(FarFieldPanel(value), data_args)
    elif isinstance(value, SMatrix):
        data_args, figure_args = _split(arguments, ("input", "output", "plot_function"), "SMatrix.show")
        view_data = _select(smatrix_view(value), data_args)
    else:
        raise TypeError(
            "show() takes a FarField, an SMatrix, a sweep's result, an OptimizationResult or a "
            f"PulseResponse, not {type(value).__name__}"
        )
    options = PlotOptions.from_arguments(figure_args)
    if em is None:
        inline = inline or os.environ.get("EMODE_PLOT_INLINE") or None
        if headless_requested(file_name) or inline == "png":
            return headless_image(view_data, options, file_name, file_type, inline=False)
        raise EModeError(
            "show() opens the plot in an EMode session's browser tab. Start one with "
            "em = emode.EMode(), or pass em=, file_name= or inline='png'."
        )
    return present(em, view_data, data_args, options, block, inline, file_name, file_type)


def smatrix_view(smatrix: Any) -> SMatrixView:
    """An ``SMatrix`` the user holds as the S-matrix plot's view, with the default selection."""
    names = list(smatrix.port_names or [str(i) for i in range(smatrix.num_ports)])
    wavelengths = list(smatrix.wavelengths or [])
    several = len(set(wavelengths)) > 1
    labels = [
        f"{n}@{w * 1e9:.0f}nm" if several and w < 1e-3 else (f"{n}@{w:g}nm" if several else n)
        for n, w in zip(names, wavelengths or [0.0] * len(names))
    ]
    modes = [[str(k) for k in range(size)] for size in smatrix.port_sizes]
    menu = MenuState(
        ports=labels,
        components=["abs^2", "angle", "real", "imag"],
        selected={"input": labels[0], "output": labels[-1], "component": "abs^2"},
    )
    return SMatrixView(smatrix=smatrix, port_labels=labels, mode_labels=modes, menu=menu)


#: The sessions of this process, newest last.
_SESSIONS: "List[weakref.ref]" = []


def register_session(em: "EMode") -> None:
    _SESSIONS.append(weakref.ref(em))


def current_session() -> Optional["EMode"]:
    """The newest session of this process that is still running, if any."""
    for ref in reversed(_SESSIONS):
        em = ref()
        if em is not None and em.__dict__.get("running"):
            return em
    return None


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
