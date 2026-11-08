"""The MCP server: tools that drive an EMode session through the public client.

One server holds at most one open session. The tools only call the public
``EMode`` methods, so they stay in step with what a script can do.
``WRAPPED_METHODS`` lists them.

MCP's stdio transport carries its messages on this process's stdout, so the
solver is always started quiet: anything it printed there would corrupt the
protocol.

Every tool but ``emode_plot`` returns a ``ToolResult``. Its ``warnings`` hold
what the client library logged at WARNING level during the call. A warning the
solver itself writes goes to the solver's own stderr, which the client does not
return, so it does not reach the agent.
"""

from __future__ import annotations

import functools
import inspect
import logging
import tempfile
import traceback
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Literal, TypeVar

import anyio.to_thread
import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field

from ..emode import EMode
from ..optimize import Constraint, Objective, Parameter
from ..plot import PlotImage
from ..types import EModeError
from .convert import FileStore, to_jsonable
from .docs import DocsError, read_page
from .specs import ConstraintSpec, ObjectiveSpec, ParameterSpec, SettingsSpec, ShapeSpec

T = TypeVar("T")

SKILL_PATH = Path(__file__).with_name("SKILL.md")

# A sweep or an optimization longer than this asks for confirmation first.
CONFIRM_ABOVE_POINTS = 20

# The public client methods the tools call.
WRAPPED_METHODS = frozenset(
    {
        "settings",
        "shape",
        "add_material",
        "refractive_index",
        "label_profile",
        "FDM",
        "FEM",
        "straight_section",
        "taper_section",
        "bend_section",
        "gds_section",
        "pwd_section",
        "EME",
        "sweep",
        "converge",
        "evaluate",
        "optimize",
        "get",
        "inspect",
        "plot",
        "close",
    }
)

INSTRUCTIONS = (
    "EMode is a photonic waveguide mode solver and EME simulator. Lengths are in nm. "
    "Call emode_open to start a session and draw the cross-section (bottom shape first), "
    "emode_solve for the modes' effective indices and losses, "
    "emode_converge before trusting a result, "
    "and emode_close when finished so the license is released. "
    "Sweeps over more than 20 points and optimizations ask for confirmation first: "
    "repeat the call with confirm=true to run them. "
    "Large arrays are saved to .npy files and returned as a file path."
)

_SECTION_FIELDS: dict[str, tuple[str, frozenset[str]]] = {
    "straight": (
        "straight_section",
        frozenset({"name", "profile", "length", "offset"}),
    ),
    "taper": ("taper_section", frozenset({"name", "profile", "profile_end", "length"})),
    "bend": (
        "bend_section",
        frozenset(
            {"name", "profile", "radius", "sweep_angle", "euler", "curvature", "length"}
        ),
    ),
    "gds": (
        "gds_section",
        frozenset(
            {
                "name",
                "profile",
                "gds_path",
                "layer_map",
                "length",
                "cell_index",
                "scale",
            }
        ),
    ),
    "pwd": (
        "pwd_section",
        frozenset(
            {
                "name",
                "profile",
                "length",
                "material",
                "window",
                "offset",
                "max_order",
                "max_kt",
                "num_modes",
            }
        ),
    ),
}


class ToolResult(BaseModel):
    """What most tools return."""

    status: Literal["done", "confirmation_required"] = Field(
        "done",
        description="'confirmation_required' means nothing ran. Repeat the call with confirm=true.",
    )
    data: Any = Field(
        None,
        description="The result as JSON. Large arrays are {file, shape, dtype} references.",
    )
    message: str | None = Field(None, description="Why confirmation is required.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings the client library logged during the call.",
    )
    files: list[str] = Field(
        default_factory=list, description="Files written for this result."
    )


class OpenResult(BaseModel):
    simulation_name: str = Field(description="Name of the session's simulation.")
    shapes: list[str] = Field(description="Names of the shapes drawn, in order.")


class ModeResult(BaseModel):
    mode: int = Field(description="Mode index, 0 is the fundamental mode.")
    n_eff: float = Field(description="Real part of the effective index.")
    n_eff_imag: float = Field(description="Imaginary part of the effective index.")
    loss_dB_per_m: float | None = Field(
        description="Propagation loss in dB/m, if the solver reports one."
    )


class SolveResult(BaseModel):
    modes: list[ModeResult]
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings the client library logged during the call.",
    )


class CloseResult(BaseModel):
    closed: bool = Field(description="False if no session was open.")


def _set_kwargs(model: BaseModel | None) -> dict[str, Any]:
    return {} if model is None else model.model_dump(exclude_none=True)


def _mode_results(n_eff: Any, loss: Any) -> list[ModeResult]:
    index = np.atleast_1d(np.asarray(n_eff)).ravel()
    losses = (
        None if loss is None else np.atleast_1d(np.asarray(loss, dtype=float)).ravel()
    )
    return [
        ModeResult(
            mode=i,
            n_eff=float(value.real),
            n_eff_imag=float(value.imag),
            loss_dB_per_m=float(losses[i])
            if losses is not None and i < len(losses)
            else None,
        )
        for i, value in enumerate(index)
    ]


class _ListHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record.getMessage())


@contextmanager
def _capture_warnings() -> Iterator[list[str]]:
    """Collect what the client library logs at WARNING level while a tool runs."""
    messages: list[str] = []
    logger = logging.getLogger("emode")
    handler = _ListHandler(messages)
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)


class _Session:
    """The server's single open EMode session, if any."""

    def __init__(self, client_factory: Callable[..., Any], store: FileStore) -> None:
        self.client_factory = client_factory
        self.store = store
        self.em: Any = None
        # the cross-sections labeled in this session, in the order labeled
        self.labels: list[str] = []

    def require(self) -> Any:
        if self.em is None:
            raise ToolError("No EMode session is open. Call emode_open first.")
        return self.em

    def shut(self, *, save: bool = False) -> bool:
        em, self.em = self.em, None
        self.labels = []
        if em is None:
            return False
        em.close(save=save)
        return True


def _describe(err: BaseException) -> str:
    """The text an agent sees for a failed tool call: the error's type, context and message.

    An ``EModeError`` also names the fields it carries, such as the function and
    argument of an ``ArgumentError``. An error with an empty message says where
    it was raised, so the text is never blank.
    """
    name = type(err).__name__
    if isinstance(err, EModeError):
        message = str(err.msg or "")
        fields = {
            k: v
            for k, v in err.to_dict().items()
            if k not in ("__data_type__", "msg") and v is not None
        }
        if fields:
            name += " (" + ", ".join(f"{k} {v!r}" for k, v in fields.items()) + ")"
    else:
        message = str(err)
    if message.strip():
        return f"{name}: {message}"
    frames = traceback.extract_tb(err.__traceback__)
    where = (
        f", raised in {frames[-1].name} ({Path(frames[-1].filename).name}:{frames[-1].lineno})"
        if frames
        else ""
    )
    return f"{name} with no message{where}"


async def _blocking(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking client call off the event loop, turning solver errors into tool errors."""
    try:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
    except EModeError as err:
        raise ToolError(_describe(err)) from err


def _takes_a_list(key: str) -> bool:
    """Whether ``optimize`` varies the key as a list: 'poling, <section>, domain_lengths'."""
    parts = [p.strip() for p in key.split(",")]
    return len(parts) == 3 and parts[0] == "poling" and parts[2] == "domain_lengths"


def _parameter(spec: ParameterSpec) -> Parameter:
    """The ``Parameter`` a spec names, a vector one when its x0 or bounds is a list.

    A list x0 sets the vector's size, or per-entry bounds do when x0 is omitted.
    A one-entry list for a key that takes one value is that value.
    """
    key, x0, bounds = spec.key, spec.x0, spec.bounds
    if not _takes_a_list(key):
        if isinstance(x0, list) and len(x0) == 1:
            x0 = x0[0]
        if isinstance(bounds, list) and len(bounds) == 1:
            bounds = bounds[0]
    if isinstance(x0, list):
        size = len(x0)
    elif isinstance(bounds, list):
        size = len(bounds)
    else:
        return Parameter(key, **_drop_none(x0=x0, bounds=bounds))
    if not _takes_a_list(key):
        raise ToolError(
            f"'{key}' takes one value, so its x0 and bounds cannot be lists. "
            "optimize varies a list only for 'poling, <section>, domain_lengths'. "
            "To vary each entry of a mask on its own, such as the two output arms of "
            "an MMI, draw each arm as its own shape and vary its scalar keys, such as "
            "'shape, <profile>, <arm>, width' and 'shape, <profile>, <arm>, position'."
        )
    if isinstance(bounds, list) and len(bounds) != size:
        raise ToolError(
            f"'{key}' has {len(bounds)} [min, max] pairs in bounds for the {size} entries "
            "of x0. Give one pair per entry, or one [min, max] for every entry."
        )
    return Parameter(key, **_drop_none(x0=x0, bounds=bounds), shape=(size,))


def _drop_none(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def create_server(
    client_factory: Callable[..., Any] = EMode,
    output_dir: Path | str | None = None,
    docs_dir: Path | str | None = None,
    docs_fetch: Callable[[str], str] | None = None,
) -> MCPServer:
    """Build the MCP server.

    ``client_factory`` makes the EMode client for a session. It is ``EMode``
    itself, and tests pass a stand-in. Large results and plots are written to
    ``output_dir``, by default a new temporary directory, and its array files
    are deleted when the server shuts down. Docs pages are read from
    ``docs_dir`` (default: the ``EMODE_DOCS_DIR`` environment variable) and
    otherwise fetched with ``docs_fetch`` (default: HTTP GET) from the docs site.
    """
    if output_dir is None:
        store = FileStore(lambda: Path(tempfile.mkdtemp(prefix="emode-mcp-")))
    else:
        store = FileStore(Path(output_dir))
    session = _Session(client_factory, store)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # A client that disconnects without emode_close must not hold the license.
            if session.em is not None:
                await _blocking(session.shut)
            store.cleanup()

    server = MCPServer("emode", instructions=INSTRUCTIONS, lifespan=lifespan)

    def tool(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            description = inspect.cleandoc(fn.__doc__ or "")

            # MCP shows the agent only "Error executing tool <name>" for an
            # exception other than ToolError, so every failure becomes one
            # carrying the error's text.
            @functools.wraps(fn)
            async def run(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await fn(*args, **kwargs)
                except (ToolError, ResourceError, MCPError):
                    raise
                except Exception as err:
                    raise ToolError(_describe(err)) from err

            return server.tool(name=name, description=description)(run)

        return register

    def done(data: Any, warnings: list[str], written_before: int) -> ToolResult:
        return ToolResult(
            data=to_jsonable(data, store),
            warnings=warnings,
            files=store.written[written_before:],
        )

    def needs_confirmation(message: str) -> ToolResult:
        return ToolResult(status="confirmation_required", message=message)

    @tool("emode_open")
    async def emode_open(
        simulation_name: str = "emode",
        license_type: Literal["2d", "3d", "default"] = "default",
        save_path: str | None = None,
        settings: SettingsSpec | None = None,
        shapes: list[ShapeSpec] | None = None,
    ) -> OpenResult:
        """Start an EMode session and draw a waveguide cross-section.

        Opens the solver, applies the settings, then draws the shapes in order
        (the first is the bottom layer). Only one session can be open, so call
        emode_close before opening another. The session is saved to save_path
        when it is closed, and not saved when save_path is omitted.
        """
        if session.em is not None:
            raise ToolError(
                "A session is already open. Call emode_close before emode_open."
            )
        options: dict[str, Any] = {
            "simulation_name": simulation_name,
            "license_type": license_type,
            "quiet": True,
            "save": save_path is not None,
        }
        if save_path is not None:
            options["save_path"] = save_path
        # The client installs signal handlers, which only works on the main thread.
        try:
            em = session.client_factory(**options)
        except EModeError as err:
            raise ToolError(str(err.msg or err)) from err
        session.em = em
        try:
            if settings is not None:
                await _blocking(em.settings, **_set_kwargs(settings))
            for shape in shapes or []:
                await _blocking(em.shape, **_set_kwargs(shape))
        except BaseException:
            # Do not leave a half-built session holding the license.
            await _blocking(session.shut)
            raise
        return OpenResult(
            simulation_name=simulation_name, shapes=[s.name for s in shapes or []]
        )

    @tool("emode_settings")
    async def emode_settings(
        settings: SettingsSpec, profile: str | None = None
    ) -> ToolResult:
        """Change the solver settings (wavelength, resolution, window, modes) of a cross-section.

        Without profile, the settings change the current cross-section only.
        A cross-section labeled with emode_profile keeps the settings it was
        labeled with, so set what every cross-section shares before labeling
        them. profile names a labeled cross-section to change instead, or is
        'all' for the current one and every labeled one, such as a coarser
        grid for a search and the converged grid again for the check.
        """
        em = session.require()
        applied = _set_kwargs(settings)
        if profile is None:
            targets: list[str | None] = [None]
        elif profile == "all":
            targets = [None, *session.labels]
        else:
            targets = [profile]
        with _capture_warnings() as warnings:
            for target in targets:
                await _blocking(em.settings, **applied, **_drop_none(profile=target))
        if profile is None:
            return done(applied, warnings, len(store.written))
        changed = [t or "_default" for t in targets]
        return done({**applied, "profiles": changed}, warnings, len(store.written))

    @tool("emode_shape")
    async def emode_shape(shape: ShapeSpec, profile: str | None = None) -> ToolResult:
        """Draw one more shape in the open session's cross-section, or change one.

        A new shape goes on top of the earlier ones. A shape with the name of
        an existing one changes it. profile names a cross-section labeled with
        emode_profile, to change that one instead of the current cross-section.
        """
        em = session.require()
        with _capture_warnings() as warnings:
            await _blocking(
                em.shape, **_set_kwargs(shape), **_drop_none(profile=profile)
            )
        return done({"name": shape.name}, warnings, len(store.written))

    @tool("emode_profile")
    async def emode_profile(name: str) -> ToolResult:
        """Label the cross-section drawn so far, so EME sections can be built on it.

        A device with several cross-sections (an input waveguide, a taper's
        wide end, a multimode body) labels each one, then changes the shapes
        with emode_shape for the next. Sections name a labeled cross-section
        as their profile. The labeled copy keeps its shapes when the current
        one changes. Change it later with emode_shape and profile, or in an
        optimization with the key 'shape, <profile>, <name>, <property>'.
        """
        em = session.require()
        with _capture_warnings() as warnings:
            await _blocking(em.label_profile, name=name)
        if name not in session.labels:
            session.labels.append(name)
        return done({"profile": name}, warnings, len(store.written))

    @tool("emode_material")
    async def emode_material(
        action: Literal["add", "index"],
        name: str,
        refractive_index_equation: str | None = None,
        wavelength_unit: Literal["nm", "um"] | None = None,
        wavelength_range: list[float] | None = None,
        loss: float | None = None,
        wavelength: float = 1550,
    ) -> ToolResult:
        """Add a material, or look up a material's refractive index.

        action 'add' defines name from refractive_index_equation, a function of
        the wavelength x (in wavelength_unit, default um), such as
        '1.45+0.001*x'. wavelength_range is in um and loss in dB/m.
        action 'index' returns the index of the named material at wavelength (nm).
        """
        em = session.require()
        with _capture_warnings() as warnings:
            if action == "add":
                if not refractive_index_equation:
                    raise ToolError("action 'add' needs refractive_index_equation.")
                kwargs = _drop_none(
                    name=name,
                    refractive_index_equation=refractive_index_equation,
                    wavelength_unit=wavelength_unit,
                    wavelength_range=wavelength_range,
                    loss=loss,
                )
                await _blocking(em.add_material, **kwargs)
                data: Any = {"added": name}
            else:
                data = await _blocking(
                    em.refractive_index, material=name, wavelength=wavelength
                )
        return done(data, warnings, len(store.written))

    @tool("emode_solve")
    async def emode_solve(
        wavelength: float | None = None,
        num_modes: int | None = None,
        solver: Literal["FDM", "FEM"] = "FDM",
    ) -> SolveResult:
        """Solve the open session's cross-section for its modes.

        Returns each mode's effective index (real and imaginary parts) and
        propagation loss. wavelength (nm) and num_modes override the session's
        settings for this and later solves. solver FEM needs an EMode3D license.
        """
        em = session.require()
        overrides = _drop_none(wavelength=wavelength, num_modes=num_modes)
        with _capture_warnings() as warnings:
            if overrides:
                await _blocking(em.settings, **overrides)
            await _blocking(getattr(em, solver), verbose=False)
            n_eff = await _blocking(em.get, "effective_index")
            try:
                loss = await _blocking(em.get, "loss_dB_per_m")
            except ToolError:
                loss = None
        return SolveResult(modes=_mode_results(n_eff, loss), warnings=warnings)

    @tool("emode_section")
    async def emode_section(
        kind: Literal["straight", "taper", "bend", "gds", "pwd"],
        name: str,
        profile: str | None = None,
        profile_end: str | None = None,
        length: float | None = None,
        offset: float | list[float] | None = None,
        radius: float | list[float] | None = None,
        sweep_angle: float | None = None,
        euler: dict[str, float] | None = None,
        curvature: list[float] | None = None,
        gds_path: str | None = None,
        layer_map: dict[str, int] | None = None,
        cell_index: int | str | None = None,
        scale: float | None = None,
        material: str | None = None,
        window: list[float] | None = None,
        max_order: list[int] | None = None,
        max_kt: float | None = None,
        num_modes: int | None = None,
    ) -> ToolResult:
        """Declare a section of an EME circuit. Sections connect in the order declared.

        kind straight: profile, length, offset. taper: profile, profile_end,
        length. bend: profile plus one curve. radius (nm) with sweep_angle
        (radians) is a circular arc. A list of radii with sweep_angle is a
        clothoid whose curvature varies linearly from the first radius to the
        last ("inf" is a straight end). euler is {"angle": radians,
        "min_radius": nm} for a symmetric Euler bend. curvature is a list of
        signed curvatures (1/nm) equally spaced along length (nm). gds:
        gds_path, layer_map, profile, length, cell_index, scale. pwd (plane
        waves): length, material, window, offset, max_order, max_kt, num_modes.
        Lengths are in nm. A field the kind does not take is refused.
        """
        em = session.require()
        method, allowed = _SECTION_FIELDS[kind]
        given = _drop_none(
            name=name,
            profile=profile,
            profile_end=profile_end,
            length=length,
            offset=offset,
            radius=radius,
            sweep_angle=sweep_angle,
            euler=euler,
            curvature=curvature,
            gds_path=gds_path,
            layer_map=layer_map,
            cell_index=cell_index,
            scale=scale,
            material=material,
            window=window,
            max_order=max_order,
            max_kt=max_kt,
            num_modes=num_modes,
        )
        extra = sorted(set(given) - allowed)
        if extra:
            raise ToolError(
                f"A {kind} section ({method}) does not take {', '.join(extra)}."
            )
        with _capture_warnings() as warnings:
            await _blocking(getattr(em, method), **given)
        return done({"section": name, "kind": kind}, warnings, len(store.written))

    @tool("emode_eme")
    async def emode_eme(warm_start: bool = False) -> ToolResult:
        """Build and solve the declared sections as an EME circuit and return its S-matrix.

        The S-matrix comes back as port_sizes, port_names, wavelengths and
        its complex data as real and imaginary parts (a file when large).
        warm_start reuses the previous refinement of tapers.
        """
        em = session.require()
        before = len(store.written)
        with _capture_warnings() as warnings:
            await _blocking(em.EME, **({"warm_start": True} if warm_start else {}))
            s_matrix = await _blocking(em.get, "S_matrix")
        return done(s_matrix, warnings, before)

    @tool("emode_sweep")
    async def emode_sweep(
        key: str,
        values: list[float | str] | None = None,
        start: float | None = None,
        stop: float | None = None,
        num: int | None = None,
        solve_type: Literal["FDM", "mesh", "EME"] | None = None,
        result: str | list[str] | None = None,
        method: Literal["direct", "interpolated"] | None = None,
        num_nodes: int | None = None,
        wavelength_range: list[float] | None = None,
        adaptive: bool | None = None,
        tolerance: float | None = None,
        confirm: bool = False,
    ) -> ToolResult:
        """Sweep a setting, shape or section property and collect results at each value.

        key is a setting name such as 'wavelength', 'shape, core, width',
        'shape, <profile>, <name>, <property>' (a shape of a labeled profile,
        swept through EME) or 'section, taper, length'. Give values, or start, stop and num for
        evenly spaced ones. result is a get() key such as 'effective_index'
        (default), or 'S_matrix' with solve_type 'EME'. method 'interpolated'
        solves num_nodes wavelengths and interpolates a wavelength sweep of the
        S-matrix. A sweep over more than 20 points (more than 20 num_nodes for
        an interpolated one) asks for confirmation first: repeat with confirm=true.
        """
        em = session.require()
        range_given = any(v is not None for v in (start, stop, num))
        if (values is None) == (not range_given):
            raise ToolError(
                "Give either values, or start, stop and num, not both and not neither."
            )
        if values is None:
            if start is None or stop is None or num is None:
                raise ToolError("A range needs start, stop and num.")
            values = np.linspace(start, stop, num).tolist()
        points = num_nodes if method == "interpolated" and num_nodes else len(values)
        if points > CONFIRM_ABOVE_POINTS and not confirm:
            return needs_confirmation(
                f"This sweep solves {points} points, more than {CONFIRM_ABOVE_POINTS}, and can take a long time. "
                "Repeat the call with confirm=true to run it."
            )
        kwargs = _drop_none(
            key=key,
            values=values,
            solve_type=solve_type,
            result=result,
            method=method,
            num_nodes=num_nodes,
            wavelength_range=wavelength_range,
            adaptive=adaptive,
            tolerance=tolerance,
        )
        before = len(store.written)
        with _capture_warnings() as warnings:
            data = await _blocking(em.sweep, **kwargs)
        return done(data, warnings, before)

    @tool("emode_converge")
    async def emode_converge(
        setting: Literal["resolution", "window", "pml", "modes", "all"] | None = None,
        target: Literal["effective_index", "group_index", "loss", "orthogonality"]
        | None = None,
        tolerance: float | None = None,
        apply: bool | None = None,
        max_solves: int | None = None,
        max_time: float | None = None,
        max_memory: float | None = None,
        mode: int | None = None,
        profile: str | None = None,
    ) -> ToolResult:
        """Find the resolution, window, PML thickness or mode count a result needs.

        Varies the setting until the target of the watched mode stops changing,
        and returns the convergence report (status, chosen settings, one row
        per solve, a message saying why it stopped if it did not converge, and
        the report as a text table). Run this before trusting a solve.
        apply=true writes the chosen settings to the cross-section converged:
        the current one, or the labeled one profile names. Copy them to the
        others with emode_settings and profile 'all'. max_time (s) and
        max_memory (GB, default three quarters of the machine's memory) are
        budgets: a step predicted to pass either is not started, and the run
        stops with status 'budget' at the finest settings it solved.
        """
        em = session.require()
        kwargs = _drop_none(
            setting=setting, target=target, tolerance=tolerance, apply=apply,
            max_solves=max_solves, max_time=max_time, max_memory=max_memory, mode=mode,
            profile=profile,
        )  # fmt: skip
        with _capture_warnings() as warnings:
            report = await _blocking(em.converge, **kwargs)
        data = report.to_json()
        data["table"] = str(report)
        return done(data, warnings, len(store.written))

    @tool("emode_evaluate")
    async def emode_evaluate(
        parameters: dict[str, float | list[float]],
        results: str | list[str] = "effective_index",
    ) -> ToolResult:
        """Evaluate the simulation at one point and return the requested results.

        parameters maps keys ('wavelength', 'shape, core, width', 'shape,
        <profile>, <name>, <property>', 'section, coupler, length') to values. Every evaluation starts from the same
        snapshot, so the same parameters give the same results.
        """
        em = session.require()
        before = len(store.written)
        with _capture_warnings() as warnings:
            data = await _blocking(em.evaluate, parameters=parameters, results=results)
        return done(data, warnings, before)

    @tool("emode_optimize")
    async def emode_optimize(
        parameters: list[ParameterSpec],
        objective: list[ObjectiveSpec],
        constraints: list[ConstraintSpec] | None = None,
        method: Literal[
            "cobyqa",
            "nelder-mead",
            "powell",
            "differential_evolution",
            "l-bfgs-b",
            "slsqp",
        ]
        | None = None,
        max_evaluations: int | None = None,
        max_time: float | None = None,
        tolerance: float | None = None,
        confirm: bool = False,
    ) -> ToolResult:
        """Find the parameter values that give the best objective, within bounds and constraints.

        An optimization makes many solves and always asks for confirmation
        first: repeat the call with confirm=true to run it. Several objectives
        form a weighted sum. The session is left solved at the best point.

        Each parameter takes one value, except 'poling, <section>,
        domain_lengths', which takes a list x0 with one [min, max] pair per
        entry in bounds (or one [min, max] for every entry). A mask with several
        openings cannot be varied entry by entry: to vary the two output arms
        of an MMI, draw each arm as its own shape and vary its width and
        position keys.
        """
        em = session.require()
        budget = max_evaluations if max_evaluations is not None else 100
        if not confirm:
            return needs_confirmation(
                f"This optimization makes up to {budget} evaluations and can take a long time. "
                "Repeat the call with confirm=true to run it."
            )

        def objective_of(spec: ObjectiveSpec) -> Objective:
            fields = spec.model_dump(exclude_none=True)
            return Objective(fields.pop("kind"), **fields)

        kwargs: dict[str, Any] = {
            "parameters": [_parameter(p) for p in parameters],
            "objective": [objective_of(o) for o in objective],
        }
        if constraints:
            kwargs["constraints"] = [
                Constraint(
                    objective_of(c.quantity)
                    if isinstance(c.quantity, ObjectiveSpec)
                    else c.quantity,
                    c.op,
                    c.value,
                )
                for c in constraints
            ]
        kwargs.update(
            _drop_none(
                method=method,
                max_evaluations=max_evaluations,
                max_time=max_time,
                tolerance=tolerance,
            )
        )
        before = len(store.written)
        with _capture_warnings() as warnings:
            report = await _blocking(em.optimize, **kwargs)
        data = report.to_json()
        data["history"] = [
            {
                "index": e["index"],
                "objective": e["objective"],
                "parameters": e["parameters"],
                "error": e["error"],
            }
            for e in data["history"]
        ]
        return done(data, warnings, before)

    @tool("emode_get")
    async def emode_get(
        key: str | None = None, profile: str | None = None
    ) -> ToolResult:
        """Read a result or setting by key, such as 'effective_index', 'loss_dB_per_m' or 'S_matrix'.

        Without a key, returns the names that can be read. Large arrays (fields)
        are saved to .npy files and returned as {file, shape, dtype}.
        """
        em = session.require()
        before = len(store.written)
        with _capture_warnings() as warnings:
            if key is None:
                data = await _blocking(em.inspect)
            else:
                extra = _drop_none(profile=profile)
                data = await _blocking(em.get, key, **extra)
        return done(data, warnings, before)

    @tool("emode_plot")
    async def emode_plot(
        component: str | None = None,
        mode: int = 0,
        port: int | str = 0,
        plane: str | None = None,
        wavelength: float | None = None,
        profile: str | None = None,
        slice_position: float | None = None,
    ) -> list[Image | str]:
        """Plot the solved fields or circuit and return the figure as a PNG image.

        component names the field, such as 'Ex' or 'Hy'. mode picks a mode, port
        a circuit port and plane the cut plane of a circuit plot (such as 'z-x').
        """
        em = session.require()
        path = store.new_path("plot", ".png")
        kwargs = _drop_none(
            component=component, mode=mode, port=port, plane=plane, wavelength=wavelength,
            profile=profile, slice_position=slice_position,
        )  # fmt: skip
        figure = await _blocking(
            em.plot, file_name=str(path), file_type="png", **kwargs
        )
        if isinstance(figure, PlotImage) and figure.png is not None:
            png = figure.png.tobytes()
        elif path.exists():
            png = path.read_bytes()
        else:
            raise ToolError("The solver returned no image for this plot.")
        store.written.append(str(path))
        if not path.exists():
            path.write_bytes(png)
        return [Image(data=png, format="png"), f"Saved to {path}"]

    @tool("emode_close")
    async def emode_close(save: bool = False) -> CloseResult:
        """Close the session and release the license. Harmless when no session is open."""
        return CloseResult(closed=await _blocking(session.shut, save=save))

    @server.resource(
        "emode://skill",
        name="skill",
        title="How to use EMode well",
        description="Guidance for an assistant: converge first, sweep with interpolation, check passivity.",
        mime_type="text/markdown",
    )
    def skill() -> str:
        return SKILL_PATH.read_text()

    @server.resource(
        "emode://docs/{+page}",
        name="docs",
        title="EMode documentation page",
        description="A docs page as Markdown, such as examples/Ring_Design or emodeguide/setup.",
        mime_type="text/markdown",
    )
    def docs_page(page: str) -> str:
        try:
            return read_page(page, docs_dir, docs_fetch)
        except DocsError as err:
            raise ResourceError(str(err)) from err

    return server
