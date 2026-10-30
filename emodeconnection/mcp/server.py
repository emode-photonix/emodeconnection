"""The MCP server: tools that drive an EMode session through the public client.

One server holds at most one open session. The tools only call the public
``EMode`` methods (``settings``, ``shape``, ``FDM``, ``get``, ``close``), so
they stay in step with what a script can do.

MCP's stdio transport carries its messages on this process's stdout, so the
solver is always started quiet: anything it printed there would corrupt the
protocol.
"""

from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal, TypeVar

import anyio.to_thread
import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from ..emodeconnection import EMode
from ..types import EModeError

T = TypeVar("T")

INSTRUCTIONS = (
    "EMode is a photonic waveguide mode solver. Lengths are in nm. "
    "Call emode_open to start a session and draw the cross-section (bottom shape first), "
    "emode_solve to get the modes' effective indices and losses, "
    "and emode_close when finished so the license is released."
)


class SettingsSpec(BaseModel):
    """Solver settings, passed to ``EMode.settings``. Unset fields keep EMode's defaults."""

    model_config = ConfigDict(extra="forbid")

    wavelength: float | None = Field(None, description="Wavelength in nm.")
    num_modes: int | None = Field(None, description="Number of modes to solve for.")
    x_resolution: float | None = Field(
        None, description="Grid cell size along x in nm."
    )
    y_resolution: float | None = Field(
        None, description="Grid cell size along y in nm."
    )
    window_width: float | None = Field(
        None, description="Simulation window width in nm."
    )
    window_height: float | None = Field(
        None, description="Simulation window height in nm."
    )
    boundary_condition: str | None = Field(
        None, description="'0' for none, or 'S'/'A' symmetry pairs, or 'TE'/'TM'."
    )
    field_to_solve: Literal["Et", "Ht"] | None = Field(
        None, description="Field to solve for."
    )
    max_effective_index: float | None = Field(
        None,
        description="Guess for the largest effective index. 0 uses the largest material index.",
    )
    tolerance: float | None = Field(None, description="Eigensolver tolerance.")
    bend_radius: float | None = Field(
        None, description="Bend radius in nm. 0 means straight."
    )


class ShapeSpec(BaseModel):
    """One layer or ridge of the cross-section, passed to ``EMode.shape``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Shape name, unique within the cross-section.")
    material: str = Field(
        description="A material name from EMode's database, such as 'Si' or 'SiO2'."
    )
    width: float | None = Field(
        None, description="Width in nm. Omit for a layer spanning the window."
    )
    height: float = Field(description="Height (thickness) in nm.")
    position: float | list[float | Literal["auto"]] | None = Field(
        None,
        description="Horizontal position, or [x, y] with 'auto' stacking on the layer below.",
    )
    etch_depth: float | None = Field(None, description="Etch depth in nm, for a rib.")
    sidewall_angle: float | None = Field(None, description="Sidewall angle in degrees.")


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


class _Session:
    """The server's single open EMode session, if any."""

    def __init__(self, client_factory: Callable[..., Any]) -> None:
        self.client_factory = client_factory
        self.em: Any = None

    def require(self) -> Any:
        if self.em is None:
            raise ToolError("No EMode session is open. Call emode_open first.")
        return self.em

    def shut(self, *, save: bool = False) -> bool:
        em, self.em = self.em, None
        if em is None:
            return False
        em.close(save=save)
        return True


async def _blocking(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking client call off the event loop, turning solver errors into tool errors."""
    try:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
    except EModeError as err:
        raise ToolError(str(err.msg or err)) from err


def create_server(client_factory: Callable[..., Any] = EMode) -> MCPServer:
    """Build the MCP server.

    ``client_factory`` makes the EMode client for a session. It is ``EMode``
    itself, and tests pass a stand-in.
    """
    session = _Session(client_factory)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # A client that disconnects without emode_close must not hold the license.
            if session.em is not None:
                await _blocking(session.shut)

    server = MCPServer("emode", instructions=INSTRUCTIONS, lifespan=lifespan)

    @server.tool(name="emode_open")
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

    @server.tool(name="emode_solve")
    async def emode_solve(
        wavelength: float | None = None, num_modes: int | None = None
    ) -> SolveResult:
        """Solve the open session's cross-section for its modes.

        Returns each mode's effective index (real and imaginary parts) and
        propagation loss. wavelength (nm) and num_modes override the session's
        settings for this and later solves.
        """
        em = session.require()
        overrides = {
            key: value
            for key, value in (("wavelength", wavelength), ("num_modes", num_modes))
            if value is not None
        }
        if overrides:
            await _blocking(em.settings, **overrides)
        await _blocking(em.FDM, verbose=False)
        n_eff = await _blocking(em.get, "effective_index")
        try:
            loss = await _blocking(em.get, "loss_dB_per_m")
        except ToolError:
            loss = None
        return SolveResult(modes=_mode_results(n_eff, loss))

    @server.tool(name="emode_close")
    async def emode_close(save: bool = False) -> CloseResult:
        """Close the session and release the license. Harmless when no session is open."""
        return CloseResult(closed=await _blocking(session.shut, save=save))

    return server
