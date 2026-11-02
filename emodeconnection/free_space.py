"""Client-side diagnostics for a free-space field's projection onto its target port.

`EM_free_space_field(...)` declares a beam (a Gaussian, say); at solve time the
server projects it onto the target port's modal basis
(`ModalBasis.project` on the server -- design in `docs/free_space_field/` of
the `emode` repo) and warns server-side, in its own log, when a meaningful
share of the input field falls outside the basis's span and is silently
dropped. `FreeSpaceProjection` is the read-only, wire-safe report of that same
number, returned by `EM_free_space_field_diagnostics(name)` so a client script
can check or plot it without parsing a log line.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator

from .types import ArgumentError, TaggedModel, register_type


@register_type
class FreeSpaceProjection(TaggedModel):
    """How much of a declared free-space field's input power the target port's
    modal basis actually represents.

    `captured_fraction` is `‖Σ c_k mode_k‖² / ‖E_user‖²` over the sampled input
    field's own grid -- 1.0 means the beam lies entirely in the basis's span;
    less than that means the difference was silently dropped by the
    projection (`docs/free_space_field/design.md` §4 of the `emode` repo).
    `num_modes` is the launch basis's mode count at the moment of projection --
    context for what "widen the basis" would mean if `captured_fraction` reads
    low. `wavelength` is that projection's own wavelength (nm); a
    multi-wavelength simulation can have a different value per port.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    port: str
    wavelength: float
    captured_fraction: float
    num_modes: int

    @property
    def dropped_fraction(self) -> float:
        """`1 - captured_fraction`, clipped to `[0, 1]`: the share of the
        input's power the target basis could not represent. Clipped because
        `captured_fraction` is a numerical ratio, not a probability, and can
        land a hair above 1.0 for a field already well inside the span."""
        return max(0.0, min(1.0, 1.0 - self.captured_fraction))

    def __str__(self) -> str:
        return (
            f'free-space field {self.name!r} at port {self.port!r} '
            f'({self.wavelength:g} nm): {100 * self.captured_fraction:.1f}% captured '
            f'by a {self.num_modes}-mode basis'
        )


@register_type
class Screen(TaggedModel):
    """A plane screen in front of a port, the argument of ``farfield(screen=...)``.

    The screen's center sits `distance` (nm) from the port plane along the
    direction the field leaves the port, at `center` (x, y in nm, the port
    plane's own coordinates). ``None`` puts it on the beam's centroid. It
    spans `width` (nm) along its own u axis and `height` (nm) along its v axis,
    sampled at `points` (one number or ``(nu, nv)``). `tilt` (degrees) turns
    it: by ``tilt[0]`` about y, which tips its normal toward +x, then by
    ``tilt[1]``, which tips the normal toward +y. The normal is then
    ``(sin tx cos ty, sin ty, cos tx cos ty)``, the u axis ``(cos tx, 0, -sin tx)``
    and the v axis completes the frame. Untilted, u is x and v is y.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    distance: float
    width: float
    height: float
    center: Optional[Tuple[float, float]] = None
    tilt: Tuple[float, float] = (0.0, 0.0)
    points: Tuple[int, int] = (101, 101)

    @field_validator("center", "tilt", mode="before")
    @classmethod
    def _pair(cls, v):
        """JSON has no tuples: a ``[x, y]`` from the other side becomes ``(x, y)``."""
        if v is None:
            return v
        return tuple(float(x) for x in v)

    @field_validator("points", mode="before")
    @classmethod
    def _points(cls, v):
        if isinstance(v, (int, np.integer)):
            return (int(v), int(v))
        return tuple(int(x) for x in v)

    @model_validator(mode="after")
    def _validate(self) -> "Screen":
        for name in ("distance", "width", "height"):
            if not getattr(self, name) > 0:
                raise ArgumentError(f"{name} must be > 0", "farfield", name)
        if not all(-90.0 < t < 90.0 for t in self.tilt):
            raise ArgumentError("each tilt must be strictly between -90 and 90 degrees", "farfield", "tilt")
        if len(self.points) != 2 or min(self.points) < 2:
            raise ArgumentError("points must be at least 2 along each axis", "farfield", "points")
        return self


@register_type
class ScreenField(TaggedModel):
    """The field on a `Screen`, for 1 W in the near field.

    `s` and `t` (nm) are the positions along the screen's u and v axes,
    `x`, `y`, `z` (nm, each ``(len(s), len(t))``) the same points in the port
    plane's frame, `z` measured from the port plane along the direction the
    field leaves it. `fields` maps 'Ex', 'Ey', 'Ez' (V/m) and 'Hx', 'Hy', 'Hz'
    (A/m) to complex arrays in that frame. `Sn` is the time-averaged Poynting
    flux through the screen along its `normal` (W/m^2) and
    `power_through_screen` its integral (W).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    screen: Screen
    s: np.ndarray
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    u_axis: Tuple[float, float, float]
    v_axis: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    fields: Dict[str, np.ndarray]
    Sn: np.ndarray
    power_through_screen: float


@register_type
class FarField(TaggedModel):
    """The far field of a mode, normalized to 1 W in the near field.

    The field on the port plane (or a profile's mode) is decomposed into
    plane waves in a uniform medium of `index` (its angular spectrum), and
    each direction of propagation is labeled by its direction cosines `ux` and
    `uy`, the sines of the angles from the axis along x and y. Directions with
    ``ux**2 + uy**2 > 1`` carry no power and read zero.

    - `fields`: 'Ex', 'Ey', 'Ez', each ``(len(ux), len(uy))``, the far field
      times ``r exp(j k r)`` (V), in the port plane's frame.
    - `intensity`: the radiant intensity (W/sr).
    - `radiated_power`: its integral over the directions sampled (W). The rest
      of the 1 W is in directions outside the sampled range, in the near
      field's lossy plane waves, or the difference between the mode's own
      power and that of its transverse E in the uniform medium.
    - `divergence`: the half-angles (degrees) along x and y from the second
      moments of the intensity in direction cosines, ``arcsin(2 sigma_u)``,
      which is the 1/e^2 half-angle of a Gaussian beam.
    - `m_squared`: the beam quality factor along x and y (ISO 11146 second
      moments of the near field and of its angular spectrum).
    - `waist` and `waist_position`: the embedded waist radius along x and y
      (nm, twice the second-moment width) and its distance (nm) from the
      plane along the direction of travel (negative: behind the plane).
    - `centroid`: the near field's intensity centroid (x, y in nm).
    - `screen`: the field on the `Screen` given to ``farfield()``, if any.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wavelength: float
    index: float
    port: Optional[str] = None
    profile: Optional[str] = None
    mode: int = 0
    ux: np.ndarray
    uy: np.ndarray
    fields: Dict[str, np.ndarray]
    intensity: np.ndarray
    radiated_power: float
    divergence: Tuple[float, float]
    m_squared: Tuple[float, float]
    waist: Tuple[float, float]
    waist_position: Tuple[float, float]
    centroid: Tuple[float, float]
    screen: Optional[ScreenField] = None

    @property
    def theta_x(self) -> np.ndarray:
        """The angle from the axis in the x-z plane (degrees), ``arcsin(ux)``."""
        return np.degrees(np.arcsin(np.clip(self.ux, -1.0, 1.0)))

    @property
    def theta_y(self) -> np.ndarray:
        """The angle from the axis in the y-z plane (degrees), ``arcsin(uy)``."""
        return np.degrees(np.arcsin(np.clip(self.uy, -1.0, 1.0)))

    def cut(self, axis: str = "x") -> Tuple[np.ndarray, np.ndarray]:
        """``(angle in degrees, intensity in W/sr)`` along `axis` ('x' or 'y'), through the
        sampled direction nearest the axis."""
        if axis == "x":
            return self.theta_x, np.asarray(self.intensity)[:, int(np.argmin(np.abs(self.uy)))]
        if axis == "y":
            return self.theta_y, np.asarray(self.intensity)[int(np.argmin(np.abs(self.ux))), :]
        raise ArgumentError(f"axis must be 'x' or 'y', got {axis!r}", "FarField.cut", "axis")

    def plot(self, ax=None):
        """Plot the radiant intensity against the angles, its x and y cuts, and the
        screen's flux when there is a screen. Returns the matplotlib Figure.

        `ax`, if given, is a sequence of Axes to draw into: two, or three with a
        screen.
        """
        from matplotlib import pyplot as plt

        panels = 3 if self.screen is not None else 2
        if ax is None:
            fig, ax = plt.subplots(1, panels, figsize=(4.5 * panels, 4), constrained_layout=True)
        else:
            fig = ax[0].figure
        tx, ty = self.theta_x, self.theta_y
        image = ax[0].pcolormesh(tx, ty, np.asarray(self.intensity).T, shading="auto")
        ax[0].set_xlabel("θx (deg)")
        ax[0].set_ylabel("θy (deg)")
        ax[0].set_title(f"far field, {self.wavelength:g} nm")
        ax[0].set_aspect("equal")
        fig.colorbar(image, ax=ax[0], label="intensity (W/sr)")
        for axis, label in (("x", "θx"), ("y", "θy")):
            angle, intensity = self.cut(axis)
            ax[1].plot(angle, intensity, label=f"{label} cut")
        ax[1].set_xlabel("angle (deg)")
        ax[1].set_ylabel("intensity (W/sr)")
        ax[1].set_title(f"M² = {self.m_squared[0]:.3f}, {self.m_squared[1]:.3f}")
        ax[1].legend()
        if self.screen is not None:
            screen = self.screen
            image = ax[2].pcolormesh(
                np.asarray(screen.s) / 1000, np.asarray(screen.t) / 1000, np.asarray(screen.Sn).T, shading="auto"
            )
            ax[2].set_xlabel("u (µm)")
            ax[2].set_ylabel("v (µm)")
            ax[2].set_title(f"screen at {screen.screen.distance / 1000:g} µm")
            ax[2].set_aspect("equal")
            fig.colorbar(image, ax=ax[2], label="flux (W/m²)")
        return fig
