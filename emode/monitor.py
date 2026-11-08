"""Wire types of monitors: the field on a plane inside a circuit.

A monitor is an output-only extraction plane, declared with `monitor()`. It
does not enter the S-matrix and cannot be connected; the circuit is excited
from its ports as usual, and the total field (forward and backward, every
slice) is sampled on the plane. Its main use is a grating coupler: the field
the grating radiates upward, its power, and its overlap with a fiber mode.
By Lorentz reciprocity the coupling from the fiber back into the launched
waveguide mode is the same number.

Lengths are in nm, angles in degrees, powers in W.
"""

from typing import Dict, Optional

import numpy as np
from pydantic import ConfigDict, model_validator

from .types import ArgumentError, TaggedModel, register_type


@register_type
class Monitor(TaggedModel):
    """A plane through the point (`w`, `y`) of the circuit, tilted by `angle` about x.

    `w` is the position along the circuit from its left port and `y` the
    height, both in nm. `angle` is in degrees from the horizontal: 0 is a
    plane parallel to the chip surface, positive tips the far end up. The
    plane spans the full window in x. Along the plane it runs as far as the
    circuit and the window allow, or `length` (nm) centered on (`w`, `y`).
    `resolution` (nm) is the sampling step along the plane (default: the
    window's x resolution).
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    w: float
    y: float
    angle: float = 0.0
    length: Optional[float] = None
    resolution: Optional[float] = None

    @model_validator(mode='after')
    def _validate(self) -> 'Monitor':
        if not -90.0 < self.angle < 90.0:
            raise ArgumentError('angle must be strictly between -90 and 90 degrees', 'monitor', 'angle')
        if self.length is not None and self.length <= 0:
            raise ArgumentError('length must be > 0', 'monitor', 'length')
        if self.resolution is not None and self.resolution <= 0:
            raise ArgumentError('resolution must be > 0', 'monitor', 'resolution')
        return self


@register_type
class PlaneField(TaggedModel):
    """The field sampled on a monitor's plane.

    `fields` maps each component ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz') to a
    complex array (len(x), len(s)): `x` (nm) across the plane, `s` (nm) along
    it, at chain positions `w` and heights `y` (nm). `Sn` is the time-averaged
    Poynting flux through the plane (W/m^2, positive along the plane's
    upward normal), `power_through_plane` its integral (W), and `power_in`
    the power launched at the port (W).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    wavelength: float
    x: np.ndarray
    s: np.ndarray
    w: np.ndarray
    y: np.ndarray
    angle: float
    fields: Dict[str, np.ndarray]
    Sn: np.ndarray
    power_through_plane: float
    power_in: float


@register_type
class BeamCoupling(TaggedModel):
    """A monitor's coupling to a Gaussian beam.

    `coupling` is `power_fraction * overlap`: the fraction of the launched
    power that crosses the plane, times the normalized overlap of the field's
    `polarization` component on the plane with the beam. The beam has mode
    field diameter `mode_field_diameter` (nm), is centered at `center`
    (x, s in nm on the plane) and tilted by `angle` (degrees) from the plane's
    normal toward +s, in a medium of `index`.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: str
    wavelength: float
    coupling: float
    overlap: float
    power_fraction: float
    power_through_plane: float
    power_in: float
    mode_field_diameter: float
    center: tuple
    angle: float
    index: float
    polarization: str
