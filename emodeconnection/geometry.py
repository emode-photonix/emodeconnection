"""User-facing geometry primitives for transformation-optics section construction.

These types are passed through `em.to_section(...)` to describe waveguide
centerlines and placements. They round-trip through the wire serialization
registry (see `emodeconnection.types.register_type`).

Local-frame convention
----------------------
Every curve is defined in its own local frame: it starts at the local origin
with the initial tangent along **+z** (the section's propagation axis). A
`Pose` then places that frame into the section frame.

C1 continuity in `Path`
-----------------------
Given the local-frame convention, the C1 joint between adjacent segments is
automatic: every segment's start tangent is +z in its own local frame, and
the previous segment's end pose carries the cumulative rotation. So a
`Line -> Arc`, `Arc -> Line`, or `Arc -> Arc` joint is C1 by construction.
For `BSpline`, the segment must be authored with its start derivative along
+z (i.e. dx/dt ~ 0, dz/dt > 0 at the start parameter); this is the BSpline
author's responsibility.
"""
from typing import Union

import numpy as np
from pydantic import ConfigDict, computed_field, field_validator, model_validator

from .types import ArgumentError, TaggedModel, register_type


@register_type
class Pose(TaggedModel):
    """Position + orientation of a curve's local frame within the section frame.

    Mirrors the server-side `emode.em_types.pose.Pose` dataclass fields
    exactly. Planar geometry only for v1 (`y` and out-of-plane rotations are
    accepted but currently ignored by the coupler builder).
    """
    model_config = ConfigDict(frozen=True)

    w: float = 0.0
    z: float = 0.0
    x: float = 0.0
    y: float = 0.0
    azimuth: float = 0.0


@register_type
class Line(TaggedModel):
    """Straight segment of `length` along the local +z axis."""
    model_config = ConfigDict(frozen=True)

    length: float

    @field_validator('length')
    @classmethod
    def _positive_length(cls, v: float) -> float:
        if v <= 0:
            raise ArgumentError(f"length must be positive, got {v}", 'Line', 'length')
        return v


@register_type
class Arc(TaggedModel):
    """Circular arc.

    `radius` is signed: positive curves toward +x (center on the +x side of
    the start point), negative curves toward -x. `sweep_angle` is signed
    radians. Arc length is `abs(radius * sweep_angle)`.
    """
    model_config = ConfigDict(frozen=True)

    radius: float
    sweep_angle: float

    @field_validator('radius')
    @classmethod
    def _nonzero_radius(cls, v: float) -> float:
        if v == 0:
            raise ArgumentError("radius cannot be zero", 'Arc', 'radius')
        return v

    @field_validator('sweep_angle')
    @classmethod
    def _nonzero_sweep(cls, v: float) -> float:
        if v == 0:
            raise ArgumentError("sweep_angle cannot be zero", 'Arc', 'sweep_angle')
        return v

    @property
    def length(self) -> float:
        return abs(self.radius * self.sweep_angle)


@register_type
class BSpline(TaggedModel):
    """B-spline curve parameterized like `scipy.interpolate.BSpline`.

    `coefficients` is a list of (x, z) control points. `knots` is the knot
    vector; standard relation `len(knots) == len(coefficients) + degree + 1`.

    Reaches the v1 build only via `em.to_gds` (preview); v1 `em.to_section`
    raises before reaching a transform.
    """
    model_config = ConfigDict(frozen=True)

    knots: list[float]
    coefficients: list[list[float]]
    degree: int = 3

    @field_validator('knots', mode='before')
    @classmethod
    def _coerce_knots(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @field_validator('coefficients', mode='before')
    @classmethod
    def _coerce_coefficients(cls, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @model_validator(mode='after')
    def _check_knot_count(self) -> 'BSpline':
        expected = len(self.coefficients) + self.degree + 1
        if len(self.knots) != expected:
            raise ArgumentError(
                f"len(knots) must be len(coefficients) + degree + 1 = {expected}, "
                f"got {len(self.knots)}",
                'BSpline', 'knots',
            )
        if self.degree < 1:
            raise ArgumentError(f"degree must be >= 1, got {self.degree}", 'BSpline', 'degree')
        for i, c in enumerate(self.coefficients):
            if len(c) != 2:
                raise ArgumentError(
                    f"coefficients[{i}] must be a length-2 (x, z) pair, got length {len(c)}",
                    'BSpline', 'coefficients',
                )
        return self


# Convenient aliases for type hints elsewhere.
Curve = Union[Line, Arc, BSpline]


@register_type
class Path(TaggedModel):
    """Composition of curve segments forming a single waveguide centerline.

    Segments are placed end-to-end in the path's local frame; given the
    local-frame convention (each segment starts at origin with +z tangent),
    C1 continuity at every adjacent joint is automatic. See module docstring.
    """
    model_config = ConfigDict(frozen=True)

    segments: tuple[Union[Line, Arc, BSpline], ...]

    @field_validator('segments', mode='before')
    @classmethod
    def _to_tuple(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @model_validator(mode='after')
    def _nonempty(self) -> 'Path':
        if len(self.segments) == 0:
            raise ArgumentError("Path must contain at least one segment", 'Path', 'segments')
        return self

    @property
    def length(self) -> float:
        return float(sum(seg.length for seg in self.segments))
