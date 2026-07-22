"""Public Ports API wire types.

`Port` is attached to any `EM_<x>_section(ports=[...])` call and is
addressable globally as `"<section_name>.<port_name>"` (see
`EM_connect_ports`). `basis_transform` is a generic slot for a per-port
modal transform; `AngledFacetMap` (a per-mode linear phase ramp about a
hinge line, modeling an angled facet) is the first thing that can go in it.
"""
from typing import Literal

from pydantic import ConfigDict

from .types import TaggedModel, register_type


@register_type
class Bounds(TaggedModel):
    """A rectangular sub-window of a port's full cross-section.

    `offset` is `(x_center, y_min)` — the window is centered on `x_center`
    horizontally and starts at `y_min` vertically, matching the existing
    `offset` convention used elsewhere for section port windows.
    """
    model_config = ConfigDict(frozen=True)

    window_width: float
    window_height: float
    offset: tuple[float, float] = (0.0, 0.0)


@register_type
class AngledFacetMap(TaggedModel):
    """Wire-level spec for an angled facet: a per-mode linear phase ramp
    about a hinge line. Mirrors the internal `phase_tilt`/`FacetSpec`.
    """
    model_config = ConfigDict(frozen=True)

    pivot_angle: float
    polar_angle: float = 0.0
    pivot_point: tuple[float, float] | None = None
    degrees: bool = True


# v1 has exactly one member. A plain alias (not yet a real union) keeps it
# trivially extensible later: `type BasisTransform = AngledFacetMap | ModeSelectMap | ...`
# without ever changing Port's type signature.
type BasisTransform = AngledFacetMap


@register_type
class Port(TaggedModel):
    """A named port on a section, addressable as `"<section_name>.<port_name>"`.

    The default for any section is
    `[Port('left', None, 'left'), Port('right', None, 'right')]`. Supplying
    a port in `EM_<x>_section(ports=[...])` replaces every existing port on
    the same `side`; multiple ports may share a side.
    """
    model_config = ConfigDict(frozen=True)

    name: str
    window: Bounds | None = None
    side: Literal['left', 'right']
    basis_transform: BasisTransform | None = None


def make_angled_facet_port(
    name: str,
    side: Literal['left', 'right'],
    pivot_angle: float,
    polar_angle: float = 0.0,
    pivot_point: tuple[float, float] | None = None,
    degrees: bool = True,
    window: Bounds | None = None,
) -> Port:
    """Build a Port with an angled-facet basis_transform attached."""
    return Port(
        name=name,
        window=window,
        side=side,
        basis_transform=AngledFacetMap(
            pivot_angle=pivot_angle,
            polar_angle=polar_angle,
            pivot_point=pivot_point,
            degrees=degrees,
        ),
    )
