"""Public Ports API wire types.

`Port` is attached to any `EM_<x>_section(ports=[...])` call and is
addressable globally as `"<section_name>.<port_name>"` (see
`EM_connect_ports`). `basis_transform` is a generic slot for a per-port
modal transform; `AngledFacetMap` (a per-mode linear phase ramp about a
hinge line, modeling an angled facet) is the first thing that can go in it.
"""
from typing import Literal, TypeAlias

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
# trivially extensible later: `BasisTransform: TypeAlias = AngledFacetMap | ModeSelectMap | ...`
# without ever changing Port's type signature. `typing.TypeAlias` (PEP 613,
# not the PEP 695 `type` statement) -- this package supports Python 3.10+,
# and `type X = ...` is 3.12-only syntax.
BasisTransform: TypeAlias = AngledFacetMap


@register_type
class JunctionOverride(TaggedModel):
    """Settings for one junction, passed as the third element of an
    `EM_connect_ports` entry:

        em.connect_ports([('wg.right', 'gap.left', JunctionOverride(...))])

    Every field defaults to `None`, meaning "inherit". Setting one overrides
    only that field, so `JunctionOverride(projection='square')` switches the
    mode-matching algebra for that junction and leaves everything else alone.

    This is the innermost of three levels, outermost first:

    1. `EM_EME_settings(projection=...)` — the whole simulation.
    2. `EM_<x>_section(settings={'projection': ...})` — one section, including
       the internal staircase a taper or GDS section refines into.
    3. this object on a connection — that one junction.

    It belongs to the *connection* rather than to either port because every
    field here describes the **interface**, not one side of it: which algebra
    joins the two mode sets, and how the resulting S-matrix is normalized. One
    declaration site per junction also means two sides can never disagree, so
    there is no left/right precedence rule to learn.

    Only explicitly connected junctions can carry one. Junctions formed by
    default left/right adjacency have no declaration site -- name the pair in
    `EM_connect_ports` to configure it, which supersedes the adjacency. The
    staircase junctions inside a refined taper/GDS section are governed by that
    section's `settings` (level 2); they are created and discarded as
    refinement bisects, so they have no stable identity to address.
    """
    model_config = ConfigDict(frozen=True)

    # 'ctcr' is the thesis Ct/Cr reduction; 'square' is the direct square
    # projection, which conserves power at a regular<->PWD junction where
    # 'ctcr' loses ~3.7%. 'ctcr' is the default until a z-staircase reference
    # settles which one is right about reflection at a real width step.
    projection: Literal['ctcr', 'square'] | None = None
    # 'square' only: take the right-incidence transmission block from the solve
    # ('solve') or impose it from the left block by reciprocity
    # ('reciprocity'). Ignored under 'ctcr', which imposes it structurally.
    reverse_incidence: Literal['reciprocity', 'solve'] | None = None
    # True unitarizes the junction S-matrix, False leaves it raw, 'gain' keeps
    # the raw matrix but clips singular values at 1.
    junction_normalization: bool | Literal['gain'] | None = None

    def is_empty(self) -> bool:
        """True when nothing is overridden, i.e. this is inert."""
        return all(getattr(self, f) is None for f in type(self).model_fields)

    def key(self) -> tuple:
        """Hashable identity, for the server's junction cache key.

        Only fields that are actually set appear, so a default-constructed
        override keys identically to no override at all and cannot split the
        cache. Every field is a `str`/`bool`/`None`, so the values are already
        hashable as-is.
        """
        return tuple(
            (f, getattr(self, f))
            for f in sorted(type(self).model_fields)
            if getattr(self, f) is not None
        )


@register_type
class Port(TaggedModel):
    """A named port on a section, addressable as `"<section_name>.<port_name>"`.

    The default for any section is
    `[Port('left', None, 'left'), Port('right', None, 'right')]`. Supplying
    a port in `EM_<x>_section(ports=[...])` replaces every existing port on
    the same `side`; multiple ports may share a side.

    A `Port` carries two offsets that mean different things, and mixing them
    up is easy:

    - `window.offset` locates the sub-window *inside this section's own
      cross-section*. It selects which material the port exposes; it does not
      move the section relative to anything.
    - `offset` (below) is a **content displacement**: it says where this
      section's cross-section sits relative to whatever the port is connected
      to. Connecting two ports pins their origins together (the point
      `(x_center, y_min)` of each port's cross-section), and `offset` shifts
      this side's content off that shared origin.

    So `window` answers "which part of me?" and `offset` answers "where am I?".
    """
    model_config = ConfigDict(frozen=True)

    name: str
    window: Bounds | None = None
    side: Literal['left', 'right']
    basis_transform: BasisTransform | None = None
    # Content displacement of this section relative to the port it connects
    # to, in nm: `offset=(1300, 0)` moves this section's cross-section +1300
    # in x from the connection's shared origin.
    offset: tuple[float, float] = (0.0, 0.0)


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
