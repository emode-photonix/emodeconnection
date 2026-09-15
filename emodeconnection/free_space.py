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

from pydantic import ConfigDict

from .types import TaggedModel, register_type


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
