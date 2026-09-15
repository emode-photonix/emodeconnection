"""Public wire types for chi(2) nonlinear processes (SHG/SFG/DFG) declared
on a section via `EM_straight_section(nonlinear=...)` /
`EM_taper_section(nonlinear=...)`.

See the server's `em_types/nonlinear/section.py` (the `Chi2Section` family)
and `em_types/nonlinear/chi2.py` (`chi2Params`/`QPMSpec`) for the internal
counterparts these cross the wire into.
"""

from typing import Literal

from pydantic import ConfigDict, model_validator

from .types import ArgumentError, TaggedModel, register_type


class Chi2Process(TaggedModel):
    """Base for a declared chi(2) nonlinear process (SHG/SFG/DFG) on a
    section. Not itself a wire type -- construct `SHGProcess`, `SFGProcess`,
    or `DFGProcess`.

    The user names the process's *input* wavelength(s); the remaining one is
    derived by energy conservation (`derived_wavelength`) and matched
    against the section's own profile wavelengths (within
    `wavelength_tolerance`) when the section is built on the server -- not
    here, since the profile isn't known yet when this object is constructed.

    `poling_period=None` (the default) means no periodic quasi-phase-
    matching; `duty_cycle`/`qpm_order` only have meaning when it's set, so
    they must stay at their defaults otherwise. `inverted=True` flips the
    local chi(2) sign for *this* section on its own -- the building block
    for manual poling from alternating sections (optionally
    `EM_copy_section` copies of a normal/inverted pair), independent of
    whether `poling_period` is also set.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    wavelength_tolerance: float = 0.01  # nm
    poling_period: float | None = None  # nm
    duty_cycle: float = 0.5
    qpm_order: int = 1
    inverted: bool = False
    radiation: bool = False
    radiation_threshold: float = 1e3  # 1/m
    solver_method: str = 'RK45'
    solver_rtol: float = 1e-8
    solver_atol: float = 1e-10
    max_step: float | None = None

    @model_validator(mode='after')
    def _validate_chi2_process(self) -> 'Chi2Process':
        name = type(self).__name__
        if self.poling_period is None:
            if self.duty_cycle != 0.5 or self.qpm_order != 1:
                raise ArgumentError(
                    'duty_cycle/qpm_order only apply when poling_period is set',
                    name,
                    'poling_period',
                )
        elif self.poling_period <= 0:
            raise ArgumentError('poling_period must be > 0', name, 'poling_period')
        if not (0 < self.duty_cycle < 1):
            raise ArgumentError('duty_cycle must be in (0, 1)', name, 'duty_cycle')
        if self.qpm_order < 1:
            raise ArgumentError('qpm_order must be >= 1', name, 'qpm_order')
        if self.radiation_threshold <= 0:
            raise ArgumentError(
                'radiation_threshold must be > 0', name, 'radiation_threshold'
            )
        if self.wavelength_tolerance <= 0:
            raise ArgumentError(
                'wavelength_tolerance must be > 0', name, 'wavelength_tolerance'
            )
        if self.solver_rtol <= 0 or self.solver_atol <= 0:
            raise ArgumentError(
                'solver_rtol/solver_atol must be > 0', name, 'solver_rtol'
            )
        return self

    @property
    def process(self) -> Literal['SHG', 'SFG', 'DFG']:
        raise NotImplementedError

    @property
    def input_wavelengths(self) -> tuple[float, ...]:
        """The wavelength(s) the user named directly."""
        raise NotImplementedError

    @property
    def derived_wavelength(self) -> float:
        """The remaining wavelength, from energy conservation."""
        raise NotImplementedError

    @property
    def wavelengths(self) -> tuple[float, ...]:
        """All three process wavelengths (inputs + derived), sorted."""
        return tuple(sorted({*self.input_wavelengths, self.derived_wavelength}))


@register_type
class SHGProcess(Chi2Process):
    """Second-harmonic generation: `pump_wavelength` -> `pump_wavelength/2`."""

    pump_wavelength: float

    @property
    def process(self) -> Literal['SHG']:
        return 'SHG'

    @property
    def input_wavelengths(self) -> tuple[float, ...]:
        return (self.pump_wavelength,)

    @property
    def derived_wavelength(self) -> float:
        return self.pump_wavelength / 2


@register_type
class SFGProcess(Chi2Process):
    """Sum-frequency generation: two pumps combine into one signal, with
    `1/signal = 1/pump_wavelength_1 + 1/pump_wavelength_2`."""

    pump_wavelength_1: float
    pump_wavelength_2: float

    @property
    def process(self) -> Literal['SFG']:
        return 'SFG'

    @property
    def input_wavelengths(self) -> tuple[float, ...]:
        return (self.pump_wavelength_1, self.pump_wavelength_2)

    @property
    def derived_wavelength(self) -> float:
        return 1.0 / (1.0 / self.pump_wavelength_1 + 1.0 / self.pump_wavelength_2)


@register_type
class DFGProcess(Chi2Process):
    """Difference-frequency generation: pump minus signal produces an idler,
    with `1/idler = 1/pump_wavelength - 1/signal_wavelength`.
    `signal_wavelength` must be longer than `pump_wavelength` -- the pump is
    the shortest (highest-energy) of the three wavelengths, by energy
    conservation."""

    pump_wavelength: float
    signal_wavelength: float

    @model_validator(mode='after')
    def _validate_dfg(self) -> 'DFGProcess':
        if self.signal_wavelength <= self.pump_wavelength:
            raise ArgumentError(
                'signal_wavelength must be longer than pump_wavelength',
                'DFGProcess',
                'signal_wavelength',
            )
        return self

    @property
    def process(self) -> Literal['DFG']:
        return 'DFG'

    @property
    def input_wavelengths(self) -> tuple[float, ...]:
        return (self.pump_wavelength, self.signal_wavelength)

    @property
    def derived_wavelength(self) -> float:
        return 1.0 / (1.0 / self.pump_wavelength - 1.0 / self.signal_wavelength)
