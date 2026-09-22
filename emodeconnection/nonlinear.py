"""Public wire types for chi(2) nonlinear processes (SHG/SFG/DFG) declared
on a section via `EM_straight_section(nonlinear=...)` /
`EM_taper_section(nonlinear=...)`, plus the CW excitation (`Source`) and
driven-solve result (`CircuitResponse`) types for
`EM_settings(excitation=...)` / `EM_get('response')`.

See the server's `em_types/nonlinear/section.py` (the `Chi2Section` family)
and `em_types/nonlinear/chi2.py` (`chi2Params`/`QPMSpec`) for the internal
counterparts `Chi2Process` crosses the wire into, and
`em_types/source.py`/`em_types/section/circuit.py::CircuitSection.respond`
for `Source`/`CircuitResponse`'s counterparts.
"""

import cmath
from typing import Literal, TypeAlias

import numpy as np
from pydantic import ConfigDict, model_validator

from .smatrix import PortIndexed
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
    # 1e-6/1e-9 matches a 1e-12 reference solution to 3e-5 relative and runs 2.2x
    # faster than the 1e-8/1e-10 it replaces (14.7 ms vs 32.1 ms on a 181 um SHG
    # slice). The envelope equations are smooth and the residual error is far
    # below the discretisation error of the modes driving them.
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-9
    max_step: float | None = None
    # Forward mode amplitudes are sampled this many times along each chi(2)
    # slice after the driven solve converges, and reach the caller as
    # `CircuitResponse.trajectories`. A refined taper is many slices, so this
    # is per slice, not per section. 0 disables the sampling entirely.
    z_samples: int = 100

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
        if self.z_samples < 0:
            raise ArgumentError('z_samples must be >= 0', name, 'z_samples')
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


@register_type
class Source(TaggedModel):
    """A CW excitation at one port/wavelength/mode, declared via
    `EM_settings(excitation=Source(...))`. `|amplitude|**2 == power` [W].

    `port` is a bare side ('left'/'right') or a qualified
    `'<section>.<port>'` name, resolved against the built circuit at
    `EM_EME()` time -- same convention as `EM_connect_ports`. `wavelength`
    is matched against the circuit's own wavelengths within
    `wavelength_tolerance`, same as `Chi2Process`.

    A future `PulsedSource` (for chi(3)/Raman work) is meant to be a
    sibling of this class, not extra fields bolted onto it -- see
    `Excitation` below.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    port: str = 'left'
    wavelength: float
    mode: int = 0
    power: float = 1.0  # W
    phase: float = 0.0  # rad
    wavelength_tolerance: float = 0.01  # nm

    @model_validator(mode='after')
    def _validate_source(self) -> 'Source':
        if self.power < 0:
            raise ArgumentError('power must be >= 0', 'Source', 'power')
        if self.mode < 0:
            raise ArgumentError('mode must be >= 0', 'Source', 'mode')
        if self.wavelength <= 0:
            raise ArgumentError('wavelength must be > 0', 'Source', 'wavelength')
        if self.wavelength_tolerance <= 0:
            raise ArgumentError(
                'wavelength_tolerance must be > 0', 'Source', 'wavelength_tolerance'
            )
        return self

    @property
    def amplitude(self) -> complex:
        """sqrt(power) * exp(j*phase), in sqrt(W)."""
        return cmath.sqrt(self.power) * cmath.exp(1j * self.phase)

    @property
    def key(self) -> tuple[str, float, int]:
        """(port, wavelength, mode) -- two Sources sharing a key both drive
        the same mode; their amplitudes sum."""
        return (self.port, self.wavelength, self.mode)


# v1 has exactly one member. A plain alias (not yet a real union) keeps it
# trivially extensible for a future PulsedSource (chi(3)/Raman work) without
# ever changing EM_settings's/EM_EME's signature -- see port.py's
# BasisTransform for the same pattern. `typing.TypeAlias` (PEP 613), not the
# PEP 695 `type` statement -- this package supports Python 3.10+.
Excitation: TypeAlias = Source


@register_type
class Chi2Trajectory(TaggedModel):
    """Forward mode amplitudes along one chi(2) section, sampled after the
    driven solve converged.

    One of these per chi(2) slice, so a refined taper produces many and a
    straight section produces one. `z` is the global propagation coordinate
    along the chain in nm, measured from its left port, so the entries of
    `CircuitResponse.trajectories` tile the chain in order.

    `wavelengths` and `amplitudes` are parallel tuples rather than a dict
    keyed by wavelength: a float-keyed dict does not survive the wire
    round-trip (see the `effective_area`/`group_index` entry in the server's
    docs/known_bugs.md).

    Amplitudes are complex fields in sqrt(W), the same normalization
    `CircuitResponse.input`/`output` use, so `sum(|a|**2)` over modes is
    watts. Forward direction only: the backward wave is solved but not
    sampled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    section: str
    z: np.ndarray  # (K,) nm, global along the chain
    wavelengths: tuple[float, ...]
    amplitudes: tuple[np.ndarray, ...]  # each (M_wl, K) complex, sqrt(W)

    def _wavelength_index(self, wavelength: float) -> int:
        for i, wl in enumerate(self.wavelengths):
            if wl == wavelength:
                return i
        # Tolerate the float drift a user gets from arithmetic on a declared
        # wavelength; the process's own matching is tolerance-based too.
        close = [i for i, wl in enumerate(self.wavelengths) if abs(wl - wavelength) < 1e-6]
        if len(close) == 1:
            return close[0]
        raise ArgumentError(
            f'not one of this trajectory\'s wavelengths {self.wavelengths}',
            'Chi2Trajectory',
            'wavelength',
        )

    def amplitude_vs_z(self, wavelength: float) -> np.ndarray:
        """Per-mode complex amplitude, shape `(modes, len(z))`."""
        return self.amplitudes[self._wavelength_index(wavelength)]

    def power_vs_z(self, wavelength: float) -> np.ndarray:
        """Total power (W) at each sampled z, summed over modes."""
        return np.sum(np.abs(self.amplitude_vs_z(wavelength)) ** 2, axis=0)


@register_type
class CircuitResponse(TaggedModel, PortIndexed):
    """The driven (power-dependent) response at one operating point,
    returned by `EM_get('response')` after `EM_EME()` with a declared
    excitation. Port layout (`port_sizes`/`port_names`/`wavelengths`) matches
    `SMatrix`'s -- same `port_index`/`amplitude`/`power` accessors.

    `input`/`output` are complex mode amplitudes (sqrt(W)) in port order,
    `input` the driven incoming wave and `output` the driven outgoing wave --
    for a `linear` response, `output == SMatrix.data @ input`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    port_sizes: tuple[int, ...]
    port_names: tuple[str, ...] | None = None
    wavelengths: tuple[float, ...] | None = None
    input: np.ndarray  # complex, shape (total_modes,)
    output: np.ndarray  # complex, shape (total_modes,)
    converged: bool
    iterations: int
    residual: float
    radiated_power: dict[float, float] | None = None  # wavelength -> total W
    sources: tuple[Source, ...] = ()
    linear: bool
    # One entry per chi(2) slice, in order along the chain. Empty when the
    # circuit has no chi(2) section, or when every process set z_samples=0.
    trajectories: tuple[Chi2Trajectory, ...] = ()

    def amplitude(
        self,
        port: int | str,
        wavelength: float | None = None,
        *,
        direction: Literal['out', 'in'] = 'out',
    ) -> np.ndarray:
        """Per-mode complex amplitude at `port` (index or name, optionally
        disambiguated by `wavelength`)."""
        idx = self._resolve_port(port) if wavelength is None else self.port_index(port, wavelength)
        s = self.port_slices[idx]
        data = self.output if direction == 'out' else self.input
        return data[s]

    def power(
        self,
        port: int | str,
        wavelength: float | None = None,
        *,
        direction: Literal['out', 'in'] = 'out',
    ) -> float:
        """Total power (W) at `port`, summed over its modes."""
        return float(np.sum(np.abs(self.amplitude(port, wavelength, direction=direction)) ** 2))

    def total_power(self, direction: Literal['out', 'in'] = 'out') -> float:
        """Total power (W) across every external port."""
        data = self.output if direction == 'out' else self.input
        return float(np.sum(np.abs(data) ** 2))

    def power_vs_z(
        self, wavelength: float, *, section: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward power along the chain: `(z_nm, power_W)`.

        Stitches every chi(2) slice's trajectory in z order, so a refined
        taper reads as one curve. `section` restricts to the slices of one
        declared section; the default uses all of them.

        Raises when no trajectory carries `wavelength`, which is what a
        caller gets for a wavelength the process does not couple, or when
        the processes ran with `z_samples=0`.
        """
        z, p = self._stitch(wavelength, section, lambda tr: tr.power_vs_z(wavelength))
        return z, p

    def amplitude_vs_z(
        self, wavelength: float, *, section: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward per-mode amplitude along the chain: `(z_nm, a)` with `a`
        shaped `(modes, len(z))`."""
        return self._stitch(
            wavelength, section, lambda tr: tr.amplitude_vs_z(wavelength), axis=-1
        )

    def _stitch(self, wavelength, section, take, axis=0):
        chosen = [
            tr
            for tr in self.trajectories
            if (section is None or tr.section == section)
            and any(abs(wl - wavelength) < 1e-6 for wl in tr.wavelengths)
        ]
        if not chosen:
            raise ArgumentError(
                f'no sampled trajectory at {wavelength} nm'
                + (f' on section {section!r}' if section is not None else '')
                + '. Declare a chi(2) process with z_samples > 0 and re-run EME().',
                'CircuitResponse.power_vs_z',
                'wavelength',
            )
        chosen.sort(key=lambda tr: float(tr.z[0]))
        return (
            np.concatenate([tr.z for tr in chosen]),
            np.concatenate([take(tr) for tr in chosen], axis=axis),
        )
