"""Wire types of the pulsed nonlinear solver.

A pulsed simulation is declared with three objects and returns one:

- `PulsedSource`, the excitation: a femtosecond pulse (or a CW reference)
  launched into one port, declared with `settings(excitation=...)`.
- `PulsedProcess`, attached to a section with its `nonlinear=` argument: which
  nonlinearities act there, which modes are tracked, and how the section's
  dispersion is sampled.
- `Poling`, the domain pattern of a poled section, as a constant period, a
  period that varies along the section, or explicit domain walls.
- `PulseResponse`, returned by `get('pulse_response')` after `EME()` or
  `pulse_propagate()`: the spectrum of every tracked mode at the recorded
  positions, with band energy and power computed here on the client.

Units, as in the rest of the API: wavelengths and lengths in nm, durations
in fs, energies in pJ, powers in W, repetition rates in Hz. Arrays returned
in `PulseResponse` are in SI (Hz, sqrt(J/Hz)) with their nm and fs axes
alongside.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import ConfigDict, model_validator

from .types import ArgumentError, TaggedModel, register_type

_C0 = 299792458.0


@register_type
class PulsedSource(TaggedModel):
    """A pulse launched into one port of the circuit.

    `mode` is the tracked mode index that carries the pulse, or a dict
    `{mode: complex amplitude}` that splits it between tracked modes (the
    amplitudes are normalized, so only their ratios matter). Give exactly one
    of `energy` (pJ) and `peak_power` (W). `duration` is the intensity FWHM in
    fs. `chirp` is the dimensionless C of exp(j C t^2 / (2 T0^2)): positive
    raises the instantaneous frequency across the pulse. `shape='cw'` launches
    a continuous wave of `peak_power` at `center_wavelength` (no `duration`).

    The pulse is resolved on `num_points` frequencies spanning
    `wavelength_range` (nm). The time window is `num_points` over the
    frequency span; the default 16384 points over 450 to 3500 nm gives 28 ps.
    `alias` oversamples the real-field products (2 for chi(2), 3 for chi(3)).
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    center_wavelength: float
    duration: Optional[float] = None
    energy: Optional[float] = None
    peak_power: Optional[float] = None
    port: str = 'left'
    mode: Union[int, Dict[int, complex]] = 0
    shape: Literal['sech', 'gaussian', 'cw'] = 'sech'
    chirp: float = 0.0
    repetition_rate: Optional[float] = None
    wavelength_range: Tuple[float, float] = (450.0, 3500.0)
    num_points: int = 16384
    alias: int = 2

    @model_validator(mode='after')
    def _validate(self) -> 'PulsedSource':
        name = 'PulsedSource'
        if self.center_wavelength <= 0:
            raise ArgumentError('center_wavelength must be > 0', name, 'center_wavelength')
        lo, hi = self.wavelength_range
        if not 0 < lo < hi:
            raise ArgumentError('wavelength_range must be (min, max) with 0 < min < max', name, 'wavelength_range')
        if not lo < self.center_wavelength < hi:
            raise ArgumentError('center_wavelength must lie inside wavelength_range', name, 'center_wavelength')
        if self.shape == 'cw':
            if self.peak_power is None or self.energy is not None:
                raise ArgumentError("a 'cw' source takes peak_power (W) and no energy", name, 'peak_power')
        else:
            if self.duration is None or self.duration <= 0:
                raise ArgumentError('duration (fs FWHM) must be > 0', name, 'duration')
            if (self.energy is None) == (self.peak_power is None):
                raise ArgumentError('give exactly one of energy (pJ) and peak_power (W)', name, 'energy')
        for field, value in (('energy', self.energy), ('peak_power', self.peak_power)):
            if value is not None and value <= 0:
                raise ArgumentError(f'{field} must be > 0', name, field)
        if self.num_points < 16 or self.num_points % 2:
            raise ArgumentError('num_points must be even and at least 16', name, 'num_points')
        if self.alias not in (1, 2, 3):
            raise ArgumentError('alias must be 1, 2 or 3', name, 'alias')
        if self.repetition_rate is not None and self.repetition_rate <= 0:
            raise ArgumentError('repetition_rate must be > 0', name, 'repetition_rate')
        if isinstance(self.mode, dict) and not self.mode:
            raise ArgumentError('mode must name at least one mode', name, 'mode')
        return self

    @property
    def mode_amplitudes(self) -> Dict[int, complex]:
        """The launched modes and their normalized complex amplitudes."""
        if isinstance(self.mode, int):
            return {self.mode: 1.0 + 0.0j}
        norm = float(np.sqrt(sum(abs(complex(v)) ** 2 for v in self.mode.values())))
        return {int(k): complex(v) / norm for k, v in self.mode.items()}


@register_type
class Poling(TaggedModel):
    """The chi(2) domain pattern of a section, in nm along the section.

    Give exactly one of: `period` (constant), `z_knots` with `periods` (a local
    period interpolated between knots, the form an optimizer varies), or
    `domain_walls` (explicit positions). `duty_cycle` is the fraction of each
    period in the first (positive) domain. `inverted_first` starts in the
    inverted domain. `jitter_rms` displaces every wall by Gaussian noise (nm)
    drawn with `seed`, for fabrication-tolerance studies.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    period: Optional[float] = None
    z_knots: Optional[List[float]] = None
    periods: Optional[List[float]] = None
    domain_walls: Optional[List[float]] = None
    duty_cycle: float = 0.5
    interpolation: Literal['linear', 'spline'] = 'linear'
    inverted_first: bool = False
    jitter_rms: float = 0.0
    seed: Optional[int] = None

    @model_validator(mode='after')
    def _validate(self) -> 'Poling':
        name = 'Poling'
        forms = [self.period is not None, self.z_knots is not None, self.domain_walls is not None]
        if sum(forms) != 1:
            raise ArgumentError('give exactly one of period, z_knots/periods and domain_walls', name, 'period')
        if self.period is not None and self.period <= 0:
            raise ArgumentError('period must be > 0', name, 'period')
        if self.z_knots is not None:
            if self.periods is None or len(self.periods) != len(self.z_knots) or len(self.z_knots) < 2:
                raise ArgumentError('z_knots and periods need the same length, at least 2', name, 'periods')
            if any(p <= 0 for p in self.periods):
                raise ArgumentError('every period must be > 0', name, 'periods')
        elif self.periods is not None:
            raise ArgumentError('periods goes with z_knots', name, 'periods')
        if self.domain_walls is not None and any(
            b <= a for a, b in zip(self.domain_walls[:-1], self.domain_walls[1:])
        ):
            raise ArgumentError('domain_walls must be strictly increasing', name, 'domain_walls')
        if not 0 < self.duty_cycle < 1:
            raise ArgumentError('duty_cycle must be in (0, 1)', name, 'duty_cycle')
        if self.jitter_rms < 0:
            raise ArgumentError('jitter_rms must be >= 0', name, 'jitter_rms')
        return self


@register_type
class PulsedProcess(TaggedModel):
    """The nonlinear optics of one section for a pulsed simulation.

    Pass it as a section's `nonlinear=`. `modes` lists the tracked mode
    indices at the source wavelength (None tracks the source's own mode or
    modes). `tpa=None` turns two-photon absorption on wherever a material has
    a two-photon coefficient. The section's cross-sections are solved at
    `dispersion_samples` wavelengths (Chebyshev nodes over the source's
    `wavelength_range`); the tracked fields are expanded in patterns whose
    singular values stay above `pattern_threshold` relative to the first, at
    most `max_rank` of them.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    chi2: bool = True
    chi3: bool = True
    raman: bool = True
    raman_mode: Literal['dominant', 'full'] = 'dominant'
    tpa: Optional[bool] = None
    modes: Optional[List[int]] = None
    poling: Optional[Poling] = None
    dispersion_samples: int = 24
    pattern_threshold: float = 1e-3
    max_rank: int = 12

    @model_validator(mode='after')
    def _validate(self) -> 'PulsedProcess':
        name = 'PulsedProcess'
        if self.dispersion_samples < 4:
            raise ArgumentError('dispersion_samples must be at least 4', name, 'dispersion_samples')
        if not 0 <= self.pattern_threshold < 1:
            raise ArgumentError('pattern_threshold must be in [0, 1)', name, 'pattern_threshold')
        if self.max_rank < 1:
            raise ArgumentError('max_rank must be >= 1', name, 'max_rank')
        if self.modes is not None and (not self.modes or any(m < 0 for m in self.modes)):
            raise ArgumentError('modes must list mode indices >= 0', name, 'modes')
        return self


@register_type
class PulseResponse(TaggedModel):
    """The pulse at the recorded positions along the chain.

    `spectrum` is (records, modes, points), complex, in sqrt(J/Hz) on
    `frequency` (Hz); `wavelength` (nm) is the same axis. `z` (nm) are the
    record positions from the launch port. `energy` (records, modes) is in pJ.
    `junction_loss` is, per junction in chain order, the fraction of the
    arriving energy that left the tracked modes there (reflected or
    converted to untracked modes). `edge_energy` is, per record, the fraction
    of the energy in the outer 5 % of the time window: above about 1e-3 the
    pulse is wrapping around the window, and `num_points` should grow (the
    solver warns there).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    z: np.ndarray
    frequency: np.ndarray
    spectrum: np.ndarray
    modes: List[int]
    reference_index: int
    energy: np.ndarray
    junction_loss: List[float]
    edge_energy: np.ndarray
    domain_walls: List[float]
    steps: int
    rejected_steps: int
    rank_per_slice: List[int]
    samples_per_slice: List[int]
    repetition_rate: Optional[float] = None

    @property
    def wavelength(self) -> np.ndarray:
        return _C0 / self.frequency * 1e9

    @property
    def dnu(self) -> float:
        return float(self.frequency[1] - self.frequency[0])

    def _record(self, z: Optional[float]) -> int:
        if z is None:
            return len(self.z) - 1
        return int(np.argmin(np.abs(np.asarray(self.z) - z)))

    def energy_outside_window(self) -> float:
        """The largest fraction of the pulse energy, at any record, in the outer 5 % of
        the time window on either side. Above about 1e-3 the pulse is wrapping
        around the window: widen it (a narrower `wavelength_range` per point, or
        more `num_points`)."""
        return float(np.max(self.edge_energy)) if len(self.edge_energy) else 0.0

    def band_energy(
        self,
        wavelength_min: float,
        wavelength_max: float,
        z: Optional[float] = None,
        mode: Optional[int] = None,
    ) -> float:
        """Energy (pJ) between two wavelengths (nm), at the record nearest `z` (default: the output).

        `mode` selects one tracked mode by its index in `modes`'s list of
        mode numbers; None sums them.
        """
        lam = self.wavelength
        band = (lam >= min(wavelength_min, wavelength_max)) & (lam <= max(wavelength_min, wavelength_max))
        a = np.asarray(self.spectrum)[self._record(z)]
        if mode is not None:
            a = a[self.modes.index(mode)][None]
        return float(np.sum(np.abs(a[:, band]) ** 2) * self.dnu * 1e12)

    def band_power(
        self,
        wavelength_min: float,
        wavelength_max: float,
        z: Optional[float] = None,
        mode: Optional[int] = None,
        repetition_rate: Optional[float] = None,
    ) -> float:
        """Average power (W) of a pulse train in the band: band energy times the repetition rate."""
        rate = repetition_rate or self.repetition_rate
        if rate is None:
            raise ArgumentError(
                'band_power needs a repetition rate (Hz): give repetition_rate= here or on the PulsedSource',
                'band_power', 'repetition_rate',
            )
        return self.band_energy(wavelength_min, wavelength_max, z, mode) * 1e-12 * rate

    def psd(self, z: Optional[float] = None, mode: Optional[int] = None) -> np.ndarray:
        """Energy spectral density in dB relative to 1 pJ/nm on `wavelength`, summed over modes by default."""
        a = np.asarray(self.spectrum)[self._record(z)]
        if mode is not None:
            a = a[self.modes.index(mode)][None]
        per_hz = np.sum(np.abs(a) ** 2, axis=0)  # J/Hz
        per_nm = per_hz * self.frequency**2 / (_C0 * 1e9) * 1e12  # pJ/nm
        return 10 * np.log10(np.maximum(per_nm, 1e-300))

    def envelope(self, z: Optional[float] = None, mode: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(time in fs, complex envelope in sqrt(W)) of the pulse at the record nearest `z`."""
        a = np.asarray(self.spectrum)[self._record(z)]
        a = a[self.modes.index(mode) if mode is not None else 0]
        n = a.size
        dnu = self.dnu
        A = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(a))) * (n * dnu)
        t = (np.arange(n) - n // 2) / (n * dnu) * 1e15
        return t, A
