"""Wire types of the pulsed nonlinear solver.

A pulsed simulation is declared with three objects and returns one:

- `PulsedSource`, the excitation: a femtosecond pulse (or a CW reference)
  launched into one port, declared with `settings(excitation=...)`. A
  measured pulse is given as a `PulseMeasurement` (sampled spectrum or
  temporal trace), which the solver fits before it launches it.
- `PulsedProcess`, attached to a section with its `nonlinear=` argument: which
  nonlinearities act there, which modes are tracked, and how the section's
  dispersion is sampled.
- `Poling`, the domain pattern of a poled section, as a constant period, a
  period that varies along the section, or explicit domain walls.
- `PulseResponse`, returned by `get('pulse_response')` after `EME()` or
  `pulse_propagate()`: the spectrum of every tracked mode at the recorded
  positions, with band energy and power computed here on the client. For a
  measured pulse it also carries the `PulseFit` the solver accepted.

Units, as in the rest of the API: wavelengths and lengths in nm, durations
in fs, energies in pJ, powers in W, repetition rates in Hz. Arrays returned
in `PulseResponse` are in SI (Hz, sqrt(J/Hz)) with their nm and fs axes
alongside.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator

from .types import ArgumentError, TaggedModel, register_type

_C0 = 299792458.0


class _LoadsOlderVersions:
    """Fills the fields an earlier version did not have when a pickled object is loaded.

    Simulation files store these objects pickled. A file written before a
    field existed has no value for it, and pydantic would then raise on first
    access, so the field's default is filled in instead.
    """

    def __setstate__(self, state: Dict) -> None:
        values = state.get('__dict__', {})
        for name, info in type(self).model_fields.items():
            if name not in values and not info.is_required():
                values[name] = info.get_default(call_default_factory=True)
        super().__setstate__(state)


def _sample_list(value):
    """A 1D array or sequence of samples as a plain list of floats."""
    if value is None:
        return None
    return np.asarray(value, dtype=float).ravel().tolist()


@register_type
class PulseMeasurement(TaggedModel):
    """A measured pulse, sampled in wavelength or in time, for `PulsedSource(shape='measured')`.

    Give exactly one of `wavelength` (nm), for a measured spectrum, and `time`
    (fs), for a temporal trace, with `intensity` on the same samples. A
    spectrum's intensity is per unit wavelength, as a spectrometer records
    it. The scale is arbitrary, since the source's `energy` or `peak_power`
    sets it. `phase` (rad), optional, is the retrieved spectral or temporal
    phase (from FROG or SPIDER, for example), in the convention of those
    retrievals: the group delay is d phase / d omega, so a positive quadratic
    spectral phase is a positive GDD, and the instantaneous angular frequency
    is omega_0 - d phase / dt. Without a phase the pulse is taken as transform
    limited, and the solver warns.

    The solver fits the samples before it launches the pulse. `fit='smooth'`
    (the default) is a smoothing spline of the intensity with a polynomial
    phase whose order (2 to 4) the fit chooses. `fit='sech'` or `'gaussian'`
    fits the analytic shapes of `PulsedSource` (width, center and chirp). The
    fit's quality is 1 minus its normalized residual: the R^2 of the
    intensity, less the intensity-weighted mean of 1 - cos of the phase error
    when a phase is given. A fit below `min_fit_quality` is refused with an
    error stating the quality reached and where the residual is largest.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    wavelength: Optional[List[float]] = None
    time: Optional[List[float]] = None
    intensity: List[float]
    phase: Optional[List[float]] = None
    fit: Literal['smooth', 'sech', 'gaussian'] = 'smooth'
    min_fit_quality: float = 0.99

    @field_validator('wavelength', 'time', 'intensity', 'phase', mode='before')
    @classmethod
    def _as_lists(cls, value):
        return _sample_list(value)

    @model_validator(mode='after')
    def _validate(self) -> 'PulseMeasurement':
        name = 'PulseMeasurement'
        if (self.wavelength is None) == (self.time is None):
            raise ArgumentError('give exactly one of wavelength and time', name, 'wavelength')
        axis_name = 'wavelength' if self.wavelength is not None else 'time'
        axis = np.asarray(self.wavelength if self.wavelength is not None else self.time)
        intensity = np.asarray(self.intensity)
        if axis.size != intensity.size:
            raise ArgumentError(f'{axis_name} and intensity must have the same length', name, 'intensity')
        if self.phase is not None and len(self.phase) != axis.size:
            raise ArgumentError(f'phase and {axis_name} must have the same length', name, 'phase')
        if axis.size < 8:
            raise ArgumentError('a measurement needs at least 8 samples', name, axis_name)
        values = [axis, intensity] + ([np.asarray(self.phase)] if self.phase is not None else [])
        if not all(np.all(np.isfinite(v)) for v in values):
            raise ArgumentError('every sample must be finite', name, axis_name)
        if self.wavelength is not None and np.any(axis <= 0):
            raise ArgumentError('every wavelength must be > 0', name, 'wavelength')
        if np.unique(axis).size != axis.size:
            raise ArgumentError(f'the {axis_name} samples must be distinct', name, axis_name)
        if np.max(intensity) <= 0:
            raise ArgumentError('the intensity needs a positive peak', name, 'intensity')
        if not 0 <= self.min_fit_quality <= 1:
            raise ArgumentError('min_fit_quality must be in [0, 1]', name, 'min_fit_quality')
        return self

    @property
    def domain(self) -> Literal['spectral', 'temporal']:
        """'spectral' for a sampled spectrum, 'temporal' for a sampled trace."""
        return 'spectral' if self.wavelength is not None else 'temporal'


@register_type
class PulseFit(TaggedModel):
    """The fit the solver accepted for a measured pulse, reported on the `PulseResponse`.

    `model` is the fitted model and `quality` its goodness of fit (see
    `PulseMeasurement`). The rest describe the launched pulse: `fwhm` (fs) its
    intensity FWHM, `center_wavelength` (nm) its spectral centroid, `gdd`
    (fs^2) and `tod` (fs^3) the second and third derivatives of its spectral
    phase in the convention of `PulseMeasurement`, and `chirp` the fitted C of
    a 'sech' or 'gaussian' model (None for 'smooth'). `transform_limited` is
    True when the measurement had no phase. `energy_outside` is the fraction
    of the fitted pulse's energy that fell outside the source's
    `wavelength_range` (or time window) and was cut. `residual_peak` is where
    the fit's residual is largest, on the measurement's own axis
    (`residual_axis`, 'wavelength' in nm or 'time' in fs).
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    model: Literal['smooth', 'sech', 'gaussian']
    quality: float
    fwhm: float
    center_wavelength: float
    gdd: float
    tod: float
    chirp: Optional[float] = None
    transform_limited: bool
    energy_outside: float = 0.0
    residual_peak: float
    residual_axis: Literal['wavelength', 'time']


@register_type
class PulsedSource(_LoadsOlderVersions, TaggedModel):
    """A pulse launched into one port of the circuit.

    `mode` is the tracked mode index that carries the pulse, or a dict
    `{mode: complex amplitude}` that splits it between tracked modes (the
    amplitudes are normalized, so only their ratios matter). Give exactly one
    of `energy` (pJ) and `peak_power` (W). `duration` is the intensity FWHM in
    fs. `chirp` is the dimensionless C of exp(j C t^2 / (2 T0^2)): positive
    raises the instantaneous frequency across the pulse. `shape='cw'` launches
    a continuous wave of `peak_power` at `center_wavelength` (no `duration`).
    `shape='measured'` launches the `measurement` (a `PulseMeasurement`),
    fitted by the solver and scaled to `energy` or `peak_power`, with no
    `duration` or `chirp` (the measurement carries both). A measured spectrum
    sits at its own wavelengths, and `center_wavelength` stays the wavelength
    at which the circuit is refined. A measured trace is centered on
    `center_wavelength`.

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
    shape: Literal['sech', 'gaussian', 'cw', 'measured'] = 'sech'
    chirp: float = 0.0
    measurement: Optional[PulseMeasurement] = None
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
        if self.measurement is not None and self.shape != 'measured':
            raise ArgumentError("a measurement goes with shape='measured'", name, 'measurement')
        if self.shape == 'cw':
            if self.peak_power is None or self.energy is not None:
                raise ArgumentError("a 'cw' source takes peak_power (W) and no energy", name, 'peak_power')
        elif self.shape == 'measured':
            if self.measurement is None:
                raise ArgumentError("shape='measured' needs a measurement (PulseMeasurement)", name, 'measurement')
            if self.duration is not None:
                raise ArgumentError('a measured source takes no duration: the measurement sets it', name, 'duration')
            if self.chirp != 0:
                raise ArgumentError('a measured source takes no chirp: the measurement sets it', name, 'chirp')
            if (self.energy is None) == (self.peak_power is None):
                raise ArgumentError('give exactly one of energy (pJ) and peak_power (W)', name, 'energy')
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
    period interpolated between knots), `domain_walls` (explicit positions),
    or `domain_lengths` (the length of every domain in order, each one free in
    an inverse design). `duty_cycle` is the fraction of each period in the
    first (positive) domain. `inverted_first` starts in the inverted domain.
    `jitter_rms` displaces every wall by Gaussian noise (nm) drawn with
    `seed`, for fabrication-tolerance studies.

    The walls of `domain_lengths` are its running sum. A wall at or past the
    end of the section is dropped, and when the domains end before the
    section does, the sign flips once more and the last domain runs to the
    end. `walls(section_length)` gives the walls of any form, and
    `to_domain_lengths(section_length)` converts any form to domain lengths.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    period: Optional[float] = None
    z_knots: Optional[List[float]] = None
    periods: Optional[List[float]] = None
    domain_walls: Optional[List[float]] = None
    domain_lengths: Optional[List[float]] = None
    duty_cycle: float = 0.5
    interpolation: Literal['linear', 'spline'] = 'linear'
    inverted_first: bool = False
    jitter_rms: float = 0.0
    seed: Optional[int] = None

    @model_validator(mode='after')
    def _validate(self) -> 'Poling':
        name = 'Poling'
        forms = [
            self.period is not None,
            self.z_knots is not None,
            self.domain_walls is not None,
            self.domain_lengths is not None,
        ]
        if sum(forms) != 1:
            raise ArgumentError(
                'give exactly one of period, z_knots/periods, domain_walls and domain_lengths', name, 'period'
            )
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
        if self.domain_lengths is not None:
            if not self.domain_lengths:
                raise ArgumentError('domain_lengths needs at least one domain', name, 'domain_lengths')
            if any(not v > 0 for v in self.domain_lengths):
                raise ArgumentError('every domain length must be > 0', name, 'domain_lengths')
        if not 0 < self.duty_cycle < 1:
            raise ArgumentError('duty_cycle must be in (0, 1)', name, 'duty_cycle')
        if self.jitter_rms < 0:
            raise ArgumentError('jitter_rms must be >= 0', name, 'jitter_rms')
        return self

    @property
    def total_length(self) -> Optional[float]:
        """The summed length (nm) of `domain_lengths`, None for the other forms."""
        return None if self.domain_lengths is None else float(sum(self.domain_lengths))

    def walls(self, section_length: float) -> List[float]:
        """The domain walls (nm from the start of the section) over a section of `section_length` nm."""
        if self.period is not None:
            walls = _walls_from_period(section_length, self.period, self.duty_cycle)
        elif self.z_knots is not None:
            walls = _walls_from_knots(
                section_length, self.z_knots, self.periods or [], self.duty_cycle, self.interpolation
            )
        elif self.domain_lengths is not None:
            walls = [float(w) for w in np.cumsum(self.domain_lengths) if w < section_length]
        else:
            walls = [float(w) for w in self.domain_walls or [] if 0 < w < section_length]
        if self.jitter_rms > 0:
            walls = _jitter(walls, self.jitter_rms, self.seed)
        return walls

    def to_domain_lengths(self, section_length: float) -> List[float]:
        """The length (nm) of every domain over a section of `section_length` nm.

        The lengths sum to the section length, so `Poling(domain_lengths=...)`
        with the same `inverted_first` has the same walls. A run can start
        from a fixed or chirped poling and free every domain from there.
        """
        edges = [0.0, *self.walls(section_length), float(section_length)]
        return [float(b - a) for a, b in zip(edges[:-1], edges[1:])]


def _walls_from_phase(z: np.ndarray, phase: np.ndarray, duty_cycle: float) -> List[float]:
    """Walls where the grating phase crosses 2 pi (k + D) (to -) and 2 pi (k + 1) (to +)."""
    total = phase[-1] / (2 * np.pi)
    targets = []
    k = 0
    while True:
        down, up = 2 * np.pi * (k + duty_cycle), 2 * np.pi * (k + 1)
        if down >= phase[-1] * (1 - 1e-12):
            break
        targets.append(down)
        if up >= phase[-1] * (1 - 1e-12):
            break
        targets.append(up)
        k += 1
        if k > total + 2:
            break
    return [float(v) for v in np.interp(targets, phase, z)]


def _walls_from_period(length: float, period: float, duty_cycle: float) -> List[float]:
    """Walls of a constant-period grating over [0, length]."""
    walls = []
    k = 0
    while True:
        for offset in (duty_cycle, 1.0):
            w = (k + offset) * period
            if w >= length * (1 - 1e-12):
                return walls
            walls.append(w)
        k += 1


def _walls_from_knots(
    length: float,
    z_knots: List[float],
    periods: List[float],
    duty_cycle: float,
    interpolation: str,
    samples: int = 200001,
) -> List[float]:
    """Walls of a grating whose local period is interpolated between (z, period) knots.

    The local grating phase, the integral of 2 pi / period(z), is integrated on
    `samples` points and a wall placed at every half cycle (shifted by the duty
    cycle).
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import CubicSpline

    zk = np.asarray(z_knots, dtype=float)
    pk = np.asarray(periods, dtype=float)
    z = np.linspace(0.0, length, samples)
    local = np.interp(z, zk, pk) if interpolation == 'linear' else CubicSpline(zk, pk)(z)
    phase = cumulative_trapezoid(2 * np.pi / local, z, initial=0.0)
    return _walls_from_phase(z, phase, duty_cycle)


def _jitter(walls: List[float], rms: float, seed: Optional[int]) -> List[float]:
    """Walls displaced by Gaussian noise of `rms`, refusing noise that reorders two walls."""
    w = np.asarray(walls, dtype=float)
    out = w + np.random.default_rng(seed).normal(0.0, rms, size=w.shape)
    if np.any(np.diff(out) <= 0):
        raise ArgumentError('the jitter reordered two domain walls: reduce jitter_rms', 'Poling', 'jitter_rms')
    return [float(v) for v in out]


@register_type
class PulsedProcess(TaggedModel):
    """The nonlinear optics of one section for a pulsed simulation.

    Pass it as a section's `nonlinear=`. `modes` lists the tracked mode
    indices at the source wavelength (None tracks the source's own mode or
    modes). `tpa=None` turns two-photon absorption on wherever a material has
    a two-photon coefficient. The section's cross-sections are solved at
    `dispersion_samples` wavelengths (Chebyshev nodes over the source's
    `wavelength_range`), by default as many as the band's width needs. The
    tracked fields are expanded in patterns whose
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
    dispersion_samples: Optional[int] = None
    pattern_threshold: float = 1e-3
    max_rank: int = 12

    @model_validator(mode='after')
    def _validate(self) -> 'PulsedProcess':
        name = 'PulsedProcess'
        if self.dispersion_samples is not None and self.dispersion_samples < 4:
            raise ArgumentError('dispersion_samples must be at least 4', name, 'dispersion_samples')
        if not 0 <= self.pattern_threshold < 1:
            raise ArgumentError('pattern_threshold must be in [0, 1)', name, 'pattern_threshold')
        if self.max_rank < 1:
            raise ArgumentError('max_rank must be >= 1', name, 'max_rank')
        if self.modes is not None and (not self.modes or any(m < 0 for m in self.modes)):
            raise ArgumentError('modes must list mode indices >= 0', name, 'modes')
        return self


@register_type
class PulseResponse(_LoadsOlderVersions, TaggedModel):
    """The pulse at the recorded positions along the chain.

    `spectrum` is (records, modes, points), complex, in sqrt(J/Hz) on
    `frequency` (Hz); `wavelength` (nm) is the same axis. `z` (nm) are the
    record positions from the launch port. `energy` (records, modes) is in pJ.
    `junction_loss` is, per junction in chain order, the fraction of the
    arriving energy that left the tracked modes there (reflected or
    converted to untracked modes). `edge_energy` is, per record, the fraction
    of the energy in the outer 5 % of the time window: above about 1e-3 the
    pulse is wrapping around the window, and `num_points` should grow (the
    solver warns there). `measured_fit` is the `PulseFit` of a measured
    source (None for an analytic one).
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
    measured_fit: Optional[PulseFit] = None

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
