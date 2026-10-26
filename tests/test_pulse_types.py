"""The pulsed solver's wire types for a measured pulse: PulseMeasurement, the
measured PulsedSource and the PulseFit that the PulseResponse reports.

Each one crosses the wire through its `__data_type__` tag, and a source or a
response saved by an earlier version, before these fields existed, still loads.
"""

from __future__ import annotations

import base64
import json
import pickle
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from emodeconnection import PulseFit, PulseMeasurement, PulsedProcess, PulsedSource, PulseResponse
from emodeconnection.emodeclient import obj_hook
from emodeconnection.types import ArgumentError, reconstruct


def _over_the_wire(value: Any) -> Any:
    """What the client rebuilds from `value` after a trip through the protocol's
    JSON, with arrays base64-encoded as the solver sends them."""
    payload = json.dumps(value.model_dump(), default=_array)
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        data = base64.b64encode(np.ascontiguousarray(value).data).decode()
        return {'__ndarray__': data, 'dtype': str(value.dtype), 'shape': value.shape}
    raise TypeError(type(value))


def _spectral(**kw: Any) -> PulseMeasurement:
    wl = np.linspace(1500.0, 1600.0, 41)
    args: dict = dict(wavelength=wl, intensity=np.exp(-(((wl - 1550.0) / 15.0) ** 2)))
    args.update(kw)
    return PulseMeasurement(**args)


def _measured_source(**kw: Any) -> PulsedSource:
    args: dict = dict(
        center_wavelength=1550.0, shape='measured', measurement=_spectral(), energy=1.0
    )
    args.update(kw)
    return PulsedSource(**args)


# ---------------------------------------------------------------------------
# PulseMeasurement
# ---------------------------------------------------------------------------


def test_a_measurement_takes_numpy_arrays_and_defaults_to_the_smooth_fit() -> None:
    m = _spectral(phase=np.zeros(41))
    assert isinstance(m.wavelength, list) and len(m.wavelength) == 41
    assert m.time is None
    assert m.fit == 'smooth'
    assert m.min_fit_quality == 0.99
    assert m.domain == 'spectral'
    t = PulseMeasurement(time=np.linspace(-500, 500, 64), intensity=np.ones(64))
    assert t.domain == 'temporal'


def test_a_measurement_holds_exactly_one_domain() -> None:
    wl = np.linspace(1500.0, 1600.0, 41)
    with pytest.raises(ArgumentError, match='exactly one of wavelength and time'):
        PulseMeasurement(intensity=np.ones(41))
    with pytest.raises(ArgumentError, match='exactly one of wavelength and time'):
        PulseMeasurement(wavelength=wl, time=wl, intensity=np.ones(41))


def test_a_measurement_refuses_mismatched_or_bad_samples() -> None:
    wl = np.linspace(1500.0, 1600.0, 41)
    with pytest.raises(ArgumentError, match='same length'):
        PulseMeasurement(wavelength=wl, intensity=np.ones(40))
    with pytest.raises(ArgumentError, match='same length'):
        PulseMeasurement(wavelength=wl, intensity=np.ones(41), phase=np.ones(3))
    with pytest.raises(ArgumentError, match='at least 8'):
        PulseMeasurement(wavelength=wl[:5], intensity=np.ones(5))
    with pytest.raises(ArgumentError, match='finite'):
        PulseMeasurement(wavelength=wl, intensity=np.full(41, np.nan))
    with pytest.raises(ArgumentError, match='positive peak'):
        PulseMeasurement(wavelength=wl, intensity=np.zeros(41))
    with pytest.raises(ArgumentError, match='wavelength'):
        PulseMeasurement(wavelength=-wl, intensity=np.ones(41))
    with pytest.raises(ArgumentError, match='distinct'):
        PulseMeasurement(wavelength=np.r_[wl[:-1], wl[0]], intensity=np.ones(41))
    with pytest.raises(ArgumentError, match='min_fit_quality'):
        _spectral(min_fit_quality=1.5)
    with pytest.raises(ValidationError, match='fit'):
        _spectral(fit='lorentzian')


def test_a_measurement_round_trips_the_wire() -> None:
    m = _spectral(phase=np.linspace(0, 1, 41), fit='gaussian', min_fit_quality=0.9)
    back = _over_the_wire(m)
    assert isinstance(back, PulseMeasurement)
    assert back == m


# ---------------------------------------------------------------------------
# PulsedSource(shape='measured')
# ---------------------------------------------------------------------------


def test_a_measured_source_needs_a_measurement_and_no_analytic_shape() -> None:
    s = _measured_source()
    assert s.duration is None and s.measurement is not None
    with pytest.raises(ArgumentError, match='measurement'):
        PulsedSource(center_wavelength=1550.0, shape='measured', energy=1.0)
    with pytest.raises(ArgumentError, match='duration'):
        _measured_source(duration=100.0)
    with pytest.raises(ArgumentError, match='chirp'):
        _measured_source(chirp=1.0)
    with pytest.raises(ArgumentError, match='exactly one'):
        _measured_source(energy=None)
    with pytest.raises(ArgumentError, match="shape='measured'"):
        PulsedSource(center_wavelength=1550.0, duration=100.0, energy=1.0, measurement=_spectral())


def test_a_measured_source_round_trips_the_wire() -> None:
    s = _measured_source(peak_power=10.0, energy=None)
    back = _over_the_wire(s)
    assert isinstance(back, PulsedSource)
    assert isinstance(back.measurement, PulseMeasurement)
    assert back == s


def _older(model: Any, *missing: str) -> Any:
    """`model` as an earlier version pickled it, before the `missing` fields existed."""
    state = model.__getstate__()
    state['__dict__'] = {k: v for k, v in state['__dict__'].items() if k not in missing}
    old = type(model).__new__(type(model))
    old.__setstate__(state)
    return pickle.loads(pickle.dumps(old))


def test_a_source_saved_before_measured_pulses_still_loads() -> None:
    s = PulsedSource(center_wavelength=1550.0, duration=100.0, energy=1.0)
    old = _older(s, 'measurement')
    assert old.measurement is None
    assert old == s


# ---------------------------------------------------------------------------
# PulseFit on the PulseResponse
# ---------------------------------------------------------------------------


def _fit() -> PulseFit:
    return PulseFit(
        model='sech',
        quality=0.998,
        fwhm=101.0,
        center_wavelength=1551.0,
        gdd=1200.0,
        tod=-300.0,
        chirp=1.5,
        transform_limited=False,
        energy_outside=0.0,
        residual_peak=1549.0,
        residual_axis='wavelength',
    )


def _response(fit: PulseFit | None) -> PulseResponse:
    return PulseResponse(
        z=np.array([0.0, 1e6]),
        frequency=np.linspace(150e12, 250e12, 16),
        spectrum=np.zeros((2, 1, 16), dtype=complex),
        modes=[0],
        reference_index=0,
        energy=np.zeros((2, 1)),
        junction_loss=[],
        edge_energy=np.zeros(2),
        domain_walls=[],
        steps=1,
        rejected_steps=0,
        rank_per_slice=[1],
        samples_per_slice=[8],
        measured_fit=fit,
    )


def test_the_response_reports_the_fit_across_the_wire() -> None:
    back = _over_the_wire(_response(_fit()))
    assert isinstance(back, PulseResponse)
    assert isinstance(back.measured_fit, PulseFit)
    assert back.measured_fit == _fit()
    assert _over_the_wire(_response(None)).measured_fit is None


def test_a_response_saved_before_measured_pulses_still_loads() -> None:
    old = _older(_response(None), 'measured_fit')
    assert old.measured_fit is None


def test_dispersion_samples_default_to_the_servers_choice():
    # None lets the server size the sampling to the band (wider bands need more)
    assert PulsedProcess().dispersion_samples is None
    assert PulsedProcess(dispersion_samples=4).dispersion_samples == 4
    with pytest.raises(ArgumentError):
        PulsedProcess(dispersion_samples=3)
    back = _over_the_wire(PulsedProcess())
    assert isinstance(back, PulsedProcess)
    assert back.dispersion_samples is None


# ---------------------------------------------------------------------------
# The time window's empty stretch on the PulseResponse
# ---------------------------------------------------------------------------


def test_the_window_gap_crosses_the_wire() -> None:
    response = _response(None).model_copy(update={'window_gap': np.array([0.9, 0.02])})
    back = _over_the_wire(response)
    np.testing.assert_allclose(back.window_gap, [0.9, 0.02])
    assert back.smallest_window_gap() == pytest.approx(0.02)


def test_a_response_saved_before_the_window_gap_still_loads() -> None:
    old = _older(_response(None), 'window_gap')
    assert len(old.window_gap) == 0
    assert old.smallest_window_gap() is None
