"""A chi(2) process's ODE tolerance (emode#971)."""

import json

import pytest

from emodeconnection import SHGProcess
from emodeconnection.types import ArgumentError, reconstruct


def test_absolute_tolerance_defaults_to_automatic():
    """None asks the server to scale the tolerance to the amplitudes."""
    assert SHGProcess(pump_wavelength=1550.0).solver_atol is None


def test_an_explicit_absolute_tolerance_is_kept():
    process = SHGProcess(pump_wavelength=1550.0, solver_atol=1e-13)
    assert process.solver_atol == 1e-13
    assert reconstruct(json.loads(json.dumps(process.model_dump()))) == process


def test_the_automatic_tolerance_round_trips():
    process = SHGProcess(pump_wavelength=1550.0)
    assert reconstruct(json.loads(json.dumps(process.model_dump()))).solver_atol is None


@pytest.mark.parametrize('atol', [0.0, -1e-9])
def test_a_non_positive_absolute_tolerance_is_rejected(atol):
    with pytest.raises(ArgumentError):
        SHGProcess(pump_wavelength=1550.0, solver_atol=atol)
