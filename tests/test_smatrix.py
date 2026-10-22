"""``SMatrix.mode_power`` and the power-weighted ``unitarity_error``.

A lossless S-matrix conserves power, ``S^H P S = P`` with ``P`` the diagonal
of per-mode real powers. Only when every mode carries unit power does that
reduce to ``S^H S = I``.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

from emodeconnection.emodeclient import obj_hook
from emodeconnection.smatrix import SMatrix
from emodeconnection.types import reconstruct


class _WireEncoder(json.JSONEncoder):
    """The solver's encoding of an ndarray, as the wire contract defines it."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            data = base64.b64encode(np.ascontiguousarray(o).data).decode()
            return {"__ndarray__": data, "dtype": str(o.dtype), "shape": o.shape}
        return super().default(o)


def _lossless(power: np.ndarray) -> np.ndarray:
    """A lossless S-matrix between modes that carry `power`, ``P^-1/2 U P^1/2``."""
    unitary = np.array([[0.6, 0.8j], [0.8j, 0.6]])
    root = np.sqrt(power)
    return unitary * root[None, :] / root[:, None]


def test_without_mode_power_the_error_is_unchanged() -> None:
    data = _lossless(np.array([1.0, 4.0]))
    s = SMatrix(port_sizes=(1, 1), data=data)
    expected = float(np.linalg.norm(data.conj().T @ data - np.eye(2)))
    assert s.mode_power is None
    assert s.unitarity_error() == expected
    assert s.unitarity_error() > 0.5


def test_mode_power_makes_a_lossless_matrix_conserve_power() -> None:
    power = np.array([1.0, 4.0])
    s = SMatrix(port_sizes=(1, 1), data=_lossless(power), mode_power=power)
    assert s.unitarity_error() == pytest.approx(0.0, abs=1e-12)


def test_a_mode_that_carries_no_power_is_not_counted() -> None:
    # A lossy mode (zero real power) may have any amplitude without breaking
    # the power balance of the modes that do carry power.
    data = np.zeros((3, 3), dtype=complex)
    data[:2, :2] = _lossless(np.array([1.0, 4.0]))
    data[2, :] = [3.0, 5.0, 7.0]
    s = SMatrix(port_sizes=(2, 1), data=data, mode_power=np.array([1.0, 4.0, 0.0]))
    assert s.unitarity_error() == pytest.approx(0.0, abs=1e-12)


def test_the_weighted_error_is_normalized_by_size() -> None:
    # The same 10% gain on every mode gives the same error at any size.
    errors = []
    for n in (2, 8):
        s = SMatrix(port_sizes=(n // 2, n // 2), data=np.sqrt(1.1) * np.eye(n), mode_power=np.ones(n))
        errors.append(s.unitarity_error())
    assert errors[0] == pytest.approx(errors[1])
    assert errors[0] == pytest.approx(0.1)


def test_the_weighted_error_does_not_depend_on_the_power_scale() -> None:
    data = np.sqrt(1.1) * np.eye(2)
    unit = SMatrix(port_sizes=(1, 1), data=data, mode_power=np.ones(2))
    milliwatt = SMatrix(port_sizes=(1, 1), data=data, mode_power=1e-3 * np.ones(2))
    assert milliwatt.unitarity_error() == pytest.approx(unit.unitarity_error())


def test_modes_that_carry_no_power_do_not_dilute_the_error() -> None:
    # Two power-carrying modes with a 10% gain, plus two lossy modes.
    data = np.zeros((4, 4), dtype=complex)
    data[[0, 2], [0, 2]] = np.sqrt(1.1)
    s = SMatrix(port_sizes=(2, 2), data=data, mode_power=np.array([1.0, 0.0, 1.0, 0.0]))
    assert s.unitarity_error() == pytest.approx(0.1)


def test_mode_power_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="mode_power"):
        SMatrix(port_sizes=(1, 1), data=np.eye(2), mode_power=np.ones(3))


def test_an_smatrix_with_mode_power_round_trips_over_the_wire() -> None:
    power = np.array([1.0, 4.0])
    s = SMatrix(port_sizes=(1, 1), data=_lossless(power), mode_power=power, port_names=("a", "b"))
    sent = json.dumps(s.model_dump(), cls=_WireEncoder)
    received = reconstruct(json.loads(sent, object_hook=obj_hook))
    assert isinstance(received, SMatrix)
    np.testing.assert_array_equal(received.mode_power, power)
    np.testing.assert_array_equal(received.data, s.data)
    assert received.unitarity_error() == pytest.approx(0.0, abs=1e-12)


def test_mode_power_sent_as_a_list_is_accepted() -> None:
    s = SMatrix(port_sizes=(1, 1), data=np.eye(2), mode_power=[1.0, 1.0])
    assert isinstance(s.mode_power, np.ndarray)
    assert s.unitarity_error() == pytest.approx(0.0)
