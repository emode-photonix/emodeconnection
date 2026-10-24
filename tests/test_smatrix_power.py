"""Power readouts of an `SMatrix`: `power()` and the weighted `insertion_loss_dB()`.

Amplitudes are not powers when the modes carry unequal power. The power leaving
column j is ``sum_i |S_ij|^2 P_i / P_j``, with P the per-mode real power.
"""

from __future__ import annotations

import numpy as np
import pytest

from emodeconnection.smatrix import SMatrix

POWER = np.array([1.0, 4.0])


def _lossless(power: np.ndarray) -> np.ndarray:
    """A lossless S-matrix between modes that carry `power`, ``P^-1/2 U P^1/2``."""
    unitary = np.array([[0.6, 0.8j], [0.8j, 0.6]])
    root = np.sqrt(power)
    return unitary * root[None, :] / root[:, None]


def _two_port(power: np.ndarray | None, data: np.ndarray | None = None) -> SMatrix:
    return SMatrix(
        port_sizes=(1, 1),
        data=_lossless(POWER) if data is None else data,
        port_names=('left', 'right'),
        mode_power=power,
    )


def test_power_weights_each_entry_by_the_ports_powers() -> None:
    s = _two_port(POWER)
    # |S_ij|^2 P_i / P_j recovers the unitary's |U_ij|^2 whatever the powers are
    np.testing.assert_allclose(s.power('left', 'right'), [[0.64]])
    np.testing.assert_allclose(s.power('right', 'left'), [[0.64]])
    np.testing.assert_allclose(s.power('left', 'left'), [[0.36]])
    # the plain amplitudes are off by P_i / P_j
    assert abs(s.transmission('left', 'right')[0, 0]) ** 2 == pytest.approx(0.64 * POWER[0] / POWER[1])


def test_a_lossless_column_weighted_sums_to_one() -> None:
    s = _two_port(POWER)
    for src, dst in (('left', 'right'), ('right', 'left')):
        np.testing.assert_allclose(s.power(src, dst) + s.power(src, src), 1.0)


def test_insertion_loss_counts_the_power_leaving_the_column() -> None:
    s = _two_port(POWER)
    assert s.insertion_loss_dB('left', 'right') == pytest.approx(-10 * np.log10(0.64))
    assert s.insertion_loss_dB('right', 'left') == pytest.approx(-10 * np.log10(0.64))
    # the unweighted amplitude would give a different number
    assert -10 * np.log10(abs(s.data[1, 0]) ** 2) != pytest.approx(-10 * np.log10(0.64))


def test_without_mode_power_the_readouts_are_the_plain_squared_amplitudes() -> None:
    s = _two_port(None)
    np.testing.assert_allclose(s.power('left', 'right'), np.abs(s.transmission('left', 'right')) ** 2)
    np.testing.assert_allclose(s.power('left', 'left'), np.abs(s.reflection('left')) ** 2)
    assert s.insertion_loss_dB('left', 'right') == pytest.approx(
        -10 * np.log10(abs(s.data[1, 0]) ** 2)
    )


def test_power_follows_the_mode_selection_of_a_multimode_port() -> None:
    data = np.arange(9).reshape(3, 3) * (1 + 1j)
    s = SMatrix(port_sizes=(2, 1), data=data, mode_power=np.array([1.0, 2.0, 5.0]))
    expected = np.abs(data[2:, :2]) ** 2 * 5.0 / np.array([1.0, 2.0])[None, :]
    np.testing.assert_allclose(s.power(0, 1), expected)
    col = np.abs(data[:, 1]) ** 2 * np.array([1.0, 2.0, 5.0]) / 2.0
    assert s.insertion_loss_dB(0, 1, mode=1) == pytest.approx(-10 * np.log10(col[2]))
