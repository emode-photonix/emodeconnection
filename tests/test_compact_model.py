"""SAX and npz exports of an ``SMatrix`` (``emode.compact_model``).

A SAX S-dictionary maps ``(input port, output port)`` to the complex amplitude
that leaves the output port for unit amplitude in the input port, so the key
``('o1@0', 'o2@0')`` holds ``data[row of o2 mode 0, column of o1 mode 0]``.
The npz form is the one gdsfactory's plugins write: a ``wavelengths`` array
in micrometers and one complex array per ``'o1@0,o2@0'`` key.
"""

from __future__ import annotations

import numpy as np
import pytest

from emode.compact_model import model_from_emode, smatrix_to_sdict, sparameters_from_emode, write_sparameters_npz
from emode.smatrix import SMatrix
from emode.sweep import SweepResult


def _two_port_two_mode() -> SMatrix:
    """Ports o1 and o2 with two modes each. Entry (row, column) = 10 * row + column + 1j."""
    data = np.array([[10 * r + c + 1j for c in range(4)] for r in range(4)])
    return SMatrix(port_sizes=(2, 2), data=data, port_names=("o1", "o2"), wavelengths=(1550.0, 1550.0))


# ---------------------------------------------------------------- smatrix_to_sdict


def test_sdict_has_mode_zero_keys_by_default() -> None:
    sdict = smatrix_to_sdict(_two_port_two_mode())
    assert set(sdict) == {
        ("o1@0", "o1@0"),
        ("o1@0", "o2@0"),
        ("o2@0", "o1@0"),
        ("o2@0", "o2@0"),
    }


def test_sdict_key_is_input_then_output() -> None:
    S = _two_port_two_mode()
    sdict = smatrix_to_sdict(S)
    # o1 mode 0 is column 0, o2 mode 0 is row 2
    assert sdict[("o1@0", "o2@0")] == S.data[2, 0]
    assert sdict[("o2@0", "o1@0")] == S.data[0, 2]
    assert sdict[("o1@0", "o1@0")] == S.data[0, 0]


def test_sdict_modes_selects_every_requested_mode() -> None:
    S = _two_port_two_mode()
    sdict = smatrix_to_sdict(S, modes=(0, 1))
    assert len(sdict) == 16
    # o1 mode 1 is column 1, o2 mode 1 is row 3
    assert sdict[("o1@1", "o2@1")] == S.data[3, 1]
    assert sdict[("o2@0", "o1@1")] == S.data[1, 2]


def test_sdict_needs_port_names() -> None:
    S = SMatrix(port_sizes=(1, 1), data=np.eye(2, dtype=complex))
    with pytest.raises(ValueError, match="port_names"):
        smatrix_to_sdict(S)


def test_sdict_refuses_a_mode_a_port_does_not_have() -> None:
    S = SMatrix(port_sizes=(2, 1), data=np.eye(3, dtype=complex), port_names=("o1", "o2"))
    with pytest.raises(ValueError, match="o2"):
        smatrix_to_sdict(S, modes=(0, 1))


def test_sdict_refuses_repeated_port_names() -> None:
    # one port name at two wavelengths cannot become one SAX port
    S = SMatrix(
        port_sizes=(1, 1),
        data=np.eye(2, dtype=complex),
        port_names=("o1", "o1"),
        wavelengths=(1550.0, 775.0),
    )
    with pytest.raises(ValueError, match="o1"):
        smatrix_to_sdict(S)


# ---------------------------------------------------------------- a tabulated model

N_EFF = 2.4
LENGTH_UM = 10.0


def _delay_line(wl_um: np.ndarray) -> np.ndarray:
    return np.exp(2j * np.pi * N_EFF * LENGTH_UM / wl_um)


def _delay_line_smatrices(wl_nm: np.ndarray) -> list[SMatrix]:
    """A lossless reflectionless delay line ``exp(i 2 pi n L / wl)`` from o1 to o2."""
    out = []
    for w in wl_nm:
        t = _delay_line(np.array(w / 1e3))
        data = np.array([[0.0, t], [t, 0.0]], dtype=complex)
        out.append(SMatrix(port_sizes=(1, 1), data=data, port_names=("o1", "o2"), wavelengths=(w, w)))
    return out


WL_NM = np.linspace(1500.0, 1600.0, 51)
MIDPOINTS_UM = 0.5 * (WL_NM[1:] + WL_NM[:-1]) / 1e3


def test_model_reproduces_a_delay_line_between_samples() -> None:
    model = model_from_emode(_delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    s21 = np.asarray(model(wl=MIDPOINTS_UM)[("o1", "o2")])
    assert np.max(np.abs(s21 - _delay_line(MIDPOINTS_UM))) < 1e-3


def test_complex_interpolation_would_miss_the_delay_line() -> None:
    # The phase steps about 0.13 rad between samples, so interpolating the
    # complex values themselves cuts the chord. The test above tells the two apart.
    t = _delay_line(WL_NM / 1e3)
    chord = np.interp(MIDPOINTS_UM, WL_NM / 1e3, t.real) + 1j * np.interp(MIDPOINTS_UM, WL_NM / 1e3, t.imag)
    assert np.max(np.abs(chord - _delay_line(MIDPOINTS_UM))) > 1e-3


def test_model_returns_samples_exactly() -> None:
    model = model_from_emode(_delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    s21 = np.asarray(model(wl=WL_NM / 1e3)[("o1", "o2")])
    np.testing.assert_allclose(s21, _delay_line(WL_NM / 1e3), atol=1e-12)


def test_model_reads_wavelengths_from_the_smatrices() -> None:
    model = model_from_emode(_delay_line_smatrices(WL_NM))
    s21 = np.asarray(model(wl=MIDPOINTS_UM)[("o1", "o2")])
    assert np.max(np.abs(s21 - _delay_line(MIDPOINTS_UM))) < 1e-3


def test_model_sorts_its_samples() -> None:
    model = model_from_emode(_delay_line_smatrices(WL_NM[::-1]), wavelengths=WL_NM[::-1])
    s21 = np.asarray(model(wl=MIDPOINTS_UM)[("o1", "o2")])
    assert np.max(np.abs(s21 - _delay_line(MIDPOINTS_UM))) < 1e-3


def test_model_takes_a_wavelength_sweep() -> None:
    sweep = SweepResult(data={"key": "wavelength", "values": WL_NM, "S_matrix": _delay_line_smatrices(WL_NM)})
    model = model_from_emode(sweep)
    s21 = np.asarray(model(wl=MIDPOINTS_UM)[("o1", "o2")])
    assert np.max(np.abs(s21 - _delay_line(MIDPOINTS_UM))) < 1e-3


def test_model_refuses_a_sweep_of_another_key() -> None:
    sweep = SweepResult(data={"key": "width", "values": WL_NM, "S_matrix": _delay_line_smatrices(WL_NM)})
    with pytest.raises(ValueError, match="wavelength"):
        model_from_emode(sweep)


def test_model_needs_wavelengths() -> None:
    S = [SMatrix(port_sizes=(1, 1), data=np.eye(2, dtype=complex), port_names=("o1", "o2"))] * 2
    with pytest.raises(ValueError, match="wavelengths"):
        model_from_emode(S)


def test_model_refuses_mismatched_wavelengths() -> None:
    with pytest.raises(ValueError, match="wavelengths"):
        model_from_emode(_delay_line_smatrices(WL_NM), wavelengths=WL_NM[:-1])


def test_model_refuses_smatrices_with_different_ports() -> None:
    S = _delay_line_smatrices(WL_NM[:2])
    S[1] = SMatrix(port_sizes=(1, 1), data=S[1].data, port_names=("a", "b"), wavelengths=S[1].wavelengths)
    with pytest.raises(ValueError, match="port"):
        model_from_emode(S, wavelengths=WL_NM[:2])


def test_model_default_wavelength_is_the_band_center() -> None:
    model = model_from_emode(_delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    np.testing.assert_allclose(np.asarray(model()[("o1", "o2")]), _delay_line(np.array(1.55)), atol=1e-3)


def test_model_keys_carry_the_mode_when_several_modes_are_asked() -> None:
    model = model_from_emode([_two_port_two_mode()] * 2, wavelengths=[1500.0, 1600.0], modes=(0, 1))
    sdict = model(wl=1.55)
    assert ("o1@1", "o2@0") in sdict
    np.testing.assert_allclose(np.asarray(sdict[("o1@1", "o2@0")]), _two_port_two_mode().data[2, 1])


def test_model_keys_are_bare_ports_for_one_mode() -> None:
    model = model_from_emode([_two_port_two_mode()] * 2, wavelengths=[1500.0, 1600.0])
    assert set(model(wl=1.55)) == {("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")}


# ---------------------------------------------------------------- the npz form


def test_sparameters_use_gdsfactory_keys_and_micrometers() -> None:
    sp = sparameters_from_emode(_delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    assert set(sp) == {"wavelengths", "o1@0,o1@0", "o1@0,o2@0", "o2@0,o1@0", "o2@0,o2@0"}
    np.testing.assert_allclose(sp["wavelengths"], WL_NM / 1e3)
    np.testing.assert_allclose(sp["o1@0,o2@0"], _delay_line(WL_NM / 1e3))


def test_npz_round_trips(tmp_path) -> None:
    path = write_sparameters_npz(tmp_path / "delay.npz", _delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    with np.load(path) as sp:
        assert sorted(sp.keys()) == sorted(["wavelengths", "o1@0,o1@0", "o1@0,o2@0", "o2@0,o1@0", "o2@0,o2@0"])
        np.testing.assert_allclose(sp["wavelengths"], WL_NM / 1e3)
        np.testing.assert_allclose(sp["o1@0,o2@0"], _delay_line(WL_NM / 1e3))
        assert np.iscomplexobj(sp["o1@0,o2@0"])


def test_npz_loads_in_gplugins_sax(tmp_path) -> None:
    read = pytest.importorskip("gplugins.sax.read")
    path = write_sparameters_npz(tmp_path / "delay.npz", _delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    theirs = read.model_from_npz(path)
    ours = model_from_emode(_delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    # At the samples both are exact. Between them gplugins interpolates the
    # complex values, so only the samples are compared.
    wl = WL_NM / 1e3
    for key in [("o1", "o2"), ("o2", "o1"), ("o1", "o1")]:
        np.testing.assert_allclose(np.asarray(theirs(wl=wl)[key]), np.asarray(ours(wl=wl)[key]), atol=1e-6)


def test_model_runs_under_jax_jit() -> None:
    jax = pytest.importorskip("jax")
    model = model_from_emode(_delay_line_smatrices(WL_NM), wavelengths=WL_NM)
    s21 = jax.jit(lambda wl: model(wl=wl)[("o1", "o2")])(MIDPOINTS_UM)
    assert np.max(np.abs(np.asarray(s21) - _delay_line(MIDPOINTS_UM))) < 1e-3
