"""S-parameter models from EMode's S-matrices, for SAX and gdsfactory.

EMode answers an EME solve with an ``SMatrix``, and a wavelength sweep with
one per wavelength. This module turns them into the forms circuit tools read,
with no import of those tools:

- ``smatrix_to_sdict``: one ``SMatrix`` as a SAX S-dictionary.
- ``model_from_emode``: a tabulated SAX model over wavelength.
- ``sparameters_from_emode`` and ``write_sparameters_npz``: the npz form
  gdsfactory's plugins write, which ``gplugins.sax.read.model_from_npz`` reads.

Conventions:

- A key ``(a, b)`` holds the amplitude that leaves port ``b`` for unit
  amplitude entering port ``a``, which is ``S[b, a]`` of the matrix. A port's
  mode is named ``'<port>@<mode>'``, as SAX names the modes of a multimode port.
- EMode's wavelengths are in nanometers. SAX models and gdsfactory's npz files
  are in micrometers, so a model takes ``wl`` in micrometers and the npz file
  stores micrometers.
- Mode 0 of each port by default. ``modes=`` asks for more.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Union

import numpy as np

from .smatrix import SMatrix

#: SAX's S-dictionary: ``(input port, output port)`` to a complex amplitude, or an array of them over wavelength.
SDict = dict[tuple[str, str], Any]

NM_PER_UM = 1e3


def _port_modes(smatrix: SMatrix, modes: Sequence[int]) -> list[tuple[str, int, int]]:
    """``(name, mode, index into data)`` for each requested mode of each port."""
    if smatrix.port_names is None:
        raise ValueError(
            'this S-matrix has no port_names, so its ports cannot be named in an S-dictionary'
        )
    names = smatrix.port_names
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(
            f'port names {repeated} appear more than once (one port at several wavelengths?), '
            'so they cannot be told apart in an S-dictionary'
        )
    out = []
    for name, size, s in zip(names, smatrix.port_sizes, smatrix.port_slices):
        for mode in modes:
            if not 0 <= mode < size:
                raise ValueError(f'port {name!r} has {size} mode(s), so it has no mode {mode}')
            out.append((name, mode, s.start + mode))
    return out


def smatrix_to_sdict(smatrix: SMatrix, modes: Sequence[int] = (0,)) -> SDict:
    """One ``SMatrix`` as a SAX S-dictionary.

    Parameters
    ----------
    smatrix : SMatrix
        An S-matrix with ``port_names``, from ``EME()`` or ``get('S_matrix')``.
    modes : sequence of int
        The modes of each port to keep. Mode 0 by default.

    Returns
    -------
    dict
        ``{('o1@0', 'o2@0'): S[o2 mode 0, o1 mode 0], ...}``, every pair of
        the kept port modes, the input first.
    """
    ports = _port_modes(smatrix, modes)
    return {
        (f'{a}@{ma}', f'{b}@{mb}'): smatrix.data[ib, ia]
        for a, ma, ia in ports
        for b, mb, ib in ports
    }


def _samples(
    smatrices: Union[Sequence[SMatrix], Mapping[str, Any]],
    wavelengths: Sequence[float] | np.ndarray | None,
) -> tuple[list[SMatrix], np.ndarray]:
    """The S-matrices and their wavelengths in nanometers, sorted by wavelength."""
    if isinstance(smatrices, Mapping):
        sweep = smatrices
        if sweep.get('key') not in ('wavelength', 'wav'):
            raise ValueError(
                f"a sweep gives a model over wavelength only when it sweeps key='wavelength', not {sweep.get('key')!r}"
            )
        if 'S_matrix' not in sweep:
            raise ValueError("the sweep has no 'S_matrix' result")
        smatrices = list(sweep['S_matrix'])
        if wavelengths is None:
            wavelengths = sweep['values']
    smatrices = list(smatrices)
    if not smatrices:
        raise ValueError('no S-matrices given')
    if wavelengths is None:
        found = []
        for S in smatrices:
            wls = set(S.wavelengths or ())
            if len(wls) != 1:
                raise ValueError(
                    'pass wavelengths=, since these S-matrices do not each carry one wavelength'
                )
            found.append(wls.pop())
        wavelengths = found
    wl = np.asarray(wavelengths, dtype=float).ravel()
    if wl.size != len(smatrices):
        raise ValueError(f'{wl.size} wavelengths for {len(smatrices)} S-matrices')
    first = smatrices[0]
    for S in smatrices[1:]:
        if S.port_names != first.port_names or S.port_sizes != first.port_sizes:
            raise ValueError(
                f'the S-matrices have different ports: {first.port_names} {first.port_sizes} '
                f'and {S.port_names} {S.port_sizes}'
            )
    order = np.argsort(wl)
    return [smatrices[i] for i in order], wl[order]


def _tables(
    smatrices: Sequence[SMatrix], modes: Sequence[int]
) -> tuple[list[tuple[str, int]], dict[tuple[str, str], np.ndarray]]:
    """Every kept port mode, and each S-dictionary entry over the samples."""
    ports = _port_modes(smatrices[0], modes)
    names = [(a, ma) for a, ma, _ in ports]
    tables = {
        (f'{a}@{ma}', f'{b}@{mb}'): np.array([S.data[ib, ia] for S in smatrices], dtype=complex)
        for a, ma, ia in ports
        for b, mb, ib in ports
    }
    return names, tables


def sparameters_from_emode(
    smatrices: Union[Sequence[SMatrix], Mapping[str, Any]],
    wavelengths: Sequence[float] | np.ndarray | None = None,
    modes: Sequence[int] = (0,),
) -> dict[str, np.ndarray]:
    """S-parameters over wavelength in the form gdsfactory's plugins write to npz.

    Parameters
    ----------
    smatrices : sequence of SMatrix, or a wavelength sweep
        One ``SMatrix`` per wavelength, or the result of
        ``sweep(key='wavelength', result='S_matrix', ...)``.
    wavelengths : sequence of float, optional
        The wavelength of each S-matrix in nanometers. By default the sweep's
        values, or each S-matrix's own wavelength.
    modes : sequence of int
        The modes of each port to keep. Mode 0 by default.

    Returns
    -------
    dict
        ``'wavelengths'`` in micrometers, ascending, and one complex array per
        ``'<in port>@<mode>,<out port>@<mode>'`` key.
    """
    smatrices, wl = _samples(smatrices, wavelengths)
    _, tables = _tables(smatrices, modes)
    out: dict[str, np.ndarray] = {'wavelengths': wl / NM_PER_UM}
    out.update({f'{a},{b}': values for (a, b), values in tables.items()})
    return out


def write_sparameters_npz(
    path: Union[str, PathLike],
    smatrices: Union[Sequence[SMatrix], Mapping[str, Any]],
    wavelengths: Sequence[float] | np.ndarray | None = None,
    modes: Sequence[int] = (0,),
) -> Path:
    """Write ``sparameters_from_emode`` to an npz file, and return its path.

    ``gplugins.sax.read.model_from_npz(path)`` reads it back as a SAX model.
    The arguments are those of ``sparameters_from_emode``.
    """
    path = Path(path)
    np.savez(path, **sparameters_from_emode(smatrices, wavelengths, modes))
    return path


def _array_module(wl: Any) -> Any:
    """jax.numpy for a jax array or tracer (so the model runs under ``jax.jit``), numpy otherwise."""
    if type(wl).__module__.split('.')[0] in ('jax', 'jaxlib'):
        import jax.numpy as jnp

        return jnp
    return np


def model_from_emode(
    smatrices: Union[Sequence[SMatrix], Mapping[str, Any]],
    wavelengths: Sequence[float] | np.ndarray | None = None,
    modes: Sequence[int] = (0,),
) -> Callable[..., SDict]:
    """A tabulated SAX model of S-matrices over wavelength.

    The model interpolates each entry's magnitude and unwrapped phase
    linearly over wavelength. Interpolating the complex values themselves
    cuts the chord across the unit circle and loses accuracy on long delays.
    Outside the sampled band it holds the end values. The samples must be
    close enough that the phase of each entry moves by less than pi between
    neighbors.

    Parameters
    ----------
    smatrices : sequence of SMatrix, or a wavelength sweep
        One ``SMatrix`` per wavelength, or the result of
        ``sweep(key='wavelength', result='S_matrix', ...)``.
    wavelengths : sequence of float, optional
        The wavelength of each S-matrix in nanometers. By default the sweep's
        values, or each S-matrix's own wavelength.
    modes : sequence of int
        The modes of each port to keep. Mode 0 by default.

    Returns
    -------
    callable
        ``model(wl=...)`` with ``wl`` in micrometers, as SAX models take it
        (the band's center by default), returning a SAX S-dictionary. With
        one mode its keys are the bare port names, ``('o1', 'o2')``, as
        ``gplugins.sax.read.model_from_npz`` gives them. With several they
        are ``('o1@0', 'o2@1')``. Under ``jax.jit`` it computes with
        ``jax.numpy``.
    """
    smatrices, wl_nm = _samples(smatrices, wavelengths)
    _, tables = _tables(smatrices, modes)
    x = wl_nm / NM_PER_UM
    single = len(modes) == 1

    def bare(port: str) -> str:
        return port.rsplit('@', 1)[0] if single else port

    magnitude = {}
    phase = {}
    for (a, b), values in tables.items():
        key = (bare(a), bare(b))
        magnitude[key] = np.abs(values)
        phase[key] = np.unwrap(np.angle(values))
    center = float(0.5 * (x[0] + x[-1]))

    def model(wl: Any = center) -> SDict:
        xp = _array_module(wl)
        w = xp.asarray(wl)
        return {
            key: xp.interp(w, x, magnitude[key]) * xp.exp(1j * xp.interp(w, x, phase[key]))
            for key in magnitude
        }

    return model
