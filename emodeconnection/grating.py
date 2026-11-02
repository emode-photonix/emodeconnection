"""Section lengths for a grating along a section: uniform, chirped or noisy.

A grating alternates between two cross-sections, a tooth and a gap, once per
period. `grating_sections` gives the length of each of those sections in
order, so a grating is declared as one straight section per length,
alternating between the two profiles:

    for k, length in enumerate(grating_sections(period=320, count=200, chirp=4)):
        em.straight_section(name=f'grating_{k}', profile=('wide', 'narrow')[k % 2],
                            length=length)

It is computed here, in the client, and sends nothing to the solver. Its
`duty_cycle`, `jitter_rms` and `seed` mean what they mean for `Poling`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import ArgumentError

_NAME = 'grating_sections'


def grating_sections(
    period: float,
    count: int,
    duty_cycle: float = 0.5,
    chirp: float = 0.0,
    jitter_rms: float = 0.0,
    seed: Optional[int] = None,
) -> List[float]:
    """The lengths (nm) of the `2 count` sections of a grating, in order.

    Parameters
    ----------
    period : float
        The grating period (nm). With a chirp it is the mean period.
    count : int
        The number of periods.
    duty_cycle : float
        The fraction of each period in its first section, in (0, 1).
    chirp : float
        The change of the period (nm) from the first period to the last. The
        period changes linearly with the period's index, from
        `period - chirp / 2` to `period + chirp / 2`, so the grating stays
        `count * period` long.
    jitter_rms : float
        Gaussian noise (nm) added to the position of every wall between two
        sections, drawn with `seed`, for fabrication-tolerance studies. The two
        ends of the grating stay where they are.
    seed : int, optional
        The seed of the noise, so the same grating can be drawn again.

    Returns
    -------
    list of float
        The first section of period 0, its second section, the first section
        of period 1, and so on.
    """
    if not period > 0:
        raise ArgumentError('period must be > 0', _NAME, 'period')
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ArgumentError('count must be a whole number of periods, at least 1', _NAME, 'count')
    if not 0 < duty_cycle < 1:
        raise ArgumentError('duty_cycle must be in (0, 1)', _NAME, 'duty_cycle')
    if not jitter_rms >= 0:
        raise ArgumentError('jitter_rms must be >= 0', _NAME, 'jitter_rms')

    steps = np.linspace(-0.5, 0.5, count) if count > 1 else np.zeros(1)
    periods = period + chirp * steps
    if np.any(periods <= 0):
        raise ArgumentError('the chirp makes a period <= 0', _NAME, 'chirp')

    lengths = np.column_stack([duty_cycle * periods, (1 - duty_cycle) * periods]).ravel()
    if jitter_rms > 0:
        walls = np.cumsum(lengths)
        walls[:-1] += np.random.default_rng(seed).normal(0.0, jitter_rms, walls.size - 1)
        lengths = np.diff(walls, prepend=0.0)
        if np.any(lengths <= 0):
            raise ArgumentError(
                'the jitter made a section of no length: reduce jitter_rms', _NAME, 'jitter_rms'
            )
    return [float(v) for v in lengths]
