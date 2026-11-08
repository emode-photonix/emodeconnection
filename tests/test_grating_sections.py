"""`grating_sections`: the section lengths of a uniform, chirped or noisy grating.

A grating of `count` periods is `2 count` sections, each period a first part
(`duty_cycle` of it) and a second part. The lengths are what the caller
declares, one section per length, alternating between its two profiles.
"""

from __future__ import annotations

import numpy as np
import pytest

from emode import grating_sections
from emode.types import ArgumentError


def _periods(lengths: list[float]) -> np.ndarray:
    pairs = np.asarray(lengths).reshape(-1, 2)
    return pairs.sum(axis=1)


def test_a_uniform_grating_alternates_its_two_parts() -> None:
    lengths = grating_sections(period=320.0, count=5)
    assert len(lengths) == 10
    assert lengths == pytest.approx([160.0] * 10)


def test_the_duty_cycle_is_the_first_part_of_every_period() -> None:
    lengths = grating_sections(period=400.0, count=3, duty_cycle=0.3)
    assert lengths == pytest.approx([120.0, 280.0] * 3)
    assert sum(lengths) == pytest.approx(1200.0)


def test_a_chirp_changes_the_period_linearly_from_first_to_last() -> None:
    """The period runs from period - chirp / 2 to period + chirp / 2, so its mean is `period`."""
    lengths = grating_sections(period=320.0, count=11, chirp=10.0)
    periods = _periods(lengths)
    assert periods == pytest.approx(np.linspace(315.0, 325.0, 11))
    assert sum(lengths) == pytest.approx(11 * 320.0)
    first, second = np.asarray(lengths).reshape(-1, 2).T
    assert first == pytest.approx(0.5 * periods)
    assert second == pytest.approx(0.5 * periods)


def test_a_negative_chirp_shortens_the_period_along_the_grating() -> None:
    periods = _periods(grating_sections(period=320.0, count=3, chirp=-4.0))
    assert periods == pytest.approx([322.0, 320.0, 318.0])


def test_jitter_moves_every_inner_wall_and_keeps_the_ends() -> None:
    uniform = np.cumsum(grating_sections(period=320.0, count=2000))
    noisy = np.cumsum(grating_sections(period=320.0, count=2000, jitter_rms=3.0, seed=1))
    assert noisy[-1] == pytest.approx(uniform[-1])
    moved = noisy[:-1] - uniform[:-1]
    assert np.all(moved != 0)
    assert np.sqrt(np.mean(moved**2)) == pytest.approx(3.0, rel=0.05)
    assert abs(np.mean(moved)) < 0.3


def test_a_seed_draws_the_same_jitter_again() -> None:
    a = grating_sections(period=320.0, count=50, jitter_rms=2.0, seed=7)
    b = grating_sections(period=320.0, count=50, jitter_rms=2.0, seed=7)
    c = grating_sections(period=320.0, count=50, jitter_rms=2.0, seed=8)
    assert a == b
    assert a != c


def test_a_single_period_has_two_sections() -> None:
    assert grating_sections(period=300.0, count=1, chirp=5.0) == pytest.approx([150.0, 150.0])


@pytest.mark.parametrize(
    'kwargs, parameter',
    [
        ({'period': 0.0, 'count': 10}, 'period'),
        ({'period': -1.0, 'count': 10}, 'period'),
        ({'period': 320.0, 'count': 0}, 'count'),
        ({'period': 320.0, 'count': 2.5}, 'count'),
        ({'period': 320.0, 'count': 10, 'duty_cycle': 0.0}, 'duty_cycle'),
        ({'period': 320.0, 'count': 10, 'duty_cycle': 1.0}, 'duty_cycle'),
        ({'period': 320.0, 'count': 10, 'chirp': 640.0}, 'chirp'),
        ({'period': 320.0, 'count': 10, 'jitter_rms': -1.0}, 'jitter_rms'),
        ({'period': 320.0, 'count': 10, 'jitter_rms': 200.0, 'seed': 0}, 'jitter_rms'),
    ],
)
def test_a_grating_that_cannot_be_built_is_refused(kwargs: dict, parameter: str) -> None:
    with pytest.raises(ArgumentError) as error:
        grating_sections(**kwargs)
    assert error.value.argument == parameter
