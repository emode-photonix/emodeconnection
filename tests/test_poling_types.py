"""The pulsed solver's `Poling` wire type: its forms, their domain walls and the
`domain_lengths` form an inverse design varies.

Every form gives the same thing, the positions (nm along the section) where
the sign of chi(2) flips. `domain_lengths` lists the length of every domain in
order, so its walls are the running sum, and any other form converts to it.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

from emodeconnection.emodeclient import obj_hook
from emodeconnection.pulse import Poling
from emodeconnection.types import ArgumentError, reconstruct

SECTION = 100_000.0  # nm


def _over_the_wire(value: Any) -> Any:
    payload = json.dumps(value.model_dump(), default=_array)
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        data = base64.b64encode(np.ascontiguousarray(value).data).decode()
        return {'__ndarray__': data, 'dtype': str(value.dtype), 'shape': value.shape}
    raise TypeError(type(value))


@pytest.mark.parametrize(
    'lengths, match',
    [([], 'at least one'), ([2000.0, 0.0], '> 0'), ([2000.0, -5.0], '> 0')],
)
def test_domain_lengths_must_be_a_non_empty_list_of_positive_lengths(
    lengths: list[float], match: str
) -> None:
    with pytest.raises(ArgumentError, match=match) as info:
        Poling(domain_lengths=lengths)
    assert info.value.argument == 'domain_lengths'


def test_domain_lengths_is_one_of_the_forms() -> None:
    with pytest.raises(ArgumentError, match='exactly one'):
        Poling(period=4000.0, domain_lengths=[2000.0])
    with pytest.raises(ArgumentError, match='exactly one'):
        Poling(domain_walls=[2000.0], domain_lengths=[2000.0])


def test_the_walls_of_domain_lengths_are_their_running_sum() -> None:
    rng = np.random.default_rng(7)
    lengths = list(rng.uniform(1500.0, 2500.0, size=40))
    # 40 domains of about 2 um end inside the 100 um section, so their end is a wall too
    walls = list(np.cumsum(lengths))
    by_lengths = Poling(domain_lengths=lengths).walls(SECTION)
    by_walls = Poling(domain_walls=walls).walls(SECTION)
    assert by_lengths == by_walls  # exactly


def test_domains_past_the_section_are_dropped_and_the_last_one_runs_to_its_end() -> None:
    poling = Poling(domain_lengths=[3000.0, 4000.0, 5000.0])
    # the third domain ends past a 9 um section, so its end is no wall
    assert poling.walls(9000.0) == [3000.0, 7000.0]
    # a section longer than the domains: the sign flips once more at their end
    assert poling.walls(20000.0) == [3000.0, 7000.0, 12000.0]
    assert poling.total_length == 12000.0


@pytest.mark.parametrize(
    'poling',
    [
        Poling(period=4000.0),
        Poling(period=4000.0, duty_cycle=0.3),
        Poling(z_knots=[0.0, 50_000.0, SECTION], periods=[4000.0, 3500.0, 4200.0]),
        Poling(
            z_knots=[0.0, 30_000.0, 60_000.0, SECTION],
            periods=[4000.0, 3500.0, 4200.0, 3900.0],
            interpolation='spline',
        ),
        Poling(period=4000.0, jitter_rms=50.0, seed=3),
        Poling(domain_walls=[1000.0, 5000.0, 7500.0]),
    ],
)
def test_every_form_converts_to_domain_lengths_with_the_same_walls(poling: Poling) -> None:
    lengths = poling.to_domain_lengths(SECTION)
    assert all(v > 0 for v in lengths)
    assert sum(lengths) == pytest.approx(SECTION, rel=1e-15)
    walls = Poling(domain_lengths=lengths, inverted_first=poling.inverted_first).walls(SECTION)
    np.testing.assert_allclose(walls, poling.walls(SECTION), rtol=0, atol=1e-9)


def test_a_fixed_period_has_its_half_period_domains() -> None:
    lengths = Poling(period=4000.0).to_domain_lengths(SECTION)
    assert len(lengths) == 50
    np.testing.assert_allclose(lengths, 2000.0, rtol=1e-12)


def test_domain_lengths_round_trip_the_wire() -> None:
    poling = Poling(domain_lengths=[2000.0, 2100.5, 1999.25], inverted_first=True)
    back = _over_the_wire(poling)
    assert isinstance(back, Poling)
    assert back == poling
