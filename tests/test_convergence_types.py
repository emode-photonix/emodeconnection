"""The convergence routine's wire types: ConvergenceStep, ConvergenceStage and
ConvergenceResult.

`converge()` returns a `ConvergenceResult`: the run's history (one row per
solve), one stage per setting it varied, the settings it chose and why it
stopped. It crosses the wire through the same `__data_type__` tag as every
other wire type, and prints as a table.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from emodeconnection import ConvergenceResult, ConvergenceStage, ConvergenceStep
from emodeconnection.emodeclient import obj_hook
from emodeconnection.types import reconstruct


def _over_the_wire(value: Any) -> Any:
    payload = json.dumps(value.model_dump())
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _result() -> ConvergenceResult:
    steps = [
        ConvergenceStep(
            setting='resolution',
            value=40.0,
            settings={'x_resolution': 40.0, 'y_resolution': 40.0},
            quantity=1.6843,
            seconds=0.5,
        ),
        ConvergenceStep(
            setting='resolution',
            value=28.3,
            settings={'x_resolution': 28.3, 'y_resolution': 28.3},
            quantity=1.6862,
            change=1.9e-3,
            estimated_error=1.9e-3,
            seconds=1.1,
        ),
        ConvergenceStep(
            setting='resolution',
            value=20.0,
            settings={'x_resolution': 20.0, 'y_resolution': 20.0},
            quantity=1.6871,
            change=9.0e-4,
            estimated_error=8.6e-4,
            seconds=2.9,
        ),
    ]
    stage = ConvergenceStage(
        setting='resolution',
        status='budget',
        chosen={'x_resolution': 20.0, 'y_resolution': 20.0},
        order=1.1,
        estimated_error=8.6e-4,
        solves=3,
        seconds=4.5,
    )
    return ConvergenceResult(
        setting='resolution',
        target='effective_index',
        tolerance=1e-4,
        status='budget',
        chosen={'x_resolution': 20.0, 'y_resolution': 20.0},
        stages=[stage],
        history=steps,
        seconds=4.5,
    )


def test_a_convergence_result_round_trips_the_wire() -> None:
    result = _result()
    back = _over_the_wire(result)
    assert isinstance(back, ConvergenceResult)
    assert back == result
    assert isinstance(back.history[1], ConvergenceStep)
    assert isinstance(back.stages[0], ConvergenceStage)
    assert back.history[1].estimated_error == pytest.approx(1.9e-3)
    assert back.stages[0].order == pytest.approx(1.1)


def test_a_result_is_converged_only_when_its_status_says_so() -> None:
    result = _result()
    assert not result.converged
    assert result.model_copy(update={'status': 'converged'}).converged


def test_the_table_has_one_row_per_solve_and_a_line_per_stage() -> None:
    table = _result().table()
    lines = table.splitlines()
    rows = [line for line in lines if line.startswith('resolution ')]
    assert len(rows) == 3
    assert '8.6e-04' in table and '1.9e-03' in table
    assert any('budget' in line for line in lines)
    assert str(_result()) == table


def test_a_window_value_reads_as_width_by_height() -> None:
    step = ConvergenceStep(
        setting='window', value=[7000.0, 4200.0], settings={}, quantity=1.0, seconds=0.1
    )
    assert '7000 x 4200' in step.value_text()


def test_the_status_must_be_one_of_the_routine_s() -> None:
    with pytest.raises(ValueError):
        ConvergenceStage(setting='resolution', status='finished', chosen={}, solves=0, seconds=0)


def test_the_history_is_plain_json() -> None:
    data = _result().to_json()
    assert json.loads(json.dumps(data))['history'][2]['value'] == 20.0
