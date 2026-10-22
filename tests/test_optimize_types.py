"""The optimizer's wire types: Parameter, Objective, Constraint, Evaluation and
OptimizationResult.

Each one is plain data. It crosses the wire through the same `__data_type__`
tag as every other wire type, and an `Evaluation` or `OptimizationResult` also
writes as JSON, which is how a run's history is kept in a file.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from emodeconnection import (
    Constraint,
    Evaluation,
    Objective,
    OptimizationResult,
    Parameter,
    SMatrix,
)
from emodeconnection.types import ArgumentError, reconstruct, serialize


def _over_the_wire(value: Any) -> Any:
    """What the other side rebuilds from `value` after a JSON trip."""
    payload = json.dumps(serialize(value.model_dump()), default=_array)
    return reconstruct(json.loads(payload))


def _array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {'re': value.real.tolist(), 'im': value.imag.tolist()}
        return value.tolist()
    raise TypeError(type(value))


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------


def test_a_parameter_takes_its_key_positionally() -> None:
    p = Parameter('section, coupler, length', bounds=(5e3, 40e3), x0=15e3)
    assert p.key == 'section, coupler, length'
    assert p.bounds == (5e3, 40e3)
    assert p.x0 == 15e3
    assert p.size == 1
    assert p.element_bounds() == [(5e3, 40e3)]


def test_a_parameter_round_trips_the_wire() -> None:
    p = Parameter('shape, top, position', bounds=(-900, -600), step=1e-4)
    back = _over_the_wire(p)
    assert isinstance(back, Parameter)
    assert back == p


def test_a_vector_parameter_has_one_bound_pair_per_element() -> None:
    p = Parameter('lengths', shape=(3,), bounds=[(1, 2), (3, 4), (5, 6)], x0=[1.5, 3.5, 5.5])
    assert p.size == 3
    assert p.element_bounds() == [(1, 2), (3, 4), (5, 6)]
    shared = Parameter('lengths', shape=(2,), bounds=(0, 1))
    assert shared.element_bounds() == [(0, 1), (0, 1)]
    back = _over_the_wire(p)
    assert back == p


def test_a_vector_parameter_rejects_the_wrong_number_of_bounds() -> None:
    with pytest.raises(ArgumentError, match='bounds'):
        Parameter('lengths', shape=(3,), bounds=[(1, 2), (3, 4)])
    with pytest.raises(ArgumentError, match='x0'):
        Parameter('lengths', shape=(2,), x0=[1.0, 2.0, 3.0])


def test_a_parameter_rejects_inverted_bounds_and_a_start_outside_them() -> None:
    with pytest.raises(ArgumentError, match='bounds'):
        Parameter('wavelength', bounds=(1600, 1500))
    with pytest.raises(ArgumentError, match='x0'):
        Parameter('wavelength', bounds=(1500, 1600), x0=1700)


# ---------------------------------------------------------------------------
# Objective and Constraint
# ---------------------------------------------------------------------------


def test_an_objective_round_trips_the_wire() -> None:
    o = Objective('S_power', out='right_top', into='left_0', goal='maximize', weight=2.0)
    back = _over_the_wire(o)
    assert isinstance(back, Objective)
    assert back == o


def test_an_objective_names_the_result_it_reads() -> None:
    assert Objective('S_power', out='right', into='left').result == 'S_matrix'
    assert Objective('effective_index', goal='minimize').result == 'effective_index'
    assert Objective('loss', goal='minimize').result == 'loss_dB_per_m'
    assert Objective('group_index', goal='minimize').result == 'group_index'
    assert Objective('band_energy', band=(1000, 1100)).result == 'pulse_response'


def test_an_objective_checks_its_selectors() -> None:
    with pytest.raises(ArgumentError, match='out'):
        Objective('S_power', into='left')
    with pytest.raises(ArgumentError, match='target'):
        Objective('effective_index', goal='target')
    with pytest.raises(ArgumentError, match='band'):
        Objective('band_energy')
    with pytest.raises(ArgumentError, match='monitor'):
        Objective('monitor_coupling')
    with pytest.raises(ArgumentError, match='weight'):
        Objective('effective_index', weight=0)


def test_an_objective_scores_an_s_matrix_by_port_name_and_mode() -> None:
    data = np.zeros((4, 4), dtype=complex)
    data[2, 0] = 0.6 * np.exp(0.5j)  # right mode 0 from left mode 0
    data[3, 0] = 0.3  # right mode 1 from left mode 0
    S = SMatrix(port_sizes=(2, 2), data=data, port_names=('left', 'right'))
    results = {'S_matrix': S}
    assert Objective('S_power', out='right', into='left').value(results) == pytest.approx(0.36)
    assert Objective('S_power', out='right_1', into='left_0').value(results) == pytest.approx(0.09)
    assert Objective('S_power', out=3, into=0).value(results) == pytest.approx(0.09)
    assert Objective('S_phase', out='right', into='left').value(results) == pytest.approx(0.5)


def test_an_objective_scores_a_mode_quantity() -> None:
    results = {'effective_index': np.array([2.4 + 1e-5j, 1.9])}
    assert Objective('effective_index').value(results) == pytest.approx(2.4)
    assert Objective('effective_index', mode=1).value(results) == pytest.approx(1.9)


def test_a_constraint_takes_a_key_or_an_objective() -> None:
    on_key = Constraint('section, coupler, length', '<=', 30e3)
    on_objective = Constraint(Objective('S_power', out='right', into='left'), '>=', 0.99)
    assert _over_the_wire(on_key) == on_key
    back = _over_the_wire(on_objective)
    assert isinstance(back.quantity, Objective)
    assert back == on_objective


def test_a_constraint_measures_its_violation() -> None:
    assert Constraint('x', '<=', 2).violation(2.5) == pytest.approx(0.5)
    assert Constraint('x', '<=', 2).violation(1.0) == 0
    assert Constraint('x', '>=', 2).violation(1.5) == pytest.approx(0.5)
    assert Constraint('x', '==', 2).violation(1.5) == pytest.approx(0.5)
    with pytest.raises(ArgumentError, match='op'):
        Constraint('x', '<', 2)


# ---------------------------------------------------------------------------
# Evaluation and OptimizationResult
# ---------------------------------------------------------------------------


def _evaluation(index: int = 0) -> Evaluation:
    S = SMatrix(port_sizes=(1, 1), data=np.array([[0.1, 0.9j], [0.9j, 0.1]]), port_names=('left', 'right'))
    return Evaluation(
        index=index,
        parameters={'section, g, length': 1000.0, 'lengths': [1.0, 2.0]},
        objective=0.81,
        constraints=[0.5],
        results={'S_matrix': S, 'effective_index': np.array([2.4 + 1e-5j])},
        seconds=0.25,
        solves=1,
        cache_hits=3,
    )


def test_an_evaluation_round_trips_the_wire() -> None:
    back = _over_the_wire(_evaluation())
    assert isinstance(back, Evaluation)
    assert back.index == 0 and back.solves == 1 and back.cache_hits == 3
    assert back.parameters == {'section, g, length': 1000.0, 'lengths': [1.0, 2.0]}
    assert isinstance(back.results['S_matrix'], SMatrix)
    assert back.gradient is None and back.error is None and back.cached is False


def test_an_evaluation_is_one_json_line() -> None:
    line = _evaluation().model_dump_json()
    assert '\n' not in line
    plain = json.loads(line)
    assert plain['objective'] == pytest.approx(0.81)
    assert plain['results']['effective_index'] == [{'re': 2.4, 'im': 1e-5}]
    assert json.dumps(_evaluation().to_json())
    back = Evaluation.model_validate_json(line)
    assert back.index == 0 and back.parameters == _evaluation().parameters


def _result() -> OptimizationResult:
    history = [_evaluation(0), _evaluation(1)]
    return OptimizationResult(
        x=[1000.0, 1.0, 2.0],
        parameters=history[1].parameters,
        objective=0.81,
        constraints=[0.5],
        converged=True,
        status='converged',
        message='done',
        history=history,
        best_index=1,
        evaluations=2,
        seconds=0.5,
        snapshot=None,
        goal='maximize',
    )


def test_an_optimization_result_round_trips_the_wire_and_json() -> None:
    res = _result()
    back = _over_the_wire(res)
    assert isinstance(back, OptimizationResult)
    assert all(isinstance(r, Evaluation) for r in back.history)
    assert back.best.index == 1
    assert back.x == res.x and back.status == 'converged'
    assert json.loads(json.dumps(res.to_json()))['history'][1]['index'] == 1
