"""The optimizer's wire types: Parameter, Objective, Constraint, Evaluation and
OptimizationResult.

Each one is plain data. It crosses the wire through the same `__data_type__`
tag as every other wire type, and an `Evaluation` or `OptimizationResult` also
writes as JSON, which is how a run's history is kept in a file.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

from emode import (
    Constraint,
    Evaluation,
    Objective,
    OptimizationResult,
    Parameter,
    SMatrix,
)
from emode.emodeclient import obj_hook
from emode.types import ArgumentError, reconstruct


def _over_the_wire(value: Any) -> Any:
    """What the client rebuilds from `value` after a trip through the protocol's
    JSON, with arrays base64-encoded as the solver sends them."""
    payload = json.dumps(value.model_dump(), default=_array)
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        data = base64.b64encode(np.ascontiguousarray(value).data).decode()
        return {'__ndarray__': data, 'dtype': str(value.dtype), 'shape': value.shape}
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
    S = SMatrix(
        port_sizes=(1, 1), data=np.array([[0.1, 0.9j], [0.9j, 0.1]]), port_names=('left', 'right')
    )
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


def test_a_parameter_objective_reads_the_point() -> None:
    o = Objective('parameter', key='section, taper, length', goal='minimize')
    assert o.result is None
    assert o.value({}, {'section, taper, length': 1200.0}) == pytest.approx(1200.0)
    assert _over_the_wire(o) == o
    with pytest.raises(ArgumentError, match='key'):
        Objective('parameter')


# ---------------------------------------------------------------------------
# The conversion objective and the poling keys of a CW process
# ---------------------------------------------------------------------------


def _response() -> Any:
    """A driven response of an SHG section: a 1 mW pump in at the left, two
    modes at each port and wavelength."""
    from emode import CircuitResponse

    pump, harmonic = 1550.0, 775.0
    sizes = (2, 2, 2, 2)
    names = ('left', 'left', 'right', 'right')
    wavelengths = (pump, harmonic, pump, harmonic)
    incoming = np.zeros(8, dtype=complex)
    incoming[0] = np.sqrt(1e-3)
    outgoing = np.zeros(8, dtype=complex)
    outgoing[4] = np.sqrt(0.9e-3)  # the pump at the right
    outgoing[6] = np.sqrt(0.08e-3) * np.exp(0.3j)  # the harmonic's mode 0
    outgoing[7] = np.sqrt(0.02e-3)  # and its mode 1
    outgoing[2] = np.sqrt(1e-9)  # a little harmonic back at the left
    return CircuitResponse(
        port_sizes=sizes,
        port_names=names,
        wavelengths=wavelengths,
        input=incoming,
        output=outgoing,
        converged=True,
        iterations=2,
        residual=0.0,
        linear=False,
    )


def test_a_conversion_objective_reads_the_driven_response() -> None:
    o = Objective('conversion', out='right', wavelength=775)
    assert o.result == 'response'
    assert o.efficiency is True
    results = {'response': _response()}
    # every mode of the harmonic at the right, over the 1 mW put in
    assert o.value(results) == pytest.approx(0.1)
    assert Objective('conversion', out='right', wavelength=775, mode=1).value(
        results
    ) == pytest.approx(0.02)
    power = Objective('conversion', out='right', wavelength=775, efficiency=False)
    assert power.value(results) == pytest.approx(0.1e-3)
    assert Objective('conversion', out='left', wavelength=775).value(results) == pytest.approx(1e-6)


def test_a_conversion_objective_matches_the_wavelength_to_a_hundredth_of_a_nm() -> None:
    results = {'response': _response()}
    assert Objective('conversion', out='right', wavelength=775.004).value(
        results
    ) == pytest.approx(0.1)
    with pytest.raises(ArgumentError, match='776'):
        Objective('conversion', out='right', wavelength=776).value(results)


def test_a_conversion_objective_needs_its_port_and_wavelength() -> None:
    with pytest.raises(ArgumentError, match='out'):
        Objective('conversion', wavelength=775)
    with pytest.raises(ArgumentError, match='wavelength'):
        Objective('conversion', out='right')
    with pytest.raises(ArgumentError, match='efficiency'):
        Objective('S_power', out='right', into='left', efficiency=False)


def test_a_conversion_objective_round_trips_the_wire() -> None:
    o = Objective('conversion', out='right', wavelength=775.0, mode=0, efficiency=False)
    back = _over_the_wire(o)
    assert isinstance(back, Objective)
    assert back == o
    assert _over_the_wire(Objective('conversion', out='right', wavelength=775.0)).efficiency


def _shg_response(pump: float, harmonic_power: float) -> Any:
    """A driven SHG response at one pump wavelength: 1 mW in, `harmonic_power`
    (W) of harmonic out at the right, one mode per port and wavelength."""
    from emode import CircuitResponse

    incoming = np.zeros(4, dtype=complex)
    incoming[0] = np.sqrt(1e-3)
    outgoing = np.zeros(4, dtype=complex)
    outgoing[2] = np.sqrt(1e-3 - harmonic_power)
    outgoing[3] = np.sqrt(harmonic_power) * np.exp(0.7j)
    return CircuitResponse(
        port_sizes=(1, 1, 1, 1),
        port_names=('left', 'left', 'right', 'right'),
        wavelengths=(pump, pump / 2, pump, pump / 2),
        input=incoming,
        output=outgoing,
        converged=True,
        iterations=2,
        residual=0.0,
        linear=False,
    )


#: Three pump wavelengths and the harmonic power each makes (W, of 1 mW in).
_BAND = {1546.0: 2e-7, 1550.0: 1.5e-6, 1554.0: 6e-7}


def _band_results() -> dict:
    return {'response': [_shg_response(p, h) for p, h in _BAND.items()]}


def _harmonics() -> list:
    return [p / 2 for p in _BAND]


def test_a_conversion_objective_finds_its_wavelength_among_several_responses() -> None:
    results = _band_results()
    for pump, power in _BAND.items():
        o = Objective('conversion', out='right', wavelength=pump / 2)
        assert o.value(results) == pytest.approx(power / 1e-3)
    with pytest.raises(ArgumentError, match='776'):
        Objective('conversion', out='right', wavelength=776).value(results)


def test_a_band_conversion_objective_reduces_its_wavelengths() -> None:
    results = _band_results()
    values = np.array(list(_BAND.values())) / 1e-3
    mean = Objective('conversion', out='right', wavelengths=_harmonics(), reduce='mean')
    assert mean.conversions(results) == pytest.approx(list(values))
    assert mean.value(results) == pytest.approx(values.mean())
    soft = Objective('conversion', out='right', wavelengths=_harmonics(), reduce='soft_min')
    assert soft.temperature == 0.05
    # a soft minimum sits between the minimum and the mean, within tau ln K of the minimum
    tau = soft.temperature * values.mean()
    assert values.min() <= soft.value(results) <= values.min() + tau * np.log(len(values))
    assert soft.value(results) < values.mean()


def test_a_one_wavelength_band_is_the_conversion_objective() -> None:
    results = {'response': _response()}
    single = Objective('conversion', out='right', wavelength=775.0).value(results)
    for reduce in ('soft_min', 'mean'):
        band = Objective('conversion', out='right', wavelengths=[775.0], reduce=reduce)
        assert band.value(results) == pytest.approx(single, rel=1e-14)


def test_the_soft_minimum_approaches_the_minimum_as_the_temperature_drops() -> None:
    results = _band_results()
    values = np.array(list(_BAND.values())) / 1e-3
    gaps = []
    for temperature in (1.0, 0.1, 0.01, 0.001):
        o = Objective(
            'conversion',
            out='right',
            wavelengths=_harmonics(),
            reduce='soft_min',
            temperature=temperature,
        )
        gaps.append(o.value(results) - values.min())
    assert all(g >= 0 for g in gaps)
    assert all(b < a for a, b in zip(gaps, gaps[1:]))
    # the gap is at most tau ln K, tau = temperature * mean: it falls with the temperature
    assert gaps[-1] <= 0.001 * values.mean() * np.log(len(values)) * (1 + 1e-9)
    assert gaps[-1] < 0.01 * values.min()
    # and toward the mean as it rises
    hot = Objective(
        'conversion', out='right', wavelengths=_harmonics(), reduce='soft_min', temperature=1e4
    )
    assert hot.value(results) == pytest.approx(values.mean(), rel=1e-4)


@pytest.mark.parametrize('reduce', ['soft_min', 'mean'])
def test_a_reduction_s_gradient_is_the_derivative_of_its_value(reduce: str) -> None:
    o = Objective(
        'conversion', out='right', wavelengths=[1.0, 2.0, 3.0], reduce=reduce, temperature=0.2
    )
    values = np.array([2e-4, 1.5e-3, 6e-4])
    gradient = np.asarray(o.reduction_gradient(values))
    for k in range(len(values)):
        h = 1e-7 * values[k]
        plus, minus = values.copy(), values.copy()
        plus[k] += h
        minus[k] -= h
        difference = (o.reduction(plus) - o.reduction(minus)) / (2 * h)
        assert gradient[k] == pytest.approx(difference, rel=1e-6, abs=1e-9)
    # both reductions scale with their values, so the gradient's moment is the value
    assert gradient @ values == pytest.approx(o.reduction(values))


def test_a_band_conversion_objective_is_validated() -> None:
    with pytest.raises(ArgumentError, match='reduce'):
        Objective('conversion', out='right', wavelengths=[770.0, 775.0])
    with pytest.raises(ArgumentError, match='wavelengths'):
        Objective('conversion', out='right', wavelength=775.0, wavelengths=[770.0], reduce='mean')
    with pytest.raises(ArgumentError, match='wavelengths'):
        Objective('conversion', out='right', wavelengths=[], reduce='mean')
    with pytest.raises(ArgumentError, match='wavelengths'):
        Objective('conversion', out='right', wavelengths=[770.0, -1.0], reduce='mean')
    with pytest.raises(ArgumentError, match='temperature'):
        Objective('conversion', out='right', wavelengths=[770.0], reduce='soft_min', temperature=0)
    with pytest.raises(ArgumentError, match='reduce'):
        Objective('conversion', out='right', wavelength=775.0, reduce='mean')
    with pytest.raises(ArgumentError, match='wavelengths'):
        Objective('S_power', out='right', into='left', wavelengths=[775.0], reduce='mean')


def test_a_band_conversion_objective_round_trips_the_wire() -> None:
    o = Objective(
        'conversion',
        out='right',
        wavelengths=[771.0, 775.0, 779.0],
        reduce='soft_min',
        temperature=0.02,
        goal='maximize',
    )
    back = _over_the_wire(o)
    assert back == o


def test_the_poling_period_and_duty_cycle_keys_are_parameters() -> None:
    period = Parameter('process, ppln, poling_period', bounds=(4000, 4400), step=1e-4)
    duty = Parameter('process, ppln, duty_cycle', bounds=(0.3, 0.7))
    assert _over_the_wire(period) == period
    assert _over_the_wire(duty) == duty
    assert period.size == 1 and duty.element_bounds() == [(0.3, 0.7)]


def test_an_older_solver_refuses_the_new_kind_by_name() -> None:
    """A solver whose client predates a kind validates the objective against its
    own list of kinds, and the error it returns names the kind it does not know."""
    from emode import optimize

    older = tuple(k for k in optimize.OBJECTIVE_KINDS if k != 'conversion')
    with pytest.raises(ArgumentError, match="'conversion'"):
        optimize._check_choice({'kind': 'conversion'}, 'kind', older, 'Objective')
