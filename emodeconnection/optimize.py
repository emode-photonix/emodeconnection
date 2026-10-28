"""Wire types of the optimizer, of `evaluate()` and of `converge()`.

An optimization is declared with three kinds of objects and returns one:

- `Parameter`, what to vary: a key in the `sweep()` grammar (`'wavelength'`,
  `'shape, <name>, <property>'`, `'section, <name>, <property>'`,
  `'setting, <profile>, <name>'`), its bounds and its starting point.
- `Objective`, what to improve: an S-matrix power or phase, a monitor
  coupling, a mode's effective or group index or loss, or the energy (or
  average power) of a pulse in a wavelength band, with a goal and a weight.
  A list of objectives is their weighted sum.
- `Constraint`, a limit on a parameter or on an objective's quantity.
- `OptimizationResult`, returned by `optimize()`: the best point, its
  objective and constraints, and the history of every evaluation as
  `Evaluation` records.

Everything here is plain data. An objective is a declaration, not a function,
so a record, the snapshot file it was evaluated from and the objective are
enough to reproduce its number anywhere. `Objective.value(results)` scores a
record's results on the client too.

Units follow the rest of the API: lengths and wavelengths in nm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import ConfigDict, field_serializer, model_validator

from .types import ArgumentError, TaggedModel, register_type

Bound = Tuple[Optional[float], Optional[float]]

OBJECTIVE_KINDS = (
    'S_power',
    'S_phase',
    'monitor_coupling',
    'effective_index',
    'group_index',
    'loss',
    'band_energy',
    'band_average',
    'parameter',
)

#: The result of `evaluate()` each kind of objective reads.
_RESULT_OF = {
    'S_power': 'S_matrix',
    'S_phase': 'S_matrix',
    'monitor_coupling': 'monitor_coupling',
    'effective_index': 'effective_index',
    'group_index': 'group_index',
    'loss': 'loss_dB_per_m',
    'band_energy': 'pulse_response',
    'band_average': 'pulse_response',
    'parameter': None,
}

GOALS = ('maximize', 'minimize', 'target')
OPERATORS = ('<=', '>=', '==')


def _check_choice(data: Any, field: str, choices: tuple, owner: str) -> None:
    if isinstance(data, dict) and field in data and data[field] not in choices:
        raise ArgumentError(
            f'{field} must be one of {", ".join(repr(c) for c in choices)}, not {data[field]!r}',
            owner,
            field,
        )


@register_type
class Parameter(TaggedModel):
    """One quantity the optimizer varies.

    `key` names it in the `sweep()` grammar. `bounds` is `(min, max)`, either
    end `None` for no limit. `x0` is the starting point, by default the value
    the simulation holds now. `step` is the relative finite-difference step
    (a fraction of the bounded range, or of the value when unbounded).

    A vector parameter has a `shape` (for instance `(n,)`), an `x0` with one
    value per element and either one `(min, max)` pair for every element or a
    list of pairs, one per element.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    key: str
    bounds: Optional[Union[Bound, List[Bound]]] = None
    x0: Optional[Union[float, List[float]]] = None
    shape: Tuple[int, ...] = ()
    step: Optional[float] = None

    def __init__(self, key: str, **data: Any) -> None:
        super().__init__(key=key, **data)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def element_bounds(self) -> List[Bound]:
        """One `(min, max)` pair per element."""
        if self.bounds is None:
            return [(None, None)] * self.size
        if isinstance(self.bounds, list):
            return [tuple(b) for b in self.bounds]  # type: ignore[misc]
        return [tuple(self.bounds)] * self.size  # type: ignore[list-item]

    def element_x0(self) -> Optional[List[float]]:
        """The starting point as one value per element, or None when not given."""
        if self.x0 is None:
            return None
        return [float(v) for v in np.ravel(self.x0)]

    @model_validator(mode='after')
    def _validate(self) -> 'Parameter':
        name = 'Parameter'
        if any(n < 1 for n in self.shape):
            raise ArgumentError('shape must have positive sizes', name, 'shape')
        if isinstance(self.bounds, list) and len(self.bounds) != self.size:
            raise ArgumentError(
                f'bounds has {len(self.bounds)} pairs for {self.size} elements', name, 'bounds'
            )
        if self.shape == () and isinstance(self.x0, list):
            raise ArgumentError('a scalar parameter takes one x0', name, 'x0')
        x0 = self.element_x0()
        if x0 is not None and len(x0) != self.size:
            raise ArgumentError(f'x0 has {len(x0)} values for {self.size} elements', name, 'x0')
        for k, (lo, hi) in enumerate(self.element_bounds()):
            if lo is not None and hi is not None and not lo < hi:
                raise ArgumentError(
                    f'bounds must be (min, max) with min < max, not {(lo, hi)}', name, 'bounds'
                )
            if x0 is not None and (
                (lo is not None and x0[k] < lo) or (hi is not None and x0[k] > hi)
            ):
                raise ArgumentError(f'x0 {x0[k]} lies outside the bounds {(lo, hi)}', name, 'x0')
        if self.step is not None and not self.step > 0:
            raise ArgumentError('step must be > 0', name, 'step')
        return self


@register_type
class Objective(TaggedModel):
    """A quantity to improve.

    `kind` and the selectors it uses:

    - `'S_power'` and `'S_phase'`, `|S|^2` and the phase of one S-matrix
      element: `out` and `into` name the ports by name (mode 0), as
      `'<port>_<mode>'`, or as a mode index of the whole matrix.
      `wavelength` picks a port of a multi-wavelength circuit.
    - `'monitor_coupling'`, the coupling of a beam into a monitor:
      `monitor` and `beam` (the arguments of `monitor_coupling()`).
    - `'effective_index'`, `'group_index'` and `'loss'` (dB/m) of mode
      `mode` (default 0) of `profile` (default the default profile).
    - `'band_energy'` (pJ) and `'band_average'` (average power, W) of a pulse
      in `band = (min, max)` nm at the output, of tracked mode `mode` (default
      all of them).
    - `'parameter'`, the value of the parameter `key` itself, to make a
      device as short or as narrow as its constraints allow.

    `goal` is `'maximize'` (the default), `'minimize'`, or `'target'` with a
    `target` value, which minimizes the squared distance to it. `weight`
    scales the objective in a weighted sum of several.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: Literal[
        'S_power',
        'S_phase',
        'monitor_coupling',
        'effective_index',
        'group_index',
        'loss',
        'band_energy',
        'band_average',
        'parameter',
    ]
    key: Optional[str] = None
    out: Optional[Union[int, str]] = None
    into: Optional[Union[int, str]] = None
    wavelength: Optional[float] = None
    mode: Optional[int] = None
    profile: Optional[str] = None
    monitor: Optional[str] = None
    beam: Optional[Dict[str, Any]] = None
    band: Optional[Tuple[float, float]] = None
    goal: Literal['maximize', 'minimize', 'target'] = 'maximize'
    target: Optional[float] = None
    weight: float = 1.0

    def __init__(self, kind: str, **data: Any) -> None:
        super().__init__(kind=kind, **data)

    @model_validator(mode='before')
    @classmethod
    def _choices(cls, data: Any) -> Any:
        _check_choice(data, 'kind', OBJECTIVE_KINDS, 'Objective')
        _check_choice(data, 'goal', GOALS, 'Objective')
        return data

    @model_validator(mode='after')
    def _validate(self) -> 'Objective':
        name = 'Objective'
        if self.kind in ('S_power', 'S_phase'):
            for selector in ('out', 'into'):
                if getattr(self, selector) is None:
                    raise ArgumentError(
                        f"an '{self.kind}' objective needs {selector}=", name, selector
                    )
        if self.kind == 'monitor_coupling' and (self.monitor is None or not self.beam):
            raise ArgumentError(
                "a 'monitor_coupling' objective needs monitor= and beam=", name, 'monitor'
            )
        if self.kind in ('band_energy', 'band_average'):
            if self.band is None or not 0 < self.band[0] < self.band[1]:
                raise ArgumentError(
                    f"a '{self.kind}' objective needs band=(min, max) in nm", name, 'band'
                )
        if (self.kind == 'parameter') != (self.key is not None):
            raise ArgumentError("give key= exactly for a 'parameter' objective", name, 'key')
        if (self.goal == 'target') != (self.target is not None):
            raise ArgumentError("give target= exactly when goal='target'", name, 'target')
        if not self.weight > 0:
            raise ArgumentError('weight must be > 0', name, 'weight')
        return self

    @property
    def result(self) -> Optional[str]:
        """The `evaluate()` result this objective reads (None for a parameter)."""
        return _RESULT_OF[self.kind]

    def value(
        self, results: Mapping[str, Any], parameters: Optional[Mapping[str, Any]] = None
    ) -> float:
        """This objective's quantity in an evaluation's `results` (and, for a
        `'parameter'` objective, its `parameters`)."""
        if self.kind == 'parameter':
            if parameters is None or self.key not in parameters:
                raise ArgumentError(f"the point holds no '{self.key}'", 'Objective', 'key')
            return float(parameters[self.key])
        if self.result not in results:
            raise ArgumentError(f"the results hold no '{self.result}'", 'Objective', 'kind')
        data = _at_wavelength(results[self.result], self.wavelength)
        if self.kind in ('S_power', 'S_phase'):
            element = _s_element(data, self.out, self.into, self.wavelength)  # type: ignore[arg-type]
            return float(abs(element) ** 2) if self.kind == 'S_power' else float(np.angle(element))
        if self.kind in ('band_energy', 'band_average'):
            lo, hi = self.band  # type: ignore[misc]
            if self.kind == 'band_energy':
                return float(data.band_energy(lo, hi, mode=self.mode))
            return float(data.band_power(lo, hi, mode=self.mode))
        values = np.ravel(np.asarray(data))
        return float(np.real(values[self.mode or 0]))

    def term(self, value: float) -> float:
        """The weighted amount this objective adds to a cost that is minimized."""
        if self.goal == 'maximize':
            return -self.weight * value
        if self.goal == 'minimize':
            return self.weight * value
        return self.weight * (value - self.target) ** 2  # type: ignore[operator]


def _at_wavelength(data: Any, wavelength: Optional[float]) -> Any:
    """One wavelength's entry of a per-wavelength result."""
    if not isinstance(data, Mapping):
        return data
    if wavelength is not None:
        return data[wavelength]
    if len(data) == 1:
        return next(iter(data.values()))
    raise ArgumentError(
        'the result holds several wavelengths: give the objective wavelength=',
        'Objective',
        'wavelength',
    )


def _s_element(
    S: Any, out: Union[int, str], into: Union[int, str], wavelength: Optional[float]
) -> complex:
    data = np.asarray(S.data if hasattr(S, 'data') else S)
    return complex(data[_mode_index(S, out, wavelength), _mode_index(S, into, wavelength)])


def _mode_index(S: Any, port: Union[int, str], wavelength: Optional[float]) -> int:
    """The row (or column) of S that `port` names: a mode index, a port name
    (its mode 0) or `'<port>_<mode>'`."""
    if isinstance(port, (int, np.integer)):
        return int(port)
    names = getattr(S, 'port_names', None)
    if not names:
        raise ArgumentError(
            f'this S-matrix has no port names, so {port!r} names nothing', 'Objective', 'out'
        )
    mode = 0
    if port not in names:
        prefix, _, suffix = port.rpartition('_')
        if prefix not in names or not suffix.isdigit():
            raise ArgumentError(
                f'no port named {port!r}, the ports are {list(names)}', 'Objective', 'out'
            )
        port, mode = prefix, int(suffix)
    index = S.port_index(port, wavelength) if wavelength is not None else S.port_index(port)
    if mode >= S.port_sizes[index]:
        raise ArgumentError(
            f'port {port!r} has {S.port_sizes[index]} modes, not mode {mode}', 'Objective', 'out'
        )
    return S.port_slices[index].start + mode


@register_type
class Constraint(TaggedModel):
    """A limit: `quantity op value`.

    `quantity` is a parameter key (`'section, coupler, length'`) or an
    `Objective`, whose quantity is limited (its goal and weight do not
    matter here). `op` is `'<='`, `'>='` or `'=='`.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    quantity: Union[Objective, str]
    op: Literal['<=', '>=', '==']
    value: float

    def __init__(self, quantity: Union[Objective, str], op: str, value: float, **data: Any) -> None:
        super().__init__(quantity=quantity, op=op, value=value, **data)

    @model_validator(mode='before')
    @classmethod
    def _choices(cls, data: Any) -> Any:
        _check_choice(data, 'op', OPERATORS, 'Constraint')
        return data

    def violation(self, value: float) -> float:
        """How far `value` is from satisfying the constraint (0 when it does)."""
        if self.op == '<=':
            return max(0.0, value - self.value)
        if self.op == '>=':
            return max(0.0, self.value - value)
        return abs(value - self.value)


def _plain(value: Any) -> Any:
    """`value` as JSON-ready data: arrays become lists, a complex number
    `{'re': ..., 'im': ...}` and a wire type the plain form of its dump."""
    if isinstance(value, TaggedModel):
        return _plain(value.model_dump())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return {'re': float(value.real), 'im': float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    return value


@register_type
class Evaluation(TaggedModel):
    """One evaluation of the simulation at one point.

    `parameters` holds the value of every parameter (a list for a vector
    parameter), `objective` the run's objective there and `constraints` the
    quantity of every constraint, in order. `results` holds only the requested
    results. `seconds` is the wall time, `solves` the mode solves made and
    `cache_hits` the solves (modes or junctions) served by the solver cache.
    `cached` is set when the point repeats an earlier one of the run, which
    is then not evaluated again. `gradient`, when present, is the objective's
    gradient with respect to the parameters, and `error` says why an
    evaluation failed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    parameters: Dict[str, Union[float, List[float]]]
    objective: Optional[float] = None
    constraints: List[float] = []
    results: Dict[str, Any] = {}
    seconds: float = 0.0
    solves: int = 0
    cache_hits: int = 0
    cached: bool = False
    seeded: bool = False
    gradient: Optional[List[float]] = None
    error: Optional[str] = None

    @field_serializer('results', when_used='json')
    def _results_as_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return _plain(results)

    def to_json(self) -> Dict[str, Any]:
        """This record as JSON-ready plain data."""
        return self.model_dump(mode='json')


@register_type
class OptimizationResult(TaggedModel):
    """What `optimize()` returns.

    `x` is the best point as a flat list (vector parameters flattened in
    order) and `parameters` the same point by key. `objective` and
    `constraints` are its values, and `goal` says whether the objective was
    maximized or minimized. `status` is `'converged'`, `'budget'` (the
    evaluation budget ran out), `'time'` (`max_time` ran out) or `'failed'`,
    and `message` says more. `history` holds every evaluation in order,
    `best_index` the index of the best one, `evaluations` the number actually
    solved (not served from the run's memo) and `snapshot` the snapshot file
    the run evaluated from.
    """

    x: List[float]
    parameters: Dict[str, Union[float, List[float]]]
    objective: Optional[float] = None
    constraints: List[float] = []
    converged: bool = False
    status: Literal['converged', 'budget', 'time', 'failed'] = 'failed'
    message: str = ''
    history: List[Evaluation] = []
    best_index: int = 0
    evaluations: int = 0
    seconds: float = 0.0
    snapshot: Optional[str] = None
    goal: Literal['maximize', 'minimize'] = 'minimize'

    @property
    def best(self) -> Evaluation:
        """The best evaluation of the run."""
        return next(r for r in self.history if r.index == self.best_index)

    def to_json(self) -> Dict[str, Any]:
        """This result, history included, as JSON-ready plain data."""
        return self.model_dump(mode='json')


# ---------------------------------------------------------------------------
# converge()
# ---------------------------------------------------------------------------

CONVERGENCE_SETTINGS = ('resolution', 'window', 'pml', 'modes')
CONVERGENCE_STATUSES = ('converged', 'oscillating', 'stalled', 'budget', 'lost mode')

ConvergenceStatus = Literal['converged', 'oscillating', 'stalled', 'budget', 'lost mode']


@register_type
class ConvergenceStep(TaggedModel):
    """One solve of a `converge()` run.

    `setting` is the setting the step varied (`'resolution'`, `'window'`,
    `'pml'` or `'modes'`) and `value` its value: the x resolution in nm, the
    window as `[width, height]` in nm, the PML thickness in nm or the number
    of modes. `settings` holds every setting the step solved with, as
    `settings()` takes them. `quantity` is the target quantity of the tracked
    mode (None when the mode was lost), `change` its change from the previous
    step and `estimated_error` the routine's estimate of its distance from the
    converged value. `seconds` is the wall time of the step.
    """

    setting: Literal['resolution', 'window', 'pml', 'modes']
    value: Union[float, List[float]]
    settings: Dict[str, Any] = {}
    quantity: Optional[float] = None
    change: Optional[float] = None
    estimated_error: Optional[float] = None
    mode: Optional[int] = None
    seconds: float = 0.0

    def value_text(self) -> str:
        """The step's value as the table prints it."""
        if self.setting == 'window' and isinstance(self.value, list):
            return f'{self.value[0]:.0f} x {self.value[1]:.0f} nm'
        if self.setting == 'modes':
            return f'{int(self.value)}'  # type: ignore[arg-type]
        if self.setting == 'resolution' and self.settings.get('solver') == 'FEM':
            return f'{float(self.value):.3g}/wl'  # type: ignore[arg-type]
        return f'{float(self.value):.4g} nm'  # type: ignore[arg-type]


@register_type
class ConvergenceStage(TaggedModel):
    """The part of a `converge()` run that varied one setting.

    `status` is why the stage stopped: `'converged'` (the estimated error was
    below the tolerance on two consecutive steps, or below it once for the
    orthogonality target), `'oscillating'` (the change flipped sign between
    two consecutive steps), `'stalled'` (the orthogonality measure stopped
    improving), `'budget'` (`max_solves` or `max_time` ran out) or
    `'lost mode'` (no mode of a step matched the tracked one). `chosen` holds
    the settings the stage chose, `order` the observed order of a resolution
    stage and `estimated_error` the chosen step's estimated error.
    """

    setting: Literal['resolution', 'window', 'pml', 'modes']
    status: ConvergenceStatus
    chosen: Dict[str, Any] = {}
    order: Optional[float] = None
    estimated_error: Optional[float] = None
    solves: int = 0
    seconds: float = 0.0


@register_type
class ConvergenceResult(TaggedModel):
    """What `converge()` returns.

    `setting` and `target` are the arguments the run was called with, and
    `tolerance` the tolerance it used (relative for the loss target, absolute
    otherwise). `status` is `'converged'` when every stage converged and
    otherwise the status of the first stage that did not. `chosen` holds every
    setting the run chose, as `settings()` takes them, and `applied` says
    whether they were written to the profile. `stages` has one entry per
    setting varied, in order, and `history` one row per solve.
    """

    setting: Literal['resolution', 'window', 'pml', 'modes', 'all']
    target: Literal['effective_index', 'group_index', 'loss', 'orthogonality']
    tolerance: float
    relative: bool = False
    status: ConvergenceStatus
    chosen: Dict[str, Any] = {}
    stages: List[ConvergenceStage] = []
    history: List[ConvergenceStep] = []
    seconds: float = 0.0
    applied: bool = False

    @property
    def converged(self) -> bool:
        return self.status == 'converged'

    def table(self) -> str:
        """The run as a fixed-format table, one row per solve and one line per stage."""
        quantity = {
            'effective_index': 'n_eff',
            'group_index': 'n_g',
            'loss': 'dB/m',
            'orthogonality': 'overlap',
        }[self.target]
        lines = [
            f'{"setting":11s}{"value":>20s}{quantity:>14s}{"change":>11s}'
            f'{"est. error":>12s}{"time (s)":>10s}'
        ]
        for step in self.history:
            lines.append(
                f'{step.setting:11s}{step.value_text():>20s}{_number(step.quantity, 14)}'
                f'{_exponent(step.change, 11)}{_exponent(step.estimated_error, 12)}'
                f'{step.seconds:>10.1f}'
            )
        lines.append('')
        kind = 'relative' if self.relative else 'absolute'
        for stage in self.stages:
            chosen = ', '.join(f'{k}={_plain(v)}' for k, v in stage.chosen.items())
            error = (
                ''
                if stage.estimated_error is None
                else f', estimated error {stage.estimated_error:.1e}'
            )
            order = '' if stage.order is None else f', order {stage.order:.2f}'
            lines.append(f'{stage.setting}: {stage.status} ({chosen}{error}{order})')
        lines.append(
            f'{self.status} at {self.tolerance:g} {kind} in {self.seconds:.1f} s'
            + ('' if not self.applied else ', settings applied')
        )
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.table()

    def to_json(self) -> Dict[str, Any]:
        """This result as JSON-ready plain data."""
        return self.model_dump(mode='json')


def _number(value: Optional[float], width: int) -> str:
    return f'{"-":>{width}s}' if value is None else f'{value:>{width}.7g}'


def _exponent(value: Optional[float], width: int) -> str:
    return f'{"-":>{width}s}' if value is None else f'{value:>{width}.1e}'
