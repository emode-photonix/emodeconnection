"""Far fields and convergence reports in the workbench's data list (emode#1766).

EMode lists each far field the session computed with ``farfield()`` as a Field
entry (``kind='farfield'``, its arguments those of the call) and each
convergence report a profile holds as a Trend entry (``kind='convergence'``,
``arguments={'profile', 'key'}``): the record ``converge()`` keeps
(``key='convergence'``) or the one ``x_resolution='converge'`` and
``fem_max_refinements='converge'`` keep (``key='resolution_report'``). The
client fetches a far field with ``farfield()`` and a report with ``get()``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from emode.optimize import ConvergenceResult, ConvergenceStage, ConvergenceStep
from emode.plotting import PlotInventory
from emode.types import EModeError
from emode.ui import session as ui
from emode.ui.server import holds_text

from plot_fixtures import farfield
from test_workbench import _calls, _entry, bench, inventory
from test_workbench import desktop  # noqa: F401  (autouse fixture)
from test_workbench import em as _workbench_em  # noqa: F401  (fixture)

FARFIELD = {"profile": "wg", "mode": 1}


def convergence() -> ConvergenceResult:
    steps = [
        ConvergenceStep(setting="resolution", value=v, quantity=q, change=c, estimated_error=e, seconds=0.5)
        for v, q, c, e in ((40.0, 2.40, None, None), (20.0, 2.43, 0.03, None), (10.0, 2.437, 0.007, 2e-3))
    ]
    stage = ConvergenceStage(setting="resolution", status="converged", chosen={"x_resolution": 10.0}, solves=3)
    return ConvergenceResult(
        setting="resolution", target="effective_index", tolerance=1e-2, status="converged", stages=[stage], history=steps
    )


def resolution_report() -> Dict[str, Any]:
    return {
        "tolerance": 1e-8,
        "status": "converged",
        "cap": None,
        "effective_index": [2.4401, 1.8202],
        "estimated_error": [1e-9, 2e-9],
        "order": [2.0, 2.0],
        "finest_resolution": 5.0,
        "history": [
            {"resolution": 20.0, "cells": 1000, "effective_index": [2.43, 1.81], "seconds": 0.1},
            {"resolution": 10.0, "cells": 4000, "effective_index": [2.438, 1.818], "seconds": 0.3},
            {"resolution": 5.0, "cells": 16000, "effective_index": [2.4400, 1.8200], "seconds": 1.2},
        ],
        "seconds": 1.6,
    }


def with_new_entries() -> PlotInventory:
    value = inventory()
    field = [d for d in value.datasets if d.category == "field"]
    trend = [d for d in value.datasets if d.category == "trend"]
    field.append(_entry("farfield:wg mode 1", "field", "farfield", "far field of wg, mode 1", FARFIELD))
    trend += [
        _entry("convergence:wg", "trend", "convergence", "wg: converge()", {"profile": "wg", "key": "convergence"}),
        _entry(
            "resolution_report:FDM",
            "trend",
            "convergence",
            "FDM: resolution 'converge'",
            {"profile": "_default", "key": "resolution_report"},
        ),
    ]
    return value.model_copy(update={"datasets": field + trend})


def _get(request: Dict[str, Any]) -> Any:
    if request.get("key") == "convergence":
        return convergence().to_json()
    if request.get("key") == "resolution_report":
        return resolution_report()
    raise AssertionError(f"unexpected get {request}")


@pytest.fixture
def em(_workbench_em):  # noqa: F811
    em = _workbench_em
    em.client.answers.update(
        {
            "EM_plot_inventory": lambda request: with_new_entries(),
            "EM_farfield": lambda request: farfield(),
            "EM_get": _get,
        }
    )
    return em


# ------------------------------------------------------------------ fetching


def test_a_far_field_is_fetched_with_its_calls_arguments(em):
    data = ui.fetch_dataset(em, "farfield", dict(FARFIELD), (400, 300))
    (call,) = _calls(em, "EM_farfield")
    assert {k: call[k] for k in FARFIELD} == FARFIELD
    assert "pixels" not in call
    assert type(data).__name__ == "FarFieldPanel" and data.panel == "intensity"


def test_a_converge_report_is_a_trend_over_its_solves(em):
    trend = ui.fetch_dataset(em, "convergence", {"profile": "wg", "key": "convergence"}, (400, 300))
    (call,) = _calls(em, "EM_get")
    assert call["key"] == "convergence" and call["profile"] == "wg"
    assert trend.source.data["key"] == "solve"
    sweep = ui.convergence_trend(convergence().to_json())
    np.testing.assert_array_equal(sweep["values"], [1, 2, 3])
    np.testing.assert_allclose(sweep["effective_index"], [2.40, 2.43, 2.437])
    assert np.isnan(sweep["estimated_error"][0]) and sweep["estimated_error"][2] == pytest.approx(2e-3)
    assert np.isnan(sweep["change"][0]) and sweep["change"][1] == pytest.approx(0.03)


def test_a_resolution_report_is_a_trend_over_the_grid_spacing():
    sweep = ui.convergence_trend(resolution_report())
    assert sweep["key"] == "resolution (nm)"
    np.testing.assert_allclose(sweep["values"], [20.0, 10.0, 5.0])
    np.testing.assert_allclose(sweep["effective_index"], [[2.43, 1.81], [2.438, 1.818], [2.44, 1.82]])
    np.testing.assert_array_equal(sweep["cells"], [1000, 4000, 16000])


def test_a_fem_resolution_report_names_its_unknowns():
    report = resolution_report()
    for step in report["history"]:
        step["unknowns"] = step.pop("cells") * 3
        step["elements"] = step["unknowns"] // 2
    sweep = ui.convergence_trend(report)
    np.testing.assert_array_equal(sweep["unknowns"], [3000, 12000, 48000])
    assert "cells" not in sweep


def test_a_report_with_no_history_says_so(em):
    em.client.answers["EM_get"] = lambda request: {"status": "budget", "history": []}
    with pytest.raises(EModeError, match="no steps"):
        ui.fetch_dataset(em, "convergence", {"profile": "wg", "key": "convergence"}, (400, 300))


def test_an_answer_that_is_not_a_report_is_refused(em):
    em.client.answers["EM_get"] = lambda request: "Not Found"
    with pytest.raises(EModeError, match="convergence report"):
        ui.fetch_dataset(em, "convergence", {"profile": "wg", "key": "convergence"}, (400, 300))


def test_an_answer_that_is_not_a_far_field_is_refused(em):
    em.client.answers["EM_farfield"] = lambda request: {"intensity": []}
    with pytest.raises(EModeError, match="farfield"):
        ui.fetch_dataset(em, "farfield", dict(FARFIELD), (400, 300))


# ------------------------------------------------------------------ on the workbench


def test_the_workbench_lists_and_draws_a_far_field(em):
    reply = bench(em, {"dataset": "farfield:wg mode 1"})
    assert reply.get("error") is None, reply.get("error")
    assert reply["kind"] == "farfield"
    assert reply["menu"]["data"] == ["intensity"]
    labels = [d["label"] for d in reply["bench"]["datasets"]]
    assert "far field of wg, mode 1" in labels
    assert len(_calls(em, "EM_farfield")) == 1


@pytest.mark.parametrize(
    "dataset, key",
    [("convergence:wg", "solve"), ("resolution_report:FDM", "resolution (nm)")],
)
def test_the_workbench_draws_a_convergence_report_as_a_trend(em, dataset, key):
    reply = bench(em, {"category": "trend", "dataset": dataset})
    assert reply.get("error") is None, reply.get("error")
    assert reply["kind"] == "sweep"
    assert reply["bench"]["category"] == "trend"
    assert any(t.get("meta") == "trace" for t in reply["figure"]["data"])
    menus = {m["name"]: m for m in reply["extra"]["menus"]}
    assert "effective_index" in menus["result"]["options"]


def test_holds_counts_far_fields_and_convergence_reports():
    text = holds_text(with_new_entries())
    assert text.endswith("a monitor, a far field, 2 convergence reports")
