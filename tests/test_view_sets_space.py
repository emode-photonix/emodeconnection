"""The view sets the space, there is no 'Physical space' setting (emode#1649).

Layout is physical space and simulation the simulation space, chosen in the View
menu. The tab's Settings list no longer offers 'Physical space', even from an
EMode that still sends it, and ``plot(physical_space=...)`` still works with a
``DeprecationWarning``, as ``view='layout'`` or ``view='simulation'``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from js_harness import needs_jsc, run_app
from test_plot_call import browser, clock, em, interactive  # noqa: F401  (fixtures)

APP_JS = Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "app.js"


def test_the_settings_list_has_no_physical_space():
    js = APP_JS.read_text()
    settings = js[
        js.index("const SETTINGS = [") : js.index("];", js.index("const SETTINGS = ["))
    ]
    assert "Physical space" not in settings


@needs_jsc
def test_the_tab_draws_no_physical_space_checkbox_from_an_older_emodes_menu():
    got = run_app(
        """
var box = document.createElement("div");
buildPlotControls(box, {menu: {views: ["layout", "simulation"],
  settings: {"Index outline": true, "Physical space": true},
  active_settings: ["Index outline", "Physical space"], selected: {view: "layout"}}, controls: {}});
return __text(box);
"""
    )
    assert "Physical space" not in got
    assert "Index outline" in got


@pytest.mark.parametrize("physical, view", [(True, "layout"), (False, "simulation")])
def test_plot_physical_space_is_deprecated_and_sets_the_view(
    em,  # noqa: F811
    browser,  # noqa: F811
    physical,
    view,
):
    with pytest.warns(DeprecationWarning, match="view="):
        em.plot(plane="z-x", physical_space=physical)
    (sent,) = em.client.calls("EM_plot")
    assert sent["view"] == view
    assert "physical_space" not in sent


def test_an_explicit_view_wins_over_physical_space(em, browser):  # noqa: F811
    with pytest.warns(DeprecationWarning):
        em.plot(plane="z-x", view="simulation", physical_space=True)
    (sent,) = em.client.calls("EM_plot")
    assert sent["view"] == "simulation"


def test_the_session_page_names_the_simulation_name():
    js = APP_JS.read_text()
    assert '["Simulation name", (e) => e.simulation_name]' in js
    assert '["Simulation", ' not in js
