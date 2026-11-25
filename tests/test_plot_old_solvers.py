"""``plot()``, ``plot_S_matrix()`` and ``material_explorer()`` on an EMode older than 1.0.9 (emode#2073).

EMode 1.0.5 to 1.0.8 draw their plots themselves, in their own window or as a
``PlotImage`` when headless. This client sends those EModes the call they
draw with, as client 1.1.25 did, with only the arguments that EMode takes, and
warns about the browser-tab options it had to leave out.
"""
from __future__ import annotations

import sys
import warnings
from typing import Any, List

import numpy as np
import pytest

from emode.plot import PlotImage
from emode.types import EModeError

from plot_standin import VERSION_1_0_8, session

VERSION_1_0_7 = {"emode": "1.0.7", "protocol": 1}


def _old_session(version: Any = VERSION_1_0_8, answer: Any = "ok"):
    em, client = session(version=version, plot=lambda request: answer)
    client.answers["EM_plot_S_matrix"] = lambda request: answer
    client.answers["EM_material_explorer"] = lambda request: answer
    if version is None:
        client.answers["EM_version"] = lambda request: (_ for _ in ()).throw(
            EModeError("not a supported EMode function")
        )
    return em, client


def _sent(client, function: str) -> dict:
    (call,) = client.calls(function)
    return {k: v for k, v in call.items() if k not in ("function", "simulation_name")}


@pytest.fixture(autouse=True)
def desktop(monkeypatch, tmp_path):
    for name in ("EMODE_HEADLESS", "EMODE_NONINTERACTIVE", "EMODE_PLOT_INLINE", "EMODE_UI_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def browser(monkeypatch):
    from emode.ui import session as ui

    opened: List[str] = []
    monkeypatch.setattr(ui.webbrowser, "open", lambda url: opened.append(url) or True)
    return opened


# ------------------------------------------------------------------ plot()


def test_plot_on_1_0_8_sends_em_plot_with_its_own_arguments_and_opens_no_tab(browser):
    em, client = _old_session()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = em.plot(plane="x-y", component="Ey", mode=1, contours=True, index_outline=False,
                      aspect_ratio="equal", legend=False, section="s1")
    assert out == "ok"
    assert _sent(client, "EM_plot") == {
        "plane": "x-y", "component": "Ey", "mode": 1, "contours": True, "index_outline": False,
        "aspect_ratio": "equal", "legend": False, "section": "s1",
    }
    assert browser == []
    assert "_ui" not in em.__dict__


def test_the_tab_only_options_are_left_out_and_named_in_a_warning():
    em, client = _old_session()
    with pytest.warns(UserWarning) as caught:
        em.plot(component="Ex", title="T", font_size=9, name="first", tags="a, b", z_position=3.0)
    sent = _sent(client, "EM_plot")
    assert sent == {"component": "Ex"}
    ignored = [str(w.message) for w in caught if "ignored" in str(w.message)]
    assert len(ignored) == 1
    for option in ("title", "font_size", "name", "tags", "z_position"):
        assert option in ignored[0]


def test_1_0_7_gets_only_the_arguments_1_0_7_takes():
    em, client = _old_session(VERSION_1_0_7)
    with pytest.warns(UserWarning, match="blend_slices.*contours.*section"):
        em.plot(view="simulation", blend_slices=False, contours=True, section="s1")
    assert _sent(client, "EM_plot") == {"view": "simulation"}


def test_an_emode_older_than_1_0_7_gets_the_call_too():
    em, client = _old_session(None)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert em.plot(component="Hx", view="layout") == "ok"
    assert _sent(client, "EM_plot") == {"component": "Hx", "view": "layout"}


def test_the_old_physical_space_argument_still_becomes_a_view():
    em, client = _old_session()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        em.plot(physical_space=False)
    assert _sent(client, "EM_plot") == {"view": "simulation"}


def test_the_window_warning_is_given_once_a_session():
    em, client = _old_session()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        em.plot()
        em.plot()
        em.plot_S_matrix()
    window = [w for w in caught if "its own window" in str(w.message)]
    assert len(window) == 1
    assert "1.0.8" in str(window[0].message) and "1.0.9" in str(window[0].message)
    assert len(client.calls("EM_plot")) == 2


def test_no_warning_about_ignored_options_when_none_were_given():
    em, _ = _old_session()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        em.plot(component="Ex", mode=0, port=0)
    assert not [w for w in caught if "ignored" in str(w.message)]


@pytest.mark.parametrize("version", [VERSION_1_0_7, VERSION_1_0_8, None])
def test_a_file_name_with_its_extension_is_not_doubled(version):
    em, client = _old_session(version)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        em.plot(file_name="/tmp/out/fig.png", file_type="png")
    sent = _sent(client, "EM_plot")
    assert sent["file_name"] == "/tmp/out/fig" and sent["file_type"] == "png"


def test_no_file_type_is_sent_without_a_file_name():
    em, client = _old_session()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        em.plot()
    assert "file_type" not in _sent(client, "EM_plot")


def test_a_headless_image_is_returned_as_emode_sent_it():
    image = PlotImage(file=None, png=np.frombuffer(b"\x89PNG....", dtype=np.uint8))
    em, _ = _old_session(answer=image)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = em.plot()
    assert isinstance(out, PlotImage) and out.png.tobytes() == b"\x89PNG...."


def test_a_notebook_shows_the_image_emode_sent(monkeypatch):
    shown: List[Any] = []
    display_module = type(sys)("IPython.display")
    display_module.display = shown.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", type(sys)("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    image = PlotImage(file=None, png=np.frombuffer(b"\x89PNG....", dtype=np.uint8))
    em, _ = _old_session(answer=image)
    em._ipython = True
    em.in_ipython = lambda: True  # type: ignore[method-assign]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert em.plot(inline="png") is None
    assert shown == [image]


def test_an_unknown_argument_is_still_refused_before_any_call():
    em, client = _old_session()
    with pytest.raises(EModeError, match="colour"):
        em.plot(colour="red")
    assert client.calls("EM_plot") == []


def test_the_gds_view_still_needs_1_0_9():
    em, client = _old_session()
    with pytest.raises(EModeError, match="needs EMode 1.0.9 or newer, and this EMode is 1.0.8"):
        em.plot(view="gds")
    assert client.calls("EM_plot") == []


def test_plotting_a_result_this_client_holds_needs_no_fallback():
    """``plot(result)`` draws client-side data, so it does not depend on EMode's version."""
    from emode.ui import session as ui

    assert hasattr(ui, "show_data")


# ------------------------------------------------------------------ plot_S_matrix(), material_explorer()


def test_plot_s_matrix_on_1_0_8_sends_em_plot_s_matrix():
    em, client = _old_session()
    with pytest.warns(UserWarning, match="title"):
        assert em.plot_S_matrix(input="a", output="b", plot_function="angle", title="T") == "ok"
    assert _sent(client, "EM_plot_S_matrix") == {"input": "a", "output": "b", "plot_function": "angle"}


def test_material_explorer_on_1_0_8_sends_the_material():
    em, client = _old_session()
    with pytest.warns(UserWarning, match="wavelengths"):
        em.material_explorer(material="Si3N4", wavelengths=(900.0, 1700.0))
    assert _sent(client, "EM_material_explorer") == {"material": "Si3N4"}


def test_material_explorer_before_1_0_8_opens_the_explorer_without_a_material():
    em, client = _old_session(VERSION_1_0_7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        em.material_explorer()
    assert _sent(client, "EM_material_explorer") == {}
    # the default material is not something the user asked for
    assert not [w for w in caught if "ignored" in str(w.message)]


def test_a_chosen_material_before_1_0_8_is_named_as_ignored():
    em, client = _old_session(VERSION_1_0_7)
    with pytest.warns(UserWarning, match="material"):
        em.material_explorer(material="Si3N4")
    assert _sent(client, "EM_material_explorer") == {}


def test_an_emode_1_0_9_without_a_page_is_still_refused():
    from plot_standin import VERSION_1_0_9

    em, client = session(version=VERSION_1_0_9)
    with pytest.raises(EModeError, match="needs EMode 1.0.9 or newer"):
        em.plot_S_matrix()
    assert client.calls("EM_plot_S_matrix") == []
