"""``plot()`` returns at once, the one tab, ``close()``'s wait, headless images and the pairing.

Against a stand-in session (``plot_standin.py``) whose ``EM_plot`` answers a
``PlotView`` fixture, with the browser, the clock and the pause replaced.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from emode.plot import PlotImage
from emode.types import ArgumentError, EModeError
from emode.ui import session as ui

from plot_fixtures import schematic, xy_view
from plot_standin import VERSION_1_0_8, session


class Clock:
    """A clock that moves only when the code under test pauses."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.on_sleep: List[Any] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        for hook in list(self.on_sleep):
            hook(self.now)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(ui, "clock", c)
    monkeypatch.setattr(ui, "sleep", c.sleep)
    return c


@pytest.fixture
def browser(monkeypatch):
    opened: List[str] = []
    monkeypatch.setattr(ui.webbrowser, "open", lambda url: opened.append(url) or True)
    return opened


@pytest.fixture(autouse=True)
def interactive(monkeypatch, tmp_path):
    """A desktop with a display: no headless or non-interactive switch set."""
    for name in ("EMODE_HEADLESS", "EMODE_NONINTERACTIVE", "EMODE_PLOT_INLINE", "EMODE_UI_EXIT_WAIT", "EMODE_UI_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def em(clock):
    em, client = session()
    em.client = client  # type: ignore[attr-defined]
    yield em
    server = em.__dict__.get("_ui")
    if server is not None:
        server.stop()


# ------------------------------------------------------------------ returning at once


def test_plot_returns_a_handle_at_once_with_its_data_registered(em, browser):
    handle = em.plot(plane="z-x", component="Ex", contours=True)
    (sent,) = em.client.calls("EM_plot")
    assert sent["plane"] == "z-x" and sent["component"] == "Ex"
    assert sent["pixels"] == list(ui.DEFAULT_PIXELS)
    assert "contours" not in sent  # a figure argument stays here
    assert isinstance(handle, ui.PlotHandle)
    assert handle.id == "p1"
    assert handle.view.plane == "z-x"
    assert em._views.get("p1").data is handle.view
    assert len(browser) == 1 and browser[0].endswith("#/plots/p1")


def test_an_unknown_argument_is_refused_before_anything_is_sent(em):
    with pytest.raises(ArgumentError, match="colour"):
        em.plot(colour="red")
    assert em.client.calls("EM_plot") == []


def test_the_first_plot_opens_the_tab_and_a_second_tells_it_while_it_lives(em, browser, clock):
    em.plot()
    server = em._ui
    server.ping()
    published: List[Any] = []
    q = server.subscribe()
    em.plot(plane="x-y")
    assert len(browser) == 1
    assert q.get_nowait() == ("view-added", em._views.get("p2").summary())


def test_a_tab_silent_for_five_seconds_is_opened_again(em, browser, clock):
    em.plot()
    em._ui.ping()
    clock.sleep(4.0)
    em.plot()
    assert len(browser) == 1
    clock.sleep(5.5)
    em.plot()
    assert len(browser) == 2


def test_a_tab_still_starting_is_not_opened_twice(em, browser, clock):
    em.plot()
    clock.sleep(2.0)  # the browser is still starting: no ping yet
    em.plot()
    assert len(browser) == 1


def test_block_returns_when_the_tab_reports_closed(em, browser, clock):
    def tab(now):
        server = em._ui
        if now < 1003:
            server.ping()
        elif now >= 1003:
            server.tab_closed = True

    clock.on_sleep.append(tab)
    em.plot(block=True)
    assert em._ui.tab_closed


def test_block_returns_when_the_heartbeat_stops(em, browser, clock):
    def tab(now):
        if now < 1002:
            em._ui.ping()

    clock.on_sleep.append(tab)
    em.plot(block=True)
    assert clock.now >= 1002 + 5.0 - 0.2


def test_block_returns_on_ctrl_c(em, browser, clock):
    def interrupt(now):
        raise KeyboardInterrupt

    clock.on_sleep.append(interrupt)
    assert isinstance(em.plot(block=True), ui.PlotHandle)


# ------------------------------------------------------------------ close() waits for the tab


def test_close_returns_at_once_when_every_view_was_fetched(em, browser, clock):
    em.plot()
    em._views.mark_fetched("p1")
    start = clock.now
    em.close()
    assert clock.now == start
    assert not em._ui.running


def test_close_waits_until_the_tab_fetches(em, browser, clock, capsys):
    em.plot()

    def tab(now):
        if now >= 1002:
            em._views.mark_fetched("p1")

    clock.on_sleep.append(tab)
    em.close()
    assert 1002 <= clock.now < 1003
    assert "waiting for the plot tab" in capsys.readouterr().out
    assert [d["function"] for d in em.client.sent][-1] == "EM_close"


def test_close_gives_up_at_the_exit_wait(em, browser, clock, monkeypatch):
    monkeypatch.setenv("EMODE_UI_EXIT_WAIT", "3")
    em.plot()
    em.close()
    assert clock.now == pytest.approx(1003.0, abs=0.2)
    assert not em._ui.running


def test_close_stops_waiting_when_the_tab_closes(em, browser, clock):
    em.plot()
    clock.on_sleep.append(lambda now: setattr(em._ui, "tab_closed", now >= 1001))
    em.close()
    assert clock.now < 1002


def test_close_does_not_wait_when_noninteractive(em, browser, clock, monkeypatch):
    monkeypatch.setenv("EMODE_NONINTERACTIVE", "1")
    handle = em.plot()
    assert browser == [] and em.__dict__.get("_ui") is None
    assert handle.view is not None
    start = clock.now
    em.close()
    assert clock.now == start


def test_close_does_not_wait_in_a_notebook(clock, browser):
    em, client = session(ipython=True)
    handle = em.plot()
    assert browser == []
    html = handle._repr_html_()
    assert "<iframe" in html and "#/plots/p1" in html and "embed=1" in html
    start = clock.now
    em.close()
    assert clock.now == start


def test_close_without_a_plot_starts_no_server(em, clock):
    em.close()
    assert em.__dict__.get("_ui") is None


# ------------------------------------------------------------------ headless


def _png_size(png: bytes) -> tuple:
    from matplotlib.image import imread

    return imread(io.BytesIO(png)).shape[:2]


@pytest.mark.parametrize("how", ["env", "file_name", "linux"])
def test_headless_starts_no_server_and_returns_an_image(em, browser, monkeypatch, tmp_path, how):
    pytest.importorskip("matplotlib")
    kwargs: dict = {}
    if how == "env":
        monkeypatch.setenv("EMODE_HEADLESS", "1")
    elif how == "file_name":
        kwargs = {"file_name": "field", "file_type": "pdf"}
    else:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    image = em.plot(**kwargs)
    assert isinstance(image, PlotImage)
    assert em.__dict__.get("_ui") is None and browser == []
    assert image.dpi == 150
    if how == "file_name":
        # a saved file outside a notebook carries no inline copy, as PlotImage says
        assert image.png is None
        assert image.file == str((tmp_path / "EM_figures" / "field.pdf").resolve())
        assert Path(image.file).read_bytes().startswith(b"%PDF")
        return
    assert image.file is None
    png = image.png.tobytes()
    assert png.startswith(b"\x89PNG")
    height, width = _png_size(png)
    # the 80 mm figure at 150 dpi, cropped to its drawing
    assert 300 < width < 80 / 25.4 * 150 * 1.3 and height > 150


def test_a_saved_file_keeps_its_extension(em, browser):
    pytest.importorskip("matplotlib")
    image = em.plot(file_name="plot.png", file_type="png")
    assert image.file.endswith(os.path.join("EM_figures", "plot.png"))


def test_a_saved_file_in_a_notebook_is_shown_too(clock, monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    shown: List[Any] = []
    display_module = type(sys)("IPython.display")
    display_module.display = shown.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", type(sys)("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    em, client = session(ipython=True)
    assert em.plot(file_name="index", file_type="png") is None
    (image,) = shown
    assert image.png is not None and image.file.endswith("index.png")


def test_inline_png_in_a_notebook_displays_the_image(clock, monkeypatch):
    pytest.importorskip("matplotlib")
    shown: List[Any] = []
    display_module = type(sys)("IPython.display")
    display_module.display = shown.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", type(sys)("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    em, client = session(ipython=True)
    assert em.plot(inline="png") is None
    (image,) = shown
    assert isinstance(image, PlotImage) and image.png is not None


def test_the_handle_saves_and_renders_a_png(em, browser, tmp_path):
    pytest.importorskip("matplotlib")
    handle = em.plot()
    assert handle.png(dpi=100).startswith(b"\x89PNG")
    path = handle.save("panel", "svg")
    assert path == (tmp_path / "EM_figures" / "panel.svg").resolve()
    assert b"<svg" in path.read_bytes()[:500]


def test_a_schematic_registers_as_a_schematic_view(browser, clock):
    em, client = session(plot=lambda request: schematic())
    handle = em.plot(view="schematic")
    assert handle.id == "s1"
    assert em._views.get("s1").kind == "schematic"
    em.close()


# ------------------------------------------------------------------ the pairing


def test_an_emode_that_draws_its_own_window_is_refused_before_any_call(clock):
    em, client = session(version=VERSION_1_0_8)
    with pytest.raises(EModeError, match="plot\\(\\) needs EMode 1.0.9 or newer, and this EMode is 1.0.8. Update it with emode-install"):
        em.plot()
    assert client.calls("EM_plot") == []


def test_an_emode_older_than_1_0_7_is_refused_too(clock):
    em, client = session(version=None)
    client.answers["EM_version"] = lambda request: (_ for _ in ()).throw(EModeError("not a supported EMode function"))
    with pytest.raises(EModeError, match="older than 1.0.7"):
        em.plot()


def test_a_reply_that_is_not_a_plot_is_reported(clock):
    em, client = session(plot=lambda request: "ok")
    with pytest.raises(EModeError, match="pip install -U emode"):
        em.plot()


def test_the_version_is_asked_once_a_session(em, browser):
    em.plot()
    em.plot()
    assert len(em.client.calls("EM_version")) == 1
