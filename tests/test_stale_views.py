"""A view opened after the session changed is fetched again, so its menus are the session's now (#182).

Eric, reviewing the live demo (2026-11-15): "At first, the profile drop down
only shows FDM and single". The demo's first ``plot()`` ran when only the
profile ``single`` existed, and the tab showed that view's menus as EMode
answered them then: the profiles and the EME sections declared later never
appeared until a control changed.

Every call that can change the session counts a generation (``EMode.call``),
and reads (``plot``, ``get``, the session page, the details, the version, the
login calls) do not. A view fetched before the current generation is fetched
again, with its own arguments, when the tab shows it while the session lives.
After the session has ended it is shown as it was.
"""

from __future__ import annotations

import http.client
import json
from typing import Any, Dict, List, Tuple

import pytest

from emode.types import ArgumentError
from emode.ui import session as ui

from plot_fixtures import xy_view
from plot_standin import session


@pytest.fixture(autouse=True)
def desktop(monkeypatch, tmp_path):
    for name in ("EMODE_HEADLESS", "EMODE_NONINTERACTIVE", "EMODE_PLOT_INLINE", "EMODE_UI_EXIT_WAIT", "EMODE_UI_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui.webbrowser, "open", lambda url: True)


class Session:
    """What the stand-in EMode holds: the labeled profiles and whether sections are declared."""

    def __init__(self) -> None:
        self.profiles: List[str] = ["single"]
        self.sections = False

    def plot(self, request: Dict[str, Any]) -> Any:
        profile = request.get("profile")
        if profile is not None and profile not in ["FDM", *self.profiles]:
            raise ArgumentError(f"profile {profile!r} not found", "plot", "profile")
        view = xy_view()
        profiles = (["EME"] if self.sections else []) + ["FDM", *self.profiles]
        selected = {**view.menu.selected, "profile": profile or "FDM"}
        return view.model_copy(
            update={"menu": view.menu.model_copy(update={"profiles": profiles, "selected": selected})}
        )

    def label(self, request: Dict[str, Any]) -> None:
        self.profiles.append(request["name"])

    def section(self, request: Dict[str, Any]) -> None:
        self.sections = True

    def forget(self, request: Dict[str, Any]) -> None:
        self.profiles.remove(request["name"])


@pytest.fixture
def em():
    state = Session()
    em, client = session(plot=state.plot)
    client.answers.update(
        {
            "EM_label_profile": state.label,
            "EM_straight_section": state.section,
            "EM_delete_profile": state.forget,
            "EM_get": lambda request: 1.5,
        }
    )
    em.client = client  # type: ignore[attr-defined]
    yield em
    server = em.__dict__.get("_ui")
    if server is not None:
        server.stop()


def _open(em: Any, view_id: str) -> Dict[str, Any]:
    """What the tab gets when it shows `view_id`: a post with no control changed."""
    server = em.__dict__["_ui"]
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    body = json.dumps({"controls": {}, "pixels": [1170, 720]}).encode()
    headers = {"Host": f"127.0.0.1:{server.port}", "X-EMode-Token": server.token, "Content-Type": "application/json"}
    conn.request("POST", f"/api/views/{view_id}/data", body=body, headers=headers)
    reply = json.loads(conn.getresponse().read())
    conn.close()
    return reply


def _plot_calls(em: Any) -> int:
    return len(em.client.calls("EM_plot"))


def test_a_view_opened_after_the_session_changed_shows_its_menus_now(em):
    handle = em.plot(plane="x-y")
    em.label_profile(name="pair")
    em.straight_section(name="in", profile="single", length=2000)
    reply = _open(em, handle.id)
    assert "error" not in reply, reply
    assert reply["menu"]["profiles"] == ["EME", "FDM", "single", "pair"]
    # with the plot's own arguments
    assert em.client.calls("EM_plot")[-1]["plane"] == "x-y"


def test_the_users_selection_is_kept(em):
    handle = em.plot(plane="x-y", profile="single", component="Ex")
    em.label_profile(name="pair")
    before = _plot_calls(em)
    reply = _open(em, handle.id)
    assert _plot_calls(em) == before + 1
    last = em.client.calls("EM_plot")[-1]
    assert last["profile"] == "single" and last["component"] == "Ex"
    assert reply["menu"]["selected"]["profile"] == "single"


def test_an_up_to_date_view_is_not_fetched_again(em):
    handle = em.plot(plane="x-y")
    em.label_profile(name="pair")
    _open(em, handle.id)
    before = _plot_calls(em)
    _open(em, handle.id)
    assert _plot_calls(em) == before


def test_a_read_does_not_make_a_view_stale(em):
    handle = em.plot(plane="x-y")
    em.get("effective_index")
    em.plot(plane="z-x")
    before = _plot_calls(em)
    _open(em, handle.id)
    assert _plot_calls(em) == before


def test_an_ended_session_shows_the_view_as_it_was(em):
    handle = em.plot(plane="x-y")
    em.label_profile(name="pair")
    em.__dict__["_ui"].session_ended = True
    before = _plot_calls(em)
    reply = _open(em, handle.id)
    assert _plot_calls(em) == before
    assert reply["menu"]["profiles"] == ["FDM", "single"]


def test_a_profile_that_is_gone_is_said_and_the_view_shown_as_it_was(em):
    handle = em.plot(plane="x-y", profile="single")
    em.delete_profile(name="single")
    reply = _open(em, handle.id)
    assert "error" not in reply, reply
    assert "single" in reply["notice"] and "not found" in reply["notice"]
    assert reply["menu"]["selected"]["profile"] == "single"


def test_the_tab_shows_the_notice():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "emode" / "ui" / "static" / "app.js").read_text()
    start = js.index("async function show(")
    body = js[start : js.index("\n}\n", start)]
    assert "reply.notice" in body
