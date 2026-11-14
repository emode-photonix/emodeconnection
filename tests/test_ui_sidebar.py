"""The plot tab's collapsible sidebar: the page, its style rules and its script carry it.

No browser runs here, so these check the static files for the pieces the behavior
needs: the toggle button and its aria state, the narrow-screen rule, the guarded
localStorage choice, the drawer's Escape and the resize of the shown plot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "emode" / "ui"


@pytest.fixture(scope="module")
def html() -> str:
    return (UI / "pages" / "app.html").read_text()


@pytest.fixture(scope="module")
def css() -> str:
    return (UI / "static" / "emode.css").read_text()


@pytest.fixture(scope="module")
def js() -> str:
    return (UI / "static" / "app.js").read_text()


def test_the_page_carries_a_toggle_button_for_the_list(html):
    button = re.search(r"<button\b[^>]*id=\"sidebar-toggle\"[^>]*>", html)
    assert button, "no button element with id sidebar-toggle"
    assert "aria-expanded=" in button.group(0)
    assert 'aria-controls="list"' in button.group(0)
    assert 'type="button"' in button.group(0)
    assert 'id="list"' in html


def test_the_page_has_no_inline_script_or_handler(html):
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert not re.search(r"\son\w+=", html)


def test_the_style_has_the_narrow_screen_rule_and_a_visible_focus(css):
    assert re.search(r"@media\s*\(max-width:\s*800px\)", css)
    assert "#sidebar-toggle:focus-visible" in css
    assert "sidebar-collapsed" in css


def test_the_narrow_rule_makes_the_list_a_drawer_over_the_plot(css):
    narrow = css[css.index("@media (max-width: 800px)") :]
    assert re.search(r"#list\s*\{[^}]*position:\s*absolute", narrow)


def test_the_phone_page_does_not_scroll_sideways(css):
    assert re.search(r"overflow-x:\s*hidden", css)
    narrow = css[css.index("@media (max-width: 800px)") :]
    assert "#layer-panel" in narrow
    assert "max-width" in narrow


def test_the_script_remembers_the_choice_under_a_named_key_inside_try(js):
    assert re.search(r'SIDEBAR_KEY\s*=\s*"emode\.sidebar"', js)
    for call in (
        "localStorage.getItem(SIDEBAR_KEY)",
        "localStorage.setItem(SIDEBAR_KEY",
    ):
        at = js.index(call)
        assert (
            js.rfind("try {", 0, at) > js.rfind("}", 0, at)
            or "try {" in js[at - 120 : at]
        ), call
    assert js.count("localStorage") == 2


def test_the_script_starts_narrow_screens_collapsed_and_tracks_the_width(js):
    assert 'matchMedia("(max-width: 800px)")' in js


def test_the_script_sets_aria_expanded_and_resizes_the_plot_after_the_transition(js):
    assert 'setAttribute("aria-expanded"' in js
    start = js.index("function setSidebar")
    body = js[start : js.index("\n}\n", start)]
    # the plot is refit to its container's width (emodeconnection#160)
    assert "fitFigure()" in body
    assert "setTimeout" in body


def test_picking_a_plot_closes_the_drawer_and_escape_closes_it(js):
    click = js[js.index("li.addEventListener") :]
    click = click[: click.index("\n")]
    assert "closeDrawer()" in click
    esc = js[js.index("function closeDrawer") :]
    assert "narrow" in esc[:300]
    assert re.search(r'ev\.key === "Escape"[^\n]*(\n[^\n]*){0,4}closeDrawer\(\)', js)
