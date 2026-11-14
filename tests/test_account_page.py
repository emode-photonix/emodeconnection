"""The account page: the login window's every function in the session's browser tab (GUI-M3).

EMode answers a session start with a ``LicenseError`` of code
``'login_required'`` when nothing signs in, and keeps the connection open for
the account page's calls (browser-interface note, section 10). The page is a
view of that state: the password form and the browser sign-in (the device code
of a two-factor account) while signed out, and the account panel once signed in
(the counts, the sharing line, the license and session tables with the per-row
clear and the clear modes, the error-reporting setting, the update notice, Log
Out). The page posts to the client's local server, which forwards each post to
EMode as a call. A password travels in a POST body only, never in a URL, a log
line or a reply, and every route needs the session's token, its ``Host`` and no
foreign ``Origin``.
"""
from __future__ import annotations

import http.client
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlsplit

import pytest

import emode.ui.account as account
from emode.emodeclient import EModeClient
from emode.types import LicenseError
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from js_harness import needs_jsc, run_app
from login_standin import EMAIL, LOGIN_INFO, PASSWORD, REFUSALS, USER_CODE, FakeAccount, login_required
from plot_standin import session
from test_ui_server import request

UI = Path(__file__).resolve().parent.parent / "emode" / "ui"


def _server(fake: FakeAccount) -> UIServer:
    return UIServer(ViewRegistry(), lambda *a: None, login=fake).start()


@pytest.fixture
def signed_out():
    fake = FakeAccount()
    server = _server(fake)
    yield server, fake
    server.stop()


@pytest.fixture
def signed_in():
    fake = FakeAccount(signed_in=True)
    server = _server(fake)
    yield server, fake
    server.stop()


def post(server: UIServer, path: str, body: Dict[str, Any], **kw: Any):
    status, headers, payload = request(server, "POST", path, body=body, **kw)
    return status, json.loads(payload) if payload else None


def get(server: UIServer, path: str, **kw: Any):
    status, headers, payload = request(server, "GET", path, **kw)
    return status, json.loads(payload) if payload else None


# ------------------------------------------------------------------ the page


@pytest.fixture(scope="module")
def html() -> str:
    return (UI / "pages" / "app.html").read_text()


def _ids(html: str) -> set:
    return set(re.findall(r'\bid="([\w-]+)"', html))


def test_the_nav_has_an_account_page(html):
    assert re.search(r'<a href="#/account"[^>]*data-route="account"', html)
    assert 'id="account"' in html


def test_the_signed_out_page_has_every_field_of_the_login_window(html):
    ids = _ids(html)
    # the logo and version, the email and password, the show and hide toggle,
    # the forgotten password link, the error row and the Log In button
    for element in (
        "account-version",
        "login-form",
        "login-email",
        "login-password",
        "password-toggle",
        "forgot-password",
        "login-error",
        "login-submit",
    ):
        assert element in ids, element
    assert re.search(r'<input[^>]*id="login-email"[^>]*type="email"', html)
    assert re.search(r'<input[^>]*id="login-password"[^>]*type="password"', html)
    assert re.search(r'<input[^>]*id="login-password"[^>]*autocomplete="current-password"', html)
    # the browser sign-in for a two-factor account: the button, the code, the link, a retry
    for element in ("device-start", "device-panel", "device-code", "device-link", "device-retry"):
        assert element in ids, element


def test_the_form_posts_nowhere_on_its_own(html):
    """The script sends the form as a POST body, so the browser never puts the password in a URL."""
    form = re.search(r"<form\b[^>]*id=\"login-form\"[^>]*>", html).group(0)
    assert 'method="post"' in form
    assert "action=" not in form
    assert not re.search(r'<input[^>]*id="login-password"[^>]*\bvalue=', html)


def test_the_signed_in_page_has_every_field_of_the_login_window(html):
    ids = _ids(html)
    for element in (
        "account-email",
        "logout",
        "count-2d",
        "count-3d",
        "sharing-link",
        "sharing-line",
        "license-table",
        "license-rows",
        "session-list",
        "session-rows",
        "clear-mode",
        "clear-sessions",
        "refresh-sessions",
        "error-reporting",
        "error-reporting-explanation",
        "update-notice",
        "my-account",
    ):
        assert element in ids, element
    for heading in ("Subscription", "Status", "Expiration", "Owner"):
        assert f"<th>{heading}</th>" in html
    for heading in ("Type", "User", "Computer", "Started", "Owner"):
        assert f"<th>{heading}</th>" in html
    clear = re.search(r'<select id="clear-mode"[^>]*>(.*?)</select>', html, re.S).group(1)
    assert re.findall(r'<option value="(\w+)"', clear) == ["none", "all", "mine", "others", "latest", "oldest"]


# ------------------------------------------------------------------ the account panel


def _local(iso: str) -> str:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).astimezone().strftime("%b %d, %H:%M")


def test_the_account_view_counts_shared_seats_as_the_window_did():
    view = account.account_view(LOGIN_INFO)
    # one own 3D seat, a shared 2D pool of two seats, one session of each checked out
    assert view["counts"] == {"2D": "1 / 2", "3D": "1 / 1"}


def test_the_account_view_lists_own_and_shared_licenses():
    view = account.account_view(LOGIN_INFO)
    assert view["licenses"] == [
        {"subscription": "EMode3D", "status": "Active", "expiration": "Jan 31, 2027", "owner": EMAIL, "shared": False},
        {"subscription": "Extra2D", "status": "Shared", "expiration": "Mar 01, 2027", "owner": "boss@example.com", "shared": True},
    ]


def test_the_account_view_lists_sessions_and_which_rows_clear():
    view = account.account_view(LOGIN_INFO)
    first, second = view["sessions"]
    assert first == {
        "type": "3D",
        "user": EMAIL,
        "computer": "laptop",
        "started": _local("2026-11-13T12:00:00"),
        "owner": EMAIL,
        "shared": False,
        "clearable": True,
        "handle": "h1",
        "confirm": (
            f"Clear the 3D session of {EMAIL} on laptop, started {_local('2026-11-13T12:00:00')}?"
            "\n\nA simulation still running in that session loses its license."
        ),
    }
    # no handle from the license manager: this user may not clear it
    assert second["clearable"] is False and second["shared"] is True and second["handle"] is None


def test_a_roaming_session_never_clears():
    info = dict(LOGIN_INFO, active_checkouts=[dict(LOGIN_INFO["active_checkouts"][0], roaming=True)])
    assert account.account_view(info)["sessions"][0]["clearable"] is False


def test_the_account_view_has_the_sharing_line_the_settings_and_the_update():
    view = account.account_view(LOGIN_INFO)
    assert view["sharing"] == {"link": account.MY_ACCOUNT_URL, "line": "You are sharing with 2 accounts."}
    assert view["error_reporting"] is True
    assert view["update_notice"].startswith("A newer version of EMode is available: 1.1.0")
    assert view["email"] == EMAIL
    assert view["version"] == "1.0.9"
    no_shares = account.account_view(dict(LOGIN_INFO, shared_by_me=[], shared_to_me=[]))
    assert no_shares["sharing"] == {"link": None, "line": None}


# ------------------------------------------------------------------ the routes


def test_the_status_route_answers_the_state(signed_out):
    server, fake = signed_out
    status, reply = get(server, "/api/login/status")
    assert status == 200
    assert reply["state"] == "signed_out"
    assert fake.calls == [("EM_login_status", {})]


def test_a_correct_password_signs_in(signed_out):
    server, fake = signed_out
    events = server.subscribe()
    status, reply = post(server, "/api/login/password", {"email": EMAIL, "password": PASSWORD})
    assert status == 200
    assert reply["state"] == "logged_in"
    assert fake.calls[-1] == ("EM_login", {"email": EMAIL, "password": PASSWORD})
    assert events.get(timeout=2) == ("login-changed", {"state": "logged_in"})


@pytest.mark.parametrize(
    "email, password, message",
    [
        (EMAIL, "wrong", REFUSALS[EMAIL]),
        ("nobody@example.com", PASSWORD, REFUSALS["nobody@example.com"]),
    ],
)
def test_a_refused_password_shows_the_license_managers_message(signed_out, email, password, message):
    server, fake = signed_out
    status, reply = post(server, "/api/login/password", {"email": email, "password": password})
    assert status == 200
    assert reply["state"] == "signed_out"
    assert reply["error"] == message


def test_an_unreachable_license_manager_is_an_error_row():
    fake = FakeAccount(unreachable=True)
    server = _server(fake)
    try:
        status, reply = post(server, "/api/login/password", {"email": EMAIL, "password": PASSWORD})
    finally:
        server.stop()
    assert status == 200
    assert "could not reach the license manager" in reply["error"]
    assert reply["state"] == "signed_out"


@pytest.mark.parametrize("body", [{"email": "", "password": PASSWORD}, {"email": EMAIL, "password": ""}, {}])
def test_an_empty_field_is_refused_before_emode_is_asked(signed_out, body):
    server, fake = signed_out
    status, reply = post(server, "/api/login/password", body)
    assert status == 200
    assert reply["error"] == "Enter your email and password."
    assert fake.calls == []


def test_the_password_is_never_in_a_reply_a_url_or_a_log(signed_out, caplog):
    server, fake = signed_out
    caplog.set_level(logging.DEBUG)
    _, refused = post(server, "/api/login/password", {"email": EMAIL, "password": "hunter2-wrong"})
    _, accepted = post(server, "/api/login/password", {"email": EMAIL, "password": PASSWORD})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert PASSWORD not in text and "hunter2-wrong" not in text
    assert PASSWORD not in json.dumps([refused, accepted])
    assert "hunter2-wrong" not in json.dumps([refused, accepted])


def test_the_password_route_takes_no_get(signed_out):
    server, fake = signed_out
    status, _ = get(server, f"/api/login/password?email={EMAIL}&password={PASSWORD}")
    assert status == 404
    assert fake.calls == []


@pytest.mark.parametrize(
    "method, path",
    [("GET", "/api/login/status"), ("GET", "/api/login/info"), ("POST", "/api/login/password"),
     ("POST", "/api/login/start"), ("POST", "/api/login/clear"), ("POST", "/api/login/logout"),
     ("POST", "/api/login/settings")],
)
@pytest.mark.parametrize("guard", ["no token", "wrong token", "foreign origin", "foreign host"])
def test_a_request_without_the_token_or_from_elsewhere_is_refused(signed_in, method, path, guard):
    server, fake = signed_in
    kw: Dict[str, Any] = {}
    if guard == "no token":
        kw["token"] = None
    elif guard == "wrong token":
        kw["token"] = "x" * 43
    elif guard == "foreign origin":
        kw["origin"] = "https://evil.example"
    else:
        kw["host"] = "evil.example"
    body = {"email": EMAIL, "password": PASSWORD} if method == "POST" else None
    status, headers, payload = request(server, method, path, body=body, **kw)
    assert status == 403 and payload == b""
    assert fake.calls == []


def test_the_browser_sign_in_shows_the_code_then_signs_in(signed_out):
    """A two-factor account signs in on the website: the page shows the code and polls the status."""
    server, fake = signed_out
    _, started = post(server, "/api/login/start", {})
    assert started["state"] == "pending"
    assert started["user_code"] == USER_CODE
    assert started["verification_url_complete"].endswith(USER_CODE)
    assert fake.calls[-1] == ("EM_login_start", {"method": "device"})
    _, pending = get(server, "/api/login/status")
    assert pending["state"] == "pending"
    fake.device = "approved"  # the user approved the code on the website, behind two-factor
    events = server.subscribe()
    _, done = get(server, "/api/login/status")
    assert done["state"] == "logged_in"
    assert events.get(timeout=2) == ("login-changed", {"state": "logged_in"})


@pytest.mark.parametrize("outcome", ["denied", "expired"])
def test_a_declined_or_expired_code_is_shown_with_a_retry(signed_out, outcome):
    server, fake = signed_out
    post(server, "/api/login/start", {})
    fake.device = outcome
    _, reply = get(server, "/api/login/status")
    assert reply["state"] == outcome and reply["error"]
    _, again = post(server, "/api/login/start", {})
    assert again["state"] == "pending"


def test_the_info_route_answers_the_account_view(signed_in):
    server, fake = signed_in
    status, reply = get(server, "/api/login/info")
    assert status == 200
    assert reply == json.loads(json.dumps(account.account_view(LOGIN_INFO)))


def test_the_info_route_while_signed_out_says_so(signed_out):
    server, fake = signed_out
    status, reply = get(server, "/api/login/info")
    assert status == 200
    assert reply == {"state": "signed_out"}


def test_a_session_row_clears_by_its_handle(signed_in):
    server, fake = signed_in
    status, reply = post(server, "/api/login/clear", {"mode": "session", "session_handle": "h1"})
    assert status == 200
    assert fake.calls[-2] == ("EM_clear_session", {"mode": "session", "session_handle": "h1"})
    assert [s["handle"] for s in reply["sessions"]] == [None]
    assert reply["counts"]["3D"] == "0 / 1"


@pytest.mark.parametrize("mode", ["all", "mine", "others", "latest", "oldest"])
def test_each_clear_mode_forwards(signed_in, mode):
    server, fake = signed_in
    status, reply = post(server, "/api/login/clear", {"mode": mode})
    assert status == 200 and "sessions" in reply
    assert fake.calls[-2] == ("EM_clear_session", {"mode": mode})


def test_an_unknown_clear_mode_is_an_error(signed_in):
    server, fake = signed_in
    status, reply = post(server, "/api/login/clear", {"mode": "everything"})
    assert status == 200 and "error" in reply


def test_the_settings_route_sets_error_reporting(signed_in):
    server, fake = signed_in
    status, reply = post(server, "/api/login/settings", {"error_reporting": False})
    assert status == 200 and reply == {"error_reporting": False}
    assert fake.calls[-1] == ("EM_settings", {"error_reporting": False})


def test_log_out_returns_the_page_to_the_form(signed_in):
    server, fake = signed_in
    events = server.subscribe()
    status, reply = post(server, "/api/login/logout", {})
    assert status == 200 and reply["state"] == "signed_out"
    assert events.get(timeout=2) == ("login-changed", {"state": "signed_out"})
    assert get(server, "/api/login/info")[1] == {"state": "signed_out"}


def test_a_session_without_a_login_backend_answers_404():
    server = UIServer(ViewRegistry(), lambda *a: None).start()
    try:
        assert get(server, "/api/login/status")[0] == 404
    finally:
        server.stop()


def test_the_account_url_opens_the_account_page(signed_out):
    server, _ = signed_out
    url = server.url(page="account")
    assert url.endswith("#/account")
    assert parse_qs(urlsplit(url).query)["token"] == [server.token]


# ------------------------------------------------------------------ EMode() waits for the sign-in


def _login_session(fake: FakeAccount):
    """A stand-in session whose EMode answers its start with login_required until the page signs in."""
    em, client = session()

    def start(request: Dict[str, Any]) -> Any:
        if fake.state != "logged_in":
            raise login_required()
        return "sim:" + request.get("simulation_name", "emode")

    for function in ("EM_login_status", "EM_login_start", "EM_login", "EM_logout", "EM_login_info",
                     "EM_clear_session", "EM_settings"):
        client.answers[function] = (
            lambda request, f=function: fake(f, **{k: v for k, v in request.items()
                                                   if k not in ("function", "simulation_name")})
        )
    client.answers["EM_init"] = start
    return em, client


@pytest.fixture
def fast_wait(monkeypatch):
    monkeypatch.setattr(account, "LOGIN_POLL", 0.01)
    monkeypatch.delenv("EMODE_HEADLESS", raising=False)


def test_emode_opens_the_account_page_and_resumes_once_signed_in(fast_wait, monkeypatch):
    fake = FakeAccount()
    em, client = _login_session(fake)
    opened: List[str] = []

    def browser(url: str) -> bool:
        # the page signs in with the password form, as a user would
        opened.append(url)
        parts = urlsplit(url)
        token = parse_qs(parts.query)["token"][0]

        def sign_in() -> None:
            conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=5)
            conn.request("POST", "/api/login/password", body=json.dumps({"email": EMAIL, "password": PASSWORD}),
                         headers={"X-EMode-Token": token, "Host": parts.netloc, "Content-Type": "application/json"})
            conn.getresponse().read()
            conn.close()

        threading.Thread(target=sign_in, daemon=True).start()
        return True

    monkeypatch.setattr(account.webbrowser, "open", browser)
    try:
        reply = em._first_call("EM_init", simulation_name="mine", save_path=".")
    finally:
        ui = em.__dict__.get("_ui")
        if ui is not None:
            ui.stop()
    assert reply == "sim:mine"
    assert len(opened) == 1 and opened[0].endswith("#/account")
    assert [c["function"] for c in client.sent].count("EM_init") == 2
    assert PASSWORD not in opened[0]


def test_emode_without_a_browser_raises_the_error_naming_the_terminal_commands(fast_wait, monkeypatch):
    monkeypatch.setenv("EMODE_HEADLESS", "1")
    fake = FakeAccount()
    em, client = _login_session(fake)
    opened: List[str] = []
    monkeypatch.setattr(account.webbrowser, "open", lambda url: opened.append(url) or True)
    with pytest.raises(LicenseError) as caught:
        em._first_call("EM_init", simulation_name="mine", save_path=".")
    assert "emode login" in str(caught.value)
    assert opened == []


def test_emode_gives_up_when_the_browser_cannot_open(fast_wait, monkeypatch):
    fake = FakeAccount()
    em, client = _login_session(fake)
    monkeypatch.setattr(account.webbrowser, "open", lambda url: False)
    with pytest.raises(LicenseError) as caught:
        em._first_call("EM_init", simulation_name="mine", save_path=".")
    assert "python -m emode login" in str(caught.value)
    assert em.__dict__.get("_ui") is None or not em._ui.running


def test_a_refused_seat_is_raised_at_once(fast_wait, monkeypatch):
    em, client = session()

    def refuse(request: Dict[str, Any]) -> Any:
        error = LicenseError("No license is free.", None)
        error.code = "refused"  # type: ignore[attr-defined]
        raise error

    client.answers["EM_init"] = refuse
    monkeypatch.setattr(account.webbrowser, "open", lambda url: pytest.fail("no page for a refusal"))
    with pytest.raises(LicenseError, match="No license is free"):
        em._first_call("EM_init", simulation_name="mine", save_path=".")


# ------------------------------------------------------------------ the socket's debug log


def test_the_socket_client_never_logs_a_password(caplog):
    sent: List[bytes] = []

    class Sock:
        def sendall(self, data: bytes) -> None:
            sent.append(data)

    client = EModeClient.__new__(EModeClient)
    client.sock = Sock()  # type: ignore[assignment]
    client.ndarray_wire = False
    caplog.set_level(logging.DEBUG)
    client.send({"function": "EM_login", "email": EMAIL, "password": PASSWORD})
    assert PASSWORD.encode() in sent[0]  # it still goes to EMode
    assert PASSWORD not in "\n".join(r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ the page's script


@needs_jsc
def test_the_form_posts_the_password_in_the_body_never_the_url():
    got = run_app(
        f"""
        document.getElementById("login-email").value = "{EMAIL}";
        document.getElementById("login-password").value = "{PASSWORD}";
        signInWithPassword();
        return __posts;
        """
    )
    path, body = got[-1]
    assert path == "api/login/password"
    assert body == {"email": EMAIL, "password": PASSWORD}
    assert all(PASSWORD not in p for p, _ in got)


@needs_jsc
def test_the_password_toggle_shows_and_hides_it():
    got = run_app(
        """
        var input = document.getElementById("login-password");
        input.type = "password";
        togglePassword();
        var shown = input.type;
        togglePassword();
        return [shown, input.type];
        """
    )
    assert got == ["text", "password"]


@needs_jsc
def test_a_pending_sign_in_shows_the_code_and_its_link():
    got = run_app(
        f"""
        showLogin({{state: "pending", user_code: "{USER_CODE}", interval: 5,
                    verification_url_complete: "https://example.com/device/?user_code={USER_CODE}"}});
        return [document.getElementById("device-panel").hidden,
                document.getElementById("device-code").textContent,
                document.getElementById("device-link").getAttribute("href"),
                document.getElementById("signed-in").hidden];
        """
    )
    assert got == [False, USER_CODE, f"https://example.com/device/?user_code={USER_CODE}", True]


@needs_jsc
def test_the_account_panel_fills_its_tables_and_a_row_clears_after_a_yes():
    view = json.dumps(account.account_view(LOGIN_INFO))
    got = run_app(
        f"""
        var view = {view};
        showAccount(view);
        var sessions = document.getElementById("session-rows");
        var licenses = document.getElementById("license-rows");
        var asked = [];
        window.confirm = function (m) {{ asked.push(m); return asked.length > 1; }};
        clearSession(view.sessions[0]);   // the first answer is no
        var before = __posts.length;
        clearSession(view.sessions[0]);   // the second is yes
        return [document.getElementById("count-2d").textContent,
                document.getElementById("count-3d").textContent,
                document.getElementById("sharing-line").textContent,
                __find(licenses, "tr").length, __find(sessions, "tr").length,
                __find(sessions, "button").length,
                asked.length, before, __posts.slice(before)];
        """
    )
    count2, count3, line, license_rows, session_rows, buttons, asked, before, posts = got
    assert (count2, count3) == ("1 / 2", "1 / 1")
    assert line == "You are sharing with 2 accounts."
    assert license_rows == 2 and session_rows == 2
    assert buttons == 1  # only the row with a handle has the clear mark
    assert asked == 2
    assert posts == [["api/login/clear", {"mode": "session", "session_handle": "h1"}]]
