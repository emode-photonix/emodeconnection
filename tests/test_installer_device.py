"""The installer's browser sign-in (emode-photonix/emode#470), with every HTTP call faked."""

from __future__ import annotations

from typing import Any

import pytest

from emode import installer

START = {
    "device_code": "dev123",
    "user_code": "ABCD-EFGH",
    "verification_url": "https://example.com/device/",
    "verification_url_complete": "https://example.com/device/?user_code=ABCD-EFGH",
    "interval": 5,
    "expires_in": 600,
}
APPROVED = {
    "status": "approved",
    "downloadToken": "tok",
    "firstName": "Ada",
    "lastName": "L",
}


class FakeResponse:
    def __init__(
        self, status_code: int = 200, body: Any = None, json_ok: bool = True
    ) -> None:
        self.status_code = status_code
        self._body = body
        self._json_ok = json_ok

    def json(self) -> Any:
        if not self._json_ok:
            raise ValueError("not json")
        return self._body


class Net:
    """A fake website: the code reply, then the token replies in order, and a mocked clock."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, code: FakeResponse, tokens: list
    ) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.sleeps: list[float] = []
        self.now = 1000.0
        self.opened: list[str] = []
        self.browser_ok = True
        self._code = code
        self._tokens = list(tokens)
        monkeypatch.setattr(installer.http_requests, "post", self._post)
        monkeypatch.setattr(installer.time, "sleep", self._sleep)
        monkeypatch.setattr(installer.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(installer.webbrowser, "open", self._open)

    def _post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.posts.append((url, json))
        if url.endswith("/installer-device/code"):
            return self._code
        return FakeResponse(200, self._tokens.pop(0))

    def _sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s

    def _open(self, url: str) -> bool:
        self.opened.append(url)
        return self.browser_ok


def net(
    monkeypatch: pytest.MonkeyPatch, tokens: list, code: FakeResponse | None = None
) -> Net:
    return Net(monkeypatch, code or FakeResponse(200, START), tokens)


def test_approved_returns_download_token(monkeypatch, capsys):
    n = net(monkeypatch, [APPROVED])
    assert installer.device_login() == "tok"
    code_url, payload = n.posts[0]
    assert code_url.endswith("/installer-device/code")
    assert len(payload["machine_id"]) == 64
    assert payload["machine_id"] == installer.device_machine_id()
    assert n.posts[1][1] == {"device_code": "dev123"}
    assert "Logged in as Ada L." in capsys.readouterr().out


def test_pending_then_approved_honors_interval_and_slow_down(monkeypatch):
    n = net(
        monkeypatch,
        [
            {"status": "pending"},
            {"status": "slow_down", "interval": 10},
            {"status": "pending"},
            APPROVED,
        ],
    )
    assert installer.device_login() == "tok"
    assert n.sleeps == [5, 5, 10, 10]


@pytest.mark.parametrize(
    ("status", "text"), [("denied", "denied"), ("expired", "expired")]
)
def test_denied_and_expired_messages(monkeypatch, capsys, status, text):
    net(monkeypatch, [{"status": status}])
    with pytest.raises(SystemExit):
        installer.device_login()
    assert text in capsys.readouterr().out


def test_local_deadline_expires(monkeypatch, capsys):
    short = dict(START, expires_in=8)
    net(monkeypatch, [{"status": "pending"}] * 5, FakeResponse(200, short))
    with pytest.raises(SystemExit):
        installer.device_login()
    assert "expired" in capsys.readouterr().out


def test_browser_opened(monkeypatch):
    n = net(monkeypatch, [APPROVED])
    installer.device_login()
    assert n.opened == [START["verification_url_complete"]]


def test_no_browser_prints_code_and_url(monkeypatch, capsys):
    n = net(monkeypatch, [APPROVED])
    n.browser_ok = False
    installer.device_login()
    out = capsys.readouterr().out
    assert "ABCD-EFGH" in out
    assert START["verification_url_complete"] in out


@pytest.mark.parametrize(
    "code",
    [
        FakeResponse(404, {"error": "no"}),
        FakeResponse(200, None, json_ok=False),
        FakeResponse(502, {"error": "Sign-in is unavailable right now."}),
        FakeResponse(429, {"error": "Too many sign-in requests. Try again later."}),
    ],
)
def test_older_website_falls_back_to_password(monkeypatch, capsys, code):
    net(monkeypatch, [], code)
    monkeypatch.setattr(installer.sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("EMODE_EMAIL", raising=False)
    monkeypatch.delenv("EMODE_PASSWORD", raising=False)
    monkeypatch.setattr(installer, "prompt_credentials", lambda: ("a@b.c", "pw"))
    monkeypatch.setattr(installer, "login", lambda e, p: f"pw-token-{e}")
    assert installer.authenticate() == "pw-token-a@b.c"
    assert "password" in capsys.readouterr().out


def test_env_password_path_skips_browser(monkeypatch):
    n = net(monkeypatch, [])
    monkeypatch.setenv("EMODE_EMAIL", "a@b.c")
    monkeypatch.setenv("EMODE_PASSWORD", "pw")
    monkeypatch.setattr(installer, "login", lambda e, p: "pw-token")
    assert installer.authenticate() == "pw-token"
    assert n.posts == []


def test_password_flag_skips_browser(monkeypatch):
    n = net(monkeypatch, [])
    monkeypatch.delenv("EMODE_EMAIL", raising=False)
    monkeypatch.setattr(installer, "prompt_credentials", lambda: ("a@b.c", "pw"))
    monkeypatch.setattr(installer, "login", lambda e, p: "pw-token")
    assert installer.authenticate(password=True) == "pw-token"
    assert n.posts == []


def test_user_chooses_password(monkeypatch):
    n = net(monkeypatch, [])
    monkeypatch.delenv("EMODE_EMAIL", raising=False)
    monkeypatch.delenv("EMODE_PASSWORD", raising=False)
    monkeypatch.setattr(installer.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "p")
    monkeypatch.setattr(installer, "prompt_credentials", lambda: ("a@b.c", "pw"))
    monkeypatch.setattr(installer, "login", lambda e, p: "pw-token")
    assert installer.authenticate() == "pw-token"
    assert n.posts == []


def test_default_is_browser(monkeypatch):
    net(monkeypatch, [APPROVED])
    monkeypatch.delenv("EMODE_EMAIL", raising=False)
    monkeypatch.delenv("EMODE_PASSWORD", raising=False)
    monkeypatch.setattr(installer.sys.stdin, "isatty", lambda: False)
    assert installer.authenticate() == "tok"


# ---------------------------------------------------------------------------
# A password refused because the account has two-factor on (emode#470)
# ---------------------------------------------------------------------------

TWO_FACTOR = {
    "error": (
        "This account uses two-factor authentication, so it signs in with your browser. "
        "Update the installer with: pip install --upgrade emode, then run emode-install "
        "again and sign in with your browser."
    ),
    "code": "two_factor_required",
}


class LoginResponse(FakeResponse):
    """A reply to /installer-login/, which the installer checks with raise_for_status."""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise installer.http_requests.exceptions.HTTPError(response=self)


class Website(Net):
    """The fake website, also answering /installer-login/ with `login`."""

    def __init__(self, monkeypatch, login: LoginResponse, code=None, tokens=None) -> None:
        super().__init__(monkeypatch, code or FakeResponse(200, START), list(tokens or []))
        self._login = login

    def _post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        if url.endswith("/installer-login/"):
            self.posts.append((url, {"email": json["email"]}))  # never record the password
            return self._login
        return super()._post(url, json, timeout)


def test_login_refused_for_two_factor_raises_two_factor_required(monkeypatch):
    Website(monkeypatch, LoginResponse(403, TWO_FACTOR))
    with pytest.raises(installer.TwoFactorRequired) as caught:
        installer.login("a@b.c", "pw")
    assert str(caught.value) == TWO_FACTOR["error"]


def test_an_ordinary_refusal_still_exits_with_the_message(monkeypatch, capsys):
    Website(monkeypatch, LoginResponse(403, {"error": "The password is incorrect."}))
    with pytest.raises(SystemExit):
        installer.login("a@b.c", "pw")
    assert "Authentication failed: The password is incorrect." in capsys.readouterr().out


def test_a_password_login_still_returns_the_token(monkeypatch):
    Website(monkeypatch, LoginResponse(200, {"downloadToken": "pw-tok", "firstName": "Ada"}))
    assert installer.login("a@b.c", "pw") == "pw-tok"


def test_the_password_flag_switches_to_the_browser_for_two_factor(monkeypatch, capsys):
    site = Website(monkeypatch, LoginResponse(403, TWO_FACTOR), tokens=[APPROVED])
    monkeypatch.delenv("EMODE_EMAIL", raising=False)
    monkeypatch.setattr(installer, "prompt_credentials", lambda: ("a@b.c", "pw"))
    assert installer.authenticate(password=True) == "tok"
    out = capsys.readouterr().out
    assert TWO_FACTOR["error"] in out
    assert "browser" in out.split(TWO_FACTOR["error"], 1)[1]
    urls = [url for url, _ in site.posts]
    assert urls[0].endswith("/installer-login/")
    assert urls[1].endswith("/installer-device/code")


def test_the_env_password_switches_to_the_browser_for_two_factor(monkeypatch):
    site = Website(monkeypatch, LoginResponse(403, TWO_FACTOR), tokens=[APPROVED])
    monkeypatch.setenv("EMODE_EMAIL", "a@b.c")
    monkeypatch.setenv("EMODE_PASSWORD", "pw")
    assert installer.authenticate() == "tok"
    assert site.opened == [START["verification_url_complete"]]


def test_the_switch_exits_when_the_website_has_no_browser_sign_in(monkeypatch, capsys):
    Website(monkeypatch, LoginResponse(403, TWO_FACTOR), code=FakeResponse(404, {"error": "no"}))
    monkeypatch.delenv("EMODE_EMAIL", raising=False)
    monkeypatch.setattr(installer, "prompt_credentials", lambda: ("a@b.c", "pw"))
    with pytest.raises(SystemExit) as caught:
        installer.authenticate(password=True)
    assert caught.value.code == 1
    assert TWO_FACTOR["error"] in capsys.readouterr().out
