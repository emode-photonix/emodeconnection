"""A stand-in for EMode's login calls, for the account page's tests.

It answers ``login_status``, ``login_start``, ``login``, ``logout``,
``login_info``, ``clear_session`` and ``settings`` as EMode does while it waits
for a sign-in (browser-interface note, section 10): a state, the account's
licenses and sessions, the device code while a browser sign-in is pending, and
the license manager's message when it refuses a password. Nothing here talks to
a license manager.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from emode.types import ArgumentError, EModeError, LicenseError

EMAIL = "me@example.com"
PASSWORD = "correct horse battery staple"
USER_CODE = "WXYZ-2345"

LOGIN_INFO = {
    "email": EMAIL,
    "licenses": [{"type": "emode3d", "expires": "2027-01-31"}],
    "active_checkouts": [
        {
            "user": EMAIL,
            "owner": EMAIL,
            "lic_type": "3d",
            "comp_name": "laptop",
            "start_time": "2026-11-13T12:00:00",
            "roaming": False,
            "handle": "h1",
        },
        {
            "user": "colleague@example.com",
            "owner": "boss@example.com",
            "lic_type": "2d",
            "comp_name": "desk",
            "start_time": "2026-11-13T13:30:00",
            "roaming": False,
        },
    ],
    "shared_by_me": [
        {"member_email": "a@example.com"},
        {"member_email": "b@example.com"},
        {"member_email": "a@example.com"},
    ],
    "shared_to_me": [
        {
            "owner_email": "boss@example.com",
            "license_type": "extra2d",
            "pool": "2d",
            "total_seats": 2,
            "active_seats": 1,
            "expiration": "2027-03-01",
        }
    ],
    "version": "1.0.9",
    "latest_version": "1.1.0",
    "update_notice": "A newer version of EMode is available: 1.1.0 (you are running 1.0.9).",
    "error_reporting": True,
}

#: What the license manager answers a password it refuses, by email.
REFUSALS = {
    EMAIL: "Incorrect password. Please try again.",
    "nobody@example.com": "No account with that email address.",
}


def login_required() -> LicenseError:
    error = LicenseError("EMode is not signed in. Run 'emode login' or 'python -m emode login'.", None)
    error.code = "login_required"  # type: ignore[attr-defined]
    return error


class FakeAccount:
    """EMode's login calls: ``call(function, **kwargs)`` answers as EMode would, recording each call."""

    def __init__(self, signed_in: bool = False, unreachable: bool = False) -> None:
        self.state = "logged_in" if signed_in else "signed_out"
        self.error: Optional[str] = None
        self.info = copy.deepcopy(LOGIN_INFO)
        self.unreachable = unreachable
        self.device: Optional[str] = None
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, function: str, **kwargs: Any) -> Any:
        self.calls.append((function, dict(kwargs)))
        name = function[3:] if function.startswith("EM_") else function
        handler = getattr(self, "_" + name, None)
        if handler is None:
            raise login_required()
        return handler(**kwargs)

    def status_reply(self) -> Dict[str, Any]:
        reply: Dict[str, Any] = {
            "state": self.state,
            "email": EMAIL if self.state == "logged_in" else None,
            "error": self.error,
        }
        if self.state == "pending":
            reply.update(
                user_code=USER_CODE,
                verification_url="https://example.com/device/",
                verification_url_complete=f"https://example.com/device/?user_code={USER_CODE}",
                interval=5.0,
                expires_in=600.0,
            )
        return reply

    # the calls -----------------------------------------------------------------

    def _login_status(self) -> Dict[str, Any]:
        if self.state == "pending" and self.device in ("approved", "denied", "expired"):
            self.state = "logged_in" if self.device == "approved" else self.device
            self.error = None if self.device == "approved" else f"The sign-in was {self.device}."
            self.device = None
        return self.status_reply()

    def _login_start(self, method: str = "device") -> Dict[str, Any]:
        if self.unreachable:
            raise EModeError("EMode could not reach the license manager.")
        self.state = "pending"
        self.error = None
        self.device = "pending"
        return self.status_reply()

    def _login(self, email: str, password: str) -> Dict[str, Any]:
        if self.unreachable:
            raise EModeError("EMode could not reach the license manager.")
        if email == EMAIL and password == PASSWORD:
            self.state, self.error = "logged_in", None
        else:
            self.state = "signed_out"
            self.error = REFUSALS.get(email, REFUSALS["nobody@example.com"])
        return self.status_reply()

    def _logout(self) -> Dict[str, Any]:
        self.state, self.error = "signed_out", None
        return self.status_reply()

    def _login_info(self) -> Dict[str, Any]:
        if self.state != "logged_in":
            raise login_required()
        return copy.deepcopy(self.info)

    def _clear_session(self, mode: str, session_handle: Optional[str] = None) -> Dict[str, Any]:
        if mode not in ("all", "mine", "others", "latest", "oldest", "session"):
            raise ArgumentError(f"'mode' must be a clear mode, not {mode!r}", "clear_session", "mode")
        if mode == "session":
            self.info["active_checkouts"] = [
                c for c in self.info["active_checkouts"] if c.get("handle") != session_handle
            ]
        return self._login_info()

    def _settings(self, error_reporting: Optional[bool] = None) -> Dict[str, Any]:
        if error_reporting is not None:
            self.info["error_reporting"] = bool(error_reporting)
        return {"error_reporting": self.info["error_reporting"]}
