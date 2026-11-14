"""The account page: signing in to EMode, and the account's licenses and sessions, in the tab.

EMode answers a session's first call with a ``LicenseError`` of code
``'login_required'`` when no login is stored, or the stored one is refused, and
keeps the connection open for the account page's calls: ``login_status``,
``login_start``, ``login``, ``logout``, ``login_info``, ``clear_session`` and
``settings``. EMode holds every secret and talks to the license manager. This
module is the client's half: the routes the page posts to, which forward each
post to EMode as one call, the account panel's rows (``account_view``), and
the wait in ``EMode()`` while the user signs in.

A password reaches this process only in the body of a POST from the page, is
passed on to EMode in one call, and is never written, logged or answered.
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import EModeError, LicenseError

if TYPE_CHECKING:
    from ..emode import EMode

#: The website's account page: license sharing, app tokens, the password.
MY_ACCOUNT_URL = "https://emodephotonix.com/my-account/"

#: Where a forgotten password is reset.
FORGOT_PASSWORD_URL = "https://emodephotonix.com/my-account/lost-password/"

#: The clear modes the page offers, as ``EMode(clear=...)`` takes them.
CLEAR_MODES = ("all", "mine", "others", "latest", "oldest")

#: The error-reporting setting's explanation, as the login window gave it.
SETTINGS_EXPLANATION = (
    "When a session closes, EMode can send EMode Photonix a short report of each internal "
    "error: the error type, the EMode function, the EMode version and the platform. It never "
    "includes your script, geometry or files. The setting applies to every session on this "
    "computer, and 'emode settings --error-reporting on|off' changes it too."
)

#: Seconds between the status checks of ``EMode()`` while it waits for a sign-in.
LOGIN_POLL = 1.0

#: Seconds ``EMode()`` waits for the page to load before it counts the tab as gone.
CONNECT_WAIT = 120.0

#: The license names the window showed, by the license manager's type.
_LICENSE_NAMES = {
    "emode2d": "EMode2D",
    "emode3d": "EMode3D",
    "extra2d": "Extra2D",
    "extra3d": "Extra3D",
    "trial3d": "Trial3D",
}

#: The calls the account page makes, by route. A GET reads, a POST changes.
_GETS = {"status": "EM_login_status", "info": "EM_login_info"}
_POSTS = {
    "password": "EM_login",
    "start": "EM_login_start",
    "logout": "EM_logout",
    "clear": "EM_clear_session",
    "settings": "EM_settings",
}


def device_signin_offered() -> bool:
    """Whether the page offers "Sign in with your browser" (the device code of a two-factor account).

    Off until the license manager's device endpoints are live: ``EMODE_DEVICE_SIGNIN=1``
    turns it on. A status from EMode that says ``device`` decides instead.
    """
    return os.environ.get("EMODE_DEVICE_SIGNIN") == "1"


def is_login_required(error: BaseException) -> bool:
    """Whether `error` is EMode's answer to a call made before signing in."""
    return isinstance(error, LicenseError) and getattr(error, "code", None) == "login_required"


# ------------------------------------------------------------------ the account panel


def _license_name(kind: str) -> str:
    return _LICENSE_NAMES.get(str(kind).split("-")[0].lower(), "Unknown")


def _date(text: Optional[str]) -> str:
    try:
        return datetime.strptime(str(text), "%Y-%m-%d").strftime("%b %d, %Y")
    except ValueError:
        return str(text) if text else "N/A"


def _started(iso: Optional[str]) -> str:
    """A session's start, sent in UTC, in this computer's time zone."""
    try:
        return datetime.fromisoformat(str(iso)).replace(tzinfo=timezone.utc).astimezone().strftime("%b %d, %H:%M")
    except ValueError:
        return "unknown"


def _clearable(row: Dict[str, Any]) -> bool:
    """The license manager sends a handle only on a row this user may clear, and a roaming row never clears."""
    return bool(row.get("handle")) and not row.get("roaming", False)


def account_view(info: Dict[str, Any]) -> Dict[str, Any]:
    """The account panel's rows from EMode's ``login_info()``, as the login window showed them.

    The counts are checked-out against total seats, a shared pool's seats
    counted once per owner and pool. A session row says whether it can be
    cleared, and the question to ask first.
    """
    licenses = info.get("licenses") or []
    shared_to_me = info.get("shared_to_me") or []
    shared_by_me = info.get("shared_by_me") or []
    checkouts = info.get("active_checkouts") or []
    email = info.get("email")

    totals = {
        "2d": sum(1 for lic in licenses if str(lic.get("type", "")).endswith("2d")),
        "3d": sum(1 for lic in licenses if str(lic.get("type", "")).endswith("3d")),
    }
    pools: Dict[Any, int] = {}
    for share in shared_to_me:
        kind = str(share.get("license_type", ""))
        pool = share.get("pool") or ("3d" if kind.endswith("3d") else "2d")
        key = (share.get("owner_email"), pool)
        pools[key] = pools.get(key, 0) + int(share.get("total_seats") or 0)
    for (_, pool), seats in pools.items():
        totals["3d" if pool == "3d" else "2d"] += seats
    used = {kind: sum(1 for c in checkouts if c.get("lic_type") == kind) for kind in ("2d", "3d")}

    rows: List[Dict[str, Any]] = [
        {
            "subscription": _license_name(lic.get("type", "")),
            "status": "Active",
            "expiration": _date(lic.get("expires")),
            "owner": email,
            "shared": False,
        }
        for lic in licenses
    ]
    rows += [
        {
            "subscription": _license_name(share.get("license_type", "")),
            "status": "Shared",
            "expiration": _date(share.get("expiration")),
            "owner": share.get("owner_email"),
            "shared": True,
        }
        for share in shared_to_me
    ]

    sessions = []
    for c in checkouts:
        kind = str(c.get("lic_type", "")).upper()
        started = _started(c.get("start_time"))
        computer = c.get("comp_name") or ""
        sessions.append(
            {
                "type": kind,
                "user": c.get("user"),
                "computer": computer,
                "started": started,
                "owner": c.get("owner"),
                "shared": c.get("user") != c.get("owner"),
                "clearable": _clearable(c),
                "handle": c.get("handle") if _clearable(c) else None,
                "confirm": (
                    f"Clear the {kind} session of {c.get('user')} on "
                    f"{computer or 'an unknown computer'}, started {started}?"
                    "\n\nA simulation still running in that session loses its license."
                ),
            }
        )

    accounts = len({s.get("member_email") for s in shared_by_me})
    line = f"You are sharing with {accounts} account{'' if accounts == 1 else 's'}." if accounts else None
    return {
        "email": email,
        "version": info.get("version"),
        "latest_version": info.get("latest_version"),
        "update_notice": info.get("update_notice"),
        "error_reporting": bool(info.get("error_reporting")),
        "counts": {
            "2D": f"{used['2d']} / {totals['2d']}",
            "3D": f"{used['3d']} / {totals['3d']}",
        },
        "sharing": {"link": MY_ACCOUNT_URL if (shared_to_me or shared_by_me) else None, "line": line},
        "licenses": rows,
        "sessions": sessions,
    }


# ------------------------------------------------------------------ the routes


def route(login: Callable[..., Any], method: str, action: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Answer the account page's `action`, or None when there is no such route.

    `login(function, **kwargs)` makes one EMode call. An error EMode answers
    is the page's error row, never an HTTP error, and a signed-out account is
    ``{'state': 'signed_out'}``. Nothing here logs, and the password is
    passed on and dropped.
    """
    function = (_GETS if method == "GET" else _POSTS).get(action)
    if function is None:
        return None
    kwargs: Dict[str, Any] = {}
    if action == "password":
        email = str(body.get("email") or "").strip()
        password = body.get("password")
        if not email or not isinstance(password, str) or not password:
            return {"state": "signed_out", "error": "Enter your email and password."}
        kwargs = {"email": email, "password": password}
    elif action == "start":
        kwargs = {"method": "device"}
    elif action == "clear":
        kwargs = {"mode": str(body.get("mode") or "")}
        if body.get("session_handle"):
            kwargs["session_handle"] = str(body["session_handle"])
    elif action == "settings":
        kwargs = {"error_reporting": bool(body.get("error_reporting"))}
    requested = kwargs.get("error_reporting")
    try:
        reply = login(function, **kwargs)
    except LicenseError as exc:
        if is_login_required(exc):
            return {"state": "signed_out"}
        return _error(action, exc)
    except EModeError as exc:
        return _error(action, exc)
    if action in ("info", "clear"):
        return account_view(reply) if isinstance(reply, dict) else {"state": "signed_out"}
    if action == "settings":
        value = reply.get("error_reporting", requested) if isinstance(reply, dict) else requested
        return {"error_reporting": bool(value)}
    return reply if isinstance(reply, dict) else {"state": "signed_out"}


def _error(action: str, exc: EModeError) -> Dict[str, Any]:
    message = getattr(exc, "msg", None) or str(exc)
    if action in ("password", "start", "status", "logout"):
        return {"state": "signed_out", "error": message}
    return {"error": message}


# ------------------------------------------------------------------ EMode() waits for the sign-in


def _no_browser() -> bool:
    if os.environ.get("EMODE_HEADLESS") == "1":
        return True
    if sys.platform.startswith("linux"):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


def sign_in(em: "EMode", error: LicenseError) -> None:
    """Open the account page and wait until EMode says the session is signed in.

    Raises `error` (EMode's message names ``emode login`` and ``python -m
    emode login``) where no browser can open, and when the tab is closed, or
    never loads, before a sign-in.
    """
    from .session import ensure_server

    if _no_browser():
        raise error
    server = ensure_server(em)
    url = server.url(page="account")
    print(f"EMode is not signed in. Sign in on the account page: {url}", flush=True)
    if not webbrowser.open(url):
        server.stop()
        raise error
    server.opened_at = time.monotonic()
    server.last_ping = None
    while True:
        status = em.call("EM_login_status")
        if isinstance(status, dict) and status.get("state") == "logged_in":
            return
        now = time.monotonic()
        never_loaded = server.last_ping is None and now - server.opened_at > CONNECT_WAIT
        if server.tab_closed or never_loaded or (server.last_ping is not None and not server.tab_alive()):
            raise error
        time.sleep(LOGIN_POLL)
