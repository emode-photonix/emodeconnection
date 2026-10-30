"""The demo license type, and a session the solver refuses at launch.

A demo session is the time-limited session of the "Try EMode now" notebook.
It reports itself as ``LicenseType.demo``. When the solver cannot start a
session at all (no license available, a busy demo, a demo that has ended), it
answers the first call with a ``LicenseError`` that says why, and ``EMode()``
raises it and shuts the solver process down.
"""

from __future__ import annotations

import atexit
import json
import signal
from typing import Any

import pytest

import emodeconnection.emodeconnection as launch_module
from emodeconnection.emodeconnection import EMode
from emodeconnection.types import LicenseError, LicenseType, reconstruct


def test_license_type_demo() -> None:
    demo = LicenseType("demo")
    assert demo is LicenseType.demo
    assert str(demo) == "demo"
    assert demo.to_dict() == {"__type__": "LicenseType", "value": "demo"}
    assert (
        LicenseType.coerce(json.loads(json.dumps(demo.to_dict()))) is LicenseType.demo
    )


def _over_the_wire(error: LicenseError) -> Any:
    payload = json.dumps(error.to_dict(), default=lambda o: o.to_dict())
    return reconstruct(json.loads(payload))


def test_a_demo_license_error_names_the_demo() -> None:
    error = _over_the_wire(
        LicenseError("Your 60 minute EMode demo session has ended.", LicenseType.demo)
    )
    assert isinstance(error, LicenseError)
    assert error.license_type is LicenseType.demo
    assert str(error) == (
        'LicenseError: current license: "EModeDEMO", '
        'error msg: "Your 60 minute EMode demo session has ended."'
    )


@pytest.mark.parametrize(
    "license_type, expected",
    [(LicenseType._2D, "EMode2D"), (LicenseType._3D, "EMode3D"), (None, "EMode2D")],
)
def test_other_license_errors_read_as_before(license_type: Any, expected: str) -> None:
    assert f'"{expected}"' in str(_over_the_wire(LicenseError("msg", license_type)))


class _Process:
    def __init__(self) -> None:
        self.waited = False

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


class _Client:
    instances: list["_Client"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False
        _Client.instances.append(self)

    @property
    def connected(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


def test_a_refused_launch_raises_the_license_error_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()
    _Client.instances.clear()
    hooks: list[Any] = []
    monkeypatch.setattr(launch_module, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(launch_module, "EModeClient", _Client)
    monkeypatch.setattr(signal, "signal", lambda signum, handler: None)
    monkeypatch.setattr(
        atexit, "register", lambda func, *args, **kwargs: hooks.append(func) or func
    )
    monkeypatch.setattr(EMode, "in_ipython", lambda self: False)

    refusal = LicenseError(
        "The EMode demo is busy, please try again in a few minutes", None
    )

    calls: list[str] = []

    def refuse(self: EMode, function: str, **kwargs: Any) -> Any:
        calls.append(function)
        raise refusal

    monkeypatch.setattr(EMode, "call", refuse)

    with pytest.raises(LicenseError) as caught:
        EMode()

    assert caught.value is refusal
    (client,) = _Client.instances
    assert client.closed
    assert process.waited
    # The exit hook has nothing left to close.
    for hook in hooks:
        hook()
    assert calls == ["EM_init"]


def test_a_launch_refusal_with_no_license_type_shows_only_the_message() -> None:
    error = _over_the_wire(LicenseError("No EMode license is available.", None))
    assert error.license_type is None
    assert str(error) == "LicenseError: No EMode license is available."
    assert "EMode2D" not in str(error)


def test_a_license_error_with_a_license_type_still_names_it() -> None:
    error = LicenseError("Over the seat limit.", LicenseType._3D)
    assert str(error) == (
        'LicenseError: current license: "EMode3D", error msg: "Over the seat limit."'
    )
