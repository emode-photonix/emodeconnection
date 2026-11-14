"""The solver's warnings for a call reach the client as a list (#95)."""

from __future__ import annotations

from typing import Any

import pytest

from emode.emode import EMode


def _em(version: str | None, answer: Any = None) -> tuple[EMode, list[str]]:
    em = EMode.__new__(EMode)
    em._emode_version_cache = version
    asked: list[str] = []

    def call(function: str, **kwargs: Any) -> Any:
        asked.append(function)
        return answer

    em.call = call  # type: ignore[method-assign]
    return em, asked


def test_warnings_asks_emode_for_what_it_logged() -> None:
    em, asked = _em("1.0.9", ["the window is small"])
    assert em.warnings() == ["the window is small"]
    assert asked == ["EM_warnings"]


@pytest.mark.parametrize("version", [None, "1.0.7", "1.0.8"])
def test_an_older_emode_keeps_no_warnings(version: str | None) -> None:
    em, asked = _em(version, ["never read"])
    assert em.warnings() == []
    assert asked == []
