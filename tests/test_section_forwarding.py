"""The transformation-optics sections reach EMode by name (emode#1390).

`coupler_section`, `transformed_section` and `bend_section(path=...)` need no
client code: `EMode.__getattr__` sends any method as `EM_<name>` with its
keywords. These tests pin that for the three, curves included.
"""

from __future__ import annotations

from typing import Any

import pytest

from emode.emode import EMode
from emode.geometry import Arc, Line, Pose


def _session() -> tuple[EMode, list[tuple[str, dict[str, Any]]]]:
    em = EMode.__new__(EMode)
    calls: list[tuple[str, dict[str, Any]]] = []
    em.call = lambda function, **kwargs: calls.append((function, kwargs))  # type: ignore[method-assign]
    return em, calls


@pytest.mark.parametrize(
    "method, kwargs",
    [
        (
            "coupler_section",
            dict(
                name="c",
                profile="p",
                waveguides=[Line(length=6000.0), Arc(radius=20000.0, sweep_angle=0.3)],
                poses=[Pose(x=-300.0), Pose(x=600.0)],
                width=600.0,
                virtual_gap=300.0,
                length=6000.0,
            ),
        ),
        ("transformed_section", dict(name="t", transform="bend", profile="p", length=3000.0)),
        (
            "bend_section",
            dict(name="b", profile="p", path=Arc(radius=20000.0, sweep_angle=0.2), method="TO"),
        ),
    ],
)
def test_a_section_reaches_its_em_function(method: str, kwargs: dict[str, Any]) -> None:
    em, calls = _session()
    getattr(em, method)(**kwargs)
    assert calls == [(f"EM_{method}", kwargs)]
