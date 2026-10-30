"""The MCP server skeleton, driven in-process with a fake EMode client.

No license and no solver: the server is built with a stand-in for ``EMode``
that records what the tools ask of it, and an in-memory MCP client calls the
tools through the real protocol.
"""

from __future__ import annotations

import asyncio
import io
import json
from contextlib import redirect_stderr
from typing import Any, ClassVar

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.client import Client

from emodeconnection.mcp import main
from emodeconnection.mcp.server import create_server
from emodeconnection.types import EModeError


class FakeEMode:
    """Records calls, answers ``get`` with a two-mode solve."""

    instances: ClassVar[list[FakeEMode]] = []
    fail_on: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed_with: dict[str, Any] | None = None
        FakeEMode.instances.append(self)

    def _record(self, name: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((name, kwargs))
        if FakeEMode.fail_on == name:
            raise EModeError(f"{name} failed on purpose")

    def settings(self, **kwargs: Any) -> None:
        self._record("settings", kwargs)

    def shape(self, **kwargs: Any) -> None:
        self._record("shape", kwargs)

    def FDM(self, **kwargs: Any) -> dict:
        self._record("FDM", kwargs)
        return {}

    def get(self, key: str, **kwargs: Any) -> Any:
        self._record("get", {"key": key, **kwargs})
        if "get" in FakeEMode.returns:
            return FakeEMode.returns["get"](key=key)
        if key == "effective_index":
            return np.array([2.5 + 0j, 1.9 + 1e-4j])
        if key == "loss_dB_per_m":
            return np.array([0.0, 12.5])
        raise EModeError(f"no such result {key}")

    def close(self, **kwargs: Any) -> None:
        self.closed_with = kwargs

    # Every other client method records its call and answers from ``returns``.
    returns: ClassVar[dict[str, Any]] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args: Any, **kwargs: Any) -> Any:
            if args:
                kwargs["key"] = args[0]
            self._record(name, kwargs)
            value = FakeEMode.returns.get(name)
            return value(**kwargs) if callable(value) else value

        return method


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    FakeEMode.instances = []
    FakeEMode.fail_on = None
    FakeEMode.returns = {}


def call(
    tool: str, args: dict[str, Any] | None = None, server: Any = None, **server_kw: Any
) -> Any:
    server = server or create_server(client_factory=FakeEMode, **server_kw)

    async def run() -> Any:
        async with Client(server) as client:
            return await client.call_tool(tool, args or {})

    return asyncio.run(run())


def session(*steps: tuple[str, dict[str, Any]], **server_kw: Any) -> list[Any]:
    """Call the tools in order against one server and return each result."""
    server = create_server(client_factory=FakeEMode, **server_kw)

    async def run() -> list[Any]:
        async with Client(server) as client:
            return [await client.call_tool(tool, args) for tool, args in steps]

    return asyncio.run(run())


WG = {
    "settings": {"wavelength": 1550, "num_modes": 2},
    "shapes": [
        {"name": "BOX", "material": "SiO2", "height": 800},
        {"name": "core", "material": "Si", "width": 600, "height": 400},
    ],
}


TOOLS = [
    "emode_close",
    "emode_converge",
    "emode_eme",
    "emode_evaluate",
    "emode_get",
    "emode_material",
    "emode_open",
    "emode_optimize",
    "emode_plot",
    "emode_section",
    "emode_settings",
    "emode_shape",
    "emode_solve",
    "emode_sweep",
]


def test_the_server_lists_the_plan_tools() -> None:
    server = create_server(client_factory=FakeEMode)

    async def run() -> Any:
        async with Client(server) as client:
            return await client.list_tools()

    tools = asyncio.run(run()).tools
    assert sorted(t.name for t in tools) == TOOLS
    for tool in tools:
        assert tool.description
        assert tool.input_schema["type"] == "object"
        if tool.name != "emode_plot":  # an image has no structured output
            assert tool.output_schema is not None


def test_open_builds_the_cross_section_through_the_public_client() -> None:
    (result,) = session(("emode_open", WG))
    assert not result.is_error
    (em,) = FakeEMode.instances
    # stdout carries the protocol, so the solver must be quiet
    assert em.init_kwargs["quiet"] is True
    assert em.init_kwargs["save"] is False
    names = [c[0] for c in em.calls]
    assert names == ["settings", "shape", "shape"]
    assert em.calls[0][1] == {"wavelength": 1550, "num_modes": 2}
    assert em.calls[1][1] == {"name": "BOX", "material": "SiO2", "height": 800}
    assert result.structured_content["shapes"] == ["BOX", "core"]


def test_open_without_settings_or_shapes_only_starts_a_session() -> None:
    (result,) = session(("emode_open", {}))
    assert not result.is_error
    assert [c[0] for c in FakeEMode.instances[0].calls] == []


def test_open_passes_the_session_options() -> None:
    session(
        (
            "emode_open",
            {"simulation_name": "wg", "license_type": "2d", "save_path": "/tmp/x"},
        )
    )
    kw = FakeEMode.instances[0].init_kwargs
    assert kw["simulation_name"] == "wg"
    assert kw["license_type"] == "2d"
    assert kw["save_path"] == "/tmp/x"
    assert kw["save"] is True


def test_open_rejects_an_unknown_setting() -> None:
    (result,) = session(("emode_open", {"settings": {"wavelenth": 1550}}))
    assert result.is_error
    assert FakeEMode.instances == []


def test_a_second_open_is_refused_while_a_session_is_running() -> None:
    first, second = session(("emode_open", {}), ("emode_open", {}))
    assert not first.is_error
    assert second.is_error
    assert "emode_close" in second.content[0].text
    assert len(FakeEMode.instances) == 1


def test_a_failed_open_closes_the_half_built_session_and_allows_a_retry() -> None:
    FakeEMode.fail_on = "shape"
    bad, _ = session(("emode_open", WG), ("emode_close", {}))
    assert bad.is_error
    assert "shape failed on purpose" in bad.content[0].text
    assert FakeEMode.instances[0].closed_with == {"save": False}
    FakeEMode.fail_on = None
    again, _ = session(("emode_open", WG), ("emode_close", {}))
    assert not again.is_error


def test_solve_returns_n_eff_and_loss_per_mode() -> None:
    _, result = session(("emode_open", WG), ("emode_solve", {}))
    assert not result.is_error
    modes = result.structured_content["modes"]
    assert [m["mode"] for m in modes] == [0, 1]
    assert modes[0]["n_eff"] == pytest.approx(2.5)
    assert modes[0]["n_eff_imag"] == pytest.approx(0.0)
    assert modes[1]["n_eff"] == pytest.approx(1.9)
    assert modes[1]["n_eff_imag"] == pytest.approx(1e-4)
    assert modes[1]["loss_dB_per_m"] == pytest.approx(12.5)
    json.dumps(result.structured_content)  # plain JSON, no numpy left in it


def test_solve_applies_overrides_before_solving() -> None:
    session(("emode_open", WG), ("emode_solve", {"wavelength": 1310, "num_modes": 4}))
    calls = FakeEMode.instances[0].calls
    names = [c[0] for c in calls]
    assert names[-4:] == ["settings", "FDM", "get", "get"]
    assert calls[-4][1] == {"wavelength": 1310, "num_modes": 4}


def test_solve_without_a_session_says_to_open_one() -> None:
    result = call("emode_solve")
    assert result.is_error
    assert "emode_open" in result.content[0].text


def test_a_solver_error_reaches_the_agent_as_a_tool_error() -> None:
    FakeEMode.fail_on = "FDM"
    _, result = session(("emode_open", WG), ("emode_solve", {}))
    assert result.is_error
    assert "FDM failed on purpose" in result.content[0].text


def test_close_ends_the_session_and_a_new_one_can_open() -> None:
    _, closed, reopened = session(
        ("emode_open", {}), ("emode_close", {}), ("emode_open", {})
    )
    assert closed.structured_content == {"closed": True}
    assert FakeEMode.instances[0].closed_with == {"save": False}
    assert not reopened.is_error
    assert len(FakeEMode.instances) == 2


def test_close_can_save_and_is_harmless_without_a_session() -> None:
    session(("emode_open", {}), ("emode_close", {"save": True}))
    assert FakeEMode.instances[0].closed_with == {"save": True}
    result = call("emode_close")
    assert not result.is_error
    assert result.structured_content == {"closed": False}


def test_shutting_the_server_down_closes_an_open_session() -> None:
    session(("emode_open", {}))
    assert FakeEMode.instances[0].closed_with == {"save": False}


def test_main_without_the_mcp_extra_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    for mod in [
        m
        for m in list(__import__("sys").modules)
        if m.startswith("emodeconnection.mcp.")
    ]:
        monkeypatch.delitem(__import__("sys").modules, mod)
    err = io.StringIO()
    with redirect_stderr(err), pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0
    assert "emodeconnection[mcp]" in err.getvalue()
