"""``PlotImage``, the figure a headless ``plot()`` returns, and how ``call()`` shows it.

The solver is never started. The wire is reproduced from its documented form
(an ndarray travels as ``{"__ndarray__": base64, "dtype", "shape"}`` and a wire
type carries its ``__data_type__`` tag), and ``call()`` talks to a stand-in
client that answers with a fixed reply.
"""

from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import emodeconnection as emc
from emodeconnection.emodeclient import obj_hook
from emodeconnection.emodeconnection import EMode
from emodeconnection.plot import PlotImage
from emodeconnection.types import EModeError, reconstruct

# The smallest valid PNG: a 1x1 transparent pixel.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class _WireEncoder(json.JSONEncoder):
    """The solver's encoding of an ndarray, as the wire contract defines it."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            data = base64.b64encode(np.ascontiguousarray(o).data).decode()
            return {"__ndarray__": data, "dtype": str(o.dtype), "shape": o.shape}
        return super().default(o)


def _image(**overrides: Any) -> PlotImage:
    fields: dict[str, Any] = {"file": None, "png": np.frombuffer(_PNG, np.uint8), "dpi": 150}
    fields.update(overrides)
    return PlotImage(**fields)


def test_plot_image_is_exported() -> None:
    assert emc.PlotImage is PlotImage


def test_plot_image_round_trips_over_the_wire() -> None:
    sent = json.dumps(_image().model_dump(), cls=_WireEncoder)
    received = reconstruct(json.loads(sent, object_hook=obj_hook))
    assert isinstance(received, PlotImage)
    assert received.file is None
    assert received.dpi == 150
    assert received._repr_png_() == _PNG


def test_a_file_only_image_round_trips() -> None:
    image = _image(file="/work/EM_figures/field.png", png=None)
    sent = json.dumps(image.model_dump(), cls=_WireEncoder)
    received = reconstruct(json.loads(sent, object_hook=obj_hook))
    assert received.file == "/work/EM_figures/field.png"
    assert received.png is None
    assert received._repr_png_() is None


def test_save_writes_the_png(tmp_path: Path) -> None:
    out = _image().save(tmp_path / "field.png")
    assert out == tmp_path / "field.png"
    assert out.read_bytes() == _PNG


def test_save_without_png_raises(tmp_path: Path) -> None:
    with pytest.raises(EModeError, match="field.png"):
        _image(file="field.png", png=None).save(tmp_path / "copy.png")


class _Reply:
    """A stand-in socket client that answers every call with one reply."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.sent: list[dict] = []

    def send(self, data: dict) -> None:
        self.sent.append(data)

    def recv(self) -> Any:
        return self.reply


def _session(reply: Any, ipython: bool) -> EMode:
    em = EMode.__new__(EMode)
    em._dsim = "emode"
    em._in_flight = False
    em._client = _Reply(reply)  # type: ignore[assignment]
    em._ipython = ipython
    em.in_ipython = lambda: ipython  # type: ignore[method-assign]
    return em


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """A stand-in ``IPython.display`` that records what it was asked to show."""
    seen: list[Any] = []
    display = types.ModuleType("IPython.display")
    display.display = seen.append  # type: ignore[attr-defined]
    ipython = types.ModuleType("IPython")
    ipython.display = display  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    return seen


def test_call_shows_a_plot_inline_in_ipython(shown: list[Any]) -> None:
    image = _image()
    assert _session(image, ipython=True).call("EM_plot") is None
    assert shown == [image]


def test_call_returns_the_plot_outside_ipython(shown: list[Any]) -> None:
    image = _image()
    assert _session(image, ipython=False).call("EM_plot") is image
    assert shown == []


def test_call_returns_a_file_only_plot_in_ipython(shown: list[Any]) -> None:
    """Nothing to show inline, so the caller gets the saved file's path."""
    image = _image(file="EM_figures/field.png", png=None)
    assert _session(image, ipython=True).call("EM_plot") is image
    assert shown == []


def test_call_returns_other_replies_unchanged_in_ipython(shown: list[Any]) -> None:
    assert _session("ok", ipython=True).call("EM_plot") == "ok"
    assert _session({"n_eff": 2.4}, ipython=True).call("EM_get") == {"n_eff": 2.4}
    assert shown == []
