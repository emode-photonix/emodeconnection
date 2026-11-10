"""``PlotView`` and ``Schematic`` cross the wire as EMode sends them and come back whole.

EMode sends a wire type as ``wire_dump`` (its tag and the fields that differ
from their defaults), arrays as ``__ndarray__``. The client decodes with
``obj_hook`` and rebuilds with ``reconstruct``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from emode import compat
from emode.emodeclient import obj_hook
from emode.plotting import PlotView, Raster, Schematic
from emode.types import _TYPE_REGISTRY, reconstruct, serialize, wire_dump

from plot_fixtures import index_view, layout_view, schematic, xy_view, zx_view


def _over_the_wire(model):
    payload = json.dumps(serialize(wire_dump(model), ndarray=True))
    return reconstruct(json.loads(payload, object_hook=obj_hook))


def _arrays(value, path="view"):
    if isinstance(value, np.ndarray):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _arrays(item, f"{path}[{i}]")
    elif hasattr(type(value), "model_fields") and not isinstance(value, type):
        for name in type(value).model_fields:
            yield from _arrays(getattr(value, name), f"{path}.{name}")


@pytest.mark.parametrize("make", [xy_view, zx_view, layout_view, index_view, schematic])
def test_a_view_round_trips(make):
    view = make()
    back = _over_the_wire(view)
    assert type(back) is type(view)
    for (path, a), (_, b) in zip(_arrays(view), _arrays(back)):
        np.testing.assert_array_equal(a, b, err_msg=path)
    assert back.model_dump(exclude={"rasters", "fills", "outlines", "ports", "facets", "monitors", "window", "mesh", "connections"}) == view.model_dump(
        exclude={"rasters", "fills", "outlines", "ports", "facets", "monitors", "window", "mesh", "connections"}
    )


@pytest.mark.parametrize("make", [xy_view, zx_view, layout_view])
def test_every_coordinate_and_value_array_is_float32(make):
    for path, array in _arrays(_over_the_wire(make())):
        if array.dtype != bool:
            assert array.dtype == np.float32, path


def test_a_raster_given_float64_holds_float32():
    r = Raster(x=np.arange(3.0), y=np.arange(2.0), z=np.zeros((2, 3)))
    assert {r.x.dtype, r.y.dtype, r.z.dtype} == {np.dtype(np.float32)}


def test_every_new_type_is_registered_and_versioned():
    names = ["PlotView", "Schematic", "Raster", "Polyline", "Region", "Segments", "Label", "LegendEntry",
             "AxesInfo", "ScaleInfo", "Slider", "MenuState", "Arrow", "Block", "Stub"]
    for name in names:
        assert name in _TYPE_REGISTRY
        assert compat.TYPE_SINCE[name] == "1.0.9"


def test_importing_emode_imports_no_matplotlib_and_no_web_server():
    import subprocess
    import sys

    code = "import sys, emode; print('matplotlib' in sys.modules, 'http.server' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False"]
