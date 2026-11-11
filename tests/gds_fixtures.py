"""A small ``GDSView`` as EMode returns it, for the GDS layer view's tests.

Three layers of a device in nm on the chip: two rectangles on 1/0 (the core),
one slab on 2/0 and three small markers on 2/5, and the simulation's core
shape for the overlay.
"""
from __future__ import annotations

import numpy as np

from emode.gds import GDSLayer, GDSView
from emode.plotting import AxesInfo, MenuState, Region


def _rect(z0: float, x0: float, z1: float, x1: float) -> np.ndarray:
    return np.array([[z0, x0], [z1, x0], [z1, x1], [z0, x1]], float)


def layer(number: int, datatype: int, polygons: list, name: str = "") -> GDSLayer:
    xy = np.concatenate(polygons)
    offsets = np.cumsum([0] + [len(p) for p in polygons])
    return GDSLayer(layer=number, datatype=datatype, name=name, xy=xy, offsets=offsets)


def gds_view() -> GDSView:
    return GDSView(
        layers=[
            layer(1, 0, [_rect(0, -250, 4000, 250), _rect(4000, -400, 9000, 400)], name="core"),
            layer(2, 0, [_rect(0, -2000, 9000, 2000)]),
            layer(2, 5, [_rect(100 + 1000 * k, 1500, 300 + 1000 * k, 1700) for k in range(3)]),
        ],
        shapes=[Region(kind="shape", name="core", x=[0, 9000, 9000, 0], y=[-250, -250, 250, 250])],
        axes=AxesInfo(
            xlabel="z", ylabel="x", x_unit="um", y_unit="um",
            xlim=(0.0, 9000.0), ylim=(-2000.0, 2000.0), invert_y=True, equal_aspect=True,
        ),
        pixel_nm=7.5,
        cells={"g": "device"},
        menu=MenuState(selected={"view": "gds"}),
    )


LYP = """<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <frame-color>#ff0000</frame-color>
  <fill-color>#ff8000</fill-color>
  <dither-pattern>I8</dither-pattern>
  <visible>true</visible>
  <transparent>false</transparent>
  <name>WG</name>
  <source>1/0@1</source>
 </properties>
 <properties>
  <frame-color>#0000ff</frame-color>
  <fill-color>#0000ff</fill-color>
  <dither-pattern>I1</dither-pattern>
  <visible>false</visible>
  <name>SLAB</name>
  <source>2/0@1</source>
 </properties>
 <group-members>
  <properties>
   <fill-color>#00ff00</fill-color>
   <dither-pattern>I12</dither-pattern>
   <name>MARK</name>
   <source>2/5@1</source>
  </properties>
 </group-members>
 <properties>
  <fill-color>#123456</fill-color>
  <source>*/*@*</source>
 </properties>
</layer-properties>
"""
