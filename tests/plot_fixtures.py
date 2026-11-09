"""Small ``PlotView`` and ``Schematic`` values, as EMode returns them, for the plot tests.

Each is built the way EMode builds the panel (coordinates in nm, float32
arrays, every overlay present), with values simple enough to check by hand.
"""
from __future__ import annotations

import numpy as np

from emode.plotting import (
    Arrow,
    AxesInfo,
    Block,
    Label,
    LegendEntry,
    MenuState,
    PlotView,
    Polyline,
    Raster,
    Region,
    ScaleInfo,
    Schematic,
    Segments,
    Slider,
    Stub,
)

SETTINGS = {
    "Index outline": True,
    "Contours": True,
    "Text display": True,
    "Mesh display": False,
    "Window display": False,
    "Forward only": False,
    "Physical space": True,
    "Blend slices": True,
}

LEVELS = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99])


def xy_view() -> PlotView:
    """An x-y panel of Ex, real part, on a nonuniform grid, with the core's outline."""
    x = np.array([-900.0, -500.0, -200.0, 0.0, 200.0, 500.0, 900.0])
    y = np.array([-400.0, 0.0, 110.0, 220.0, 600.0])
    X, Y = np.meshgrid(x, y)
    z = np.exp(-((X / 300) ** 2) - ((Y - 110) / 150) ** 2) * np.sign(X + 1)
    return PlotView(
        plane="x-y",
        state="profile",
        rasters=[Raster(x=x, y=y, z=z, kind="diverging", contour_levels=LEVELS, interpolation="bilinear")],
        outlines=[Polyline(kind="outline", x=[-250, 250, 250, -250, -250], y=[0, 0, 220, 220, 0], label="core")],
        window=[
            Region(kind="window", x=[-1000, 1000, 1000, -1000], y=[-500, -500, 700, 700]),
            Region(kind="pml", x=[-1100, 1100, 1100, -1100], y=[-600, -600, 800, 800], hatch="xx"),
        ],
        mesh=[Segments(kind="mesh", xy=np.array([[[-900, -400], [-900, 600]], [[-500, -400], [-500, 600]]]))],
        texts=[
            Label(text="Re(Ex), 0", x=0.04, y=1.02, kind="title"),
            Label(text="$n_\\mathrm{eff} = $2.400", x=0.96, y=1.02, ha="right", kind="neff"),
        ],
        legend=[LegendEntry(label="core")],
        axes=AxesInfo(xlabel="x", ylabel="y", xlim=(-1000.0, 1000.0), ylim=(-500.0, 700.0), box_aspect=2 ** -0.5),
        scale=ScaleInfo(kind="signed", vmin=-1.0, vmax=1.0),
        menu=MenuState(
            data=["Ex", "Ey", "Index", "Shapes"],
            components=["real", "imag", "abs", "abs^2", "log"],
            normalizations=["vector", "component", "section"],
            modes=["0, TE", "1, TM"],
            profiles=["FDM"],
            settings=SETTINGS,
            active_settings=list(SETTINGS),
            selected={"data": "Ex", "component": "real", "normalization": "vector", "mode": "0, TE", "profile": "FDM"},
        ),
        wavelength=1550.0,
    )


def zx_view() -> PlotView:
    """A z-x panel of |Ex| in two slices of different transverse grids, a port and a facet."""
    z1, z2 = np.linspace(0, 1000, 5), np.linspace(1000, 3000, 9)
    x1, x2 = np.linspace(-800, 800, 4), np.linspace(-1200, 1200, 6)
    r1 = Raster(x=z1, y=x1, z=np.full((4, 5), 0.5))
    r2 = Raster(x=z2, y=x2, z=np.full((6, 9), 0.25))
    return PlotView(
        plane="z-x",
        view="simulation",
        rasters=[r1, r2],
        background=0.0,
        outlines=[Polyline(kind="outline", x=[0, 3000], y=[-250, -250]), Polyline(kind="outline", x=[0, 3000], y=[250, 250])],
        ports=[Polyline(kind="port", x=[0, 0], y=[-800, 800], label="left", label_at=(0.0, 0.0))],
        facets=[Polyline(kind="facet", x=[2900, 3000], y=[-500, 500], label="facet", label_at=(3000.0, 500.0))],
        monitors=[],
        mesh=[Segments(kind="boundary", xy=np.array([[[1000, -1200], [1000, 1200]]]), alpha=0.5)],
        texts=[Label(text="|Ex|, left:0", x=0.04, y=1.02)],
        axes=AxesInfo(
            xlabel="z", ylabel="x", x_unit="um", y_unit="nm", xlim=(0.0, 3000.0), ylim=(-1200.0, 1200.0),
            invert_y=True, box_aspect=2 ** -0.5,
        ),
        scale=ScaleInfo(kind="positive", vmin=0.0, vmax=1.0, reference=3.2e-3),
        menu=MenuState(
            data=["Ex", "Ey", "Outline", "Index", "Shapes"],
            components=["real", "imag", "abs", "abs^2", "log"],
            normalizations=["vector", "component", "section"],
            modes=["0, TE", "1, TM"],
            planes=["x-y", "z-x", "z-y", "schematic"],
            ports=["left", "right"],
            sections=["all", "s1", "s2"],
            z_resolutions=["Auto", "25", "50", "100", "250", "500"],
            slider=Slider(label="x:", min=-1200, max=1200, unit="nm", value="sum", options=["sum", "mean", "abs sum"]),
            settings={**SETTINGS, "Contours": False},
            active_settings=[s for s in SETTINGS if s != "Contours"],
            selected={
                "data": "Ex", "component": "abs", "normalization": "vector", "mode": "0, TE", "plane": "z-x",
                "port": "left", "section": "all", "z_resolution": "Auto",
            },
        ),
        sections=["s1", "s2"],
        arrows=[Arrow(direction="right", target="s2", x=0.98, y=0.5)],
        wavelength=1550.0,
        z_resolution_nm=250.0,
    )


def layout_view() -> PlotView:
    """A layout panel on the chip, both axes in um, equal aspect, masked off the chip."""
    x = np.linspace(0, 10000, 11)
    y = np.linspace(-5000, 5000, 11)
    mask = np.zeros((11, 11), bool)
    mask[:3] = True
    return PlotView(
        plane="z-x",
        view="layout",
        rasters=[Raster(x=x, y=y, z=np.linspace(0, 1, 121).reshape(11, 11), mask=mask)],
        axes=AxesInfo(xlabel="z", ylabel="x", x_unit="um", y_unit="um", xlim=(0.0, 10000.0), ylim=(-5000.0, 5000.0),
                      invert_y=True, equal_aspect=True),
        scale=ScaleInfo(kind="positive"),
    )


def index_view() -> PlotView:
    """An Index panel: a background and a core filled with their indices' colors, no raster."""
    return PlotView(
        plane="x-y",
        fills=[
            Region(kind="background", x=[-1000, 1000, 1000, -1000], y=[-500, -500, 700, 700], value=0.0),
            Region(kind="fill", x=[-250, 250, 250, -250], y=[0, 0, 220, 220], value=1.0, name="core"),
        ],
        legend=[LegendEntry(label="core", value=1.0)],
        axes=AxesInfo(xlabel="x", ylabel="y", xlim=(-1000.0, 1000.0), ylim=(-500.0, 700.0)),
        scale=ScaleInfo(kind="index", ticks=[0.0, 1.0], tick_labels=["1.44", "3.48"], tick_format="%3.2f"),
    )


def schematic() -> Schematic:
    """Two blocks connected left to right, the left one's input launched."""
    return Schematic(
        blocks=[
            Block(name="a", label="a\nstraight, 1 um\nwg", x=0.0, y=0.0, width=3.4, height=1.7),
            Block(name="b", label="b\nstraight, 1 um\nwg", x=6.4, y=0.0, width=3.4, height=1.7),
        ],
        stubs=[
            Stub(section="a", side="left", x0=-1.7, y0=0.0, x1=-2.25, y1=0.0, label="a.left", external=True),
            Stub(section="a", side="right", x0=1.7, y0=0.0, x1=2.25, y1=0.0),
            Stub(section="b", side="left", x0=4.7, y0=0.0, x1=4.15, y1=0.0),
            Stub(section="b", side="right", x0=8.1, y0=0.0, x1=8.65, y1=0.0, label="b.right", external=True),
        ],
        connections=[Polyline(kind="connection", x=[2.25, 3.2, 3.2, 4.15], y=[0, 0, 0, 0])],
        launch=Stub(section="a", side="left", x0=-1.7, y0=0.0, x1=-2.25, y1=0.0, label="a.left", external=True),
        monitors=[Label(text="monitor m", x=6.4, y=-1.2, frame="data", ha="center", va="top")],
        limits=(-4.5, 10.9, -2.3, 2.3),
        neighbors={"a": {"right": "b"}, "b": {"left": "a"}},
        menu=MenuState(planes=["x-y", "z-x", "z-y", "schematic"], ports=["a.left", "b.right"], selected={"plane": "schematic"}),
    )
