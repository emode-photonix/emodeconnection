"""The GDS layer view of a device: what ``plot(view='gds')`` gets back, and KLayout layer files.

``GDSView`` holds one ``GDSLayer`` per GDS (layer, datatype) of the device's
GDS cells, each with its polygons in nm on the chip, placed where the device
puts its sections (as ``plot(view='layout')`` and ``check_layout()`` place
them). A large cell arrives decimated to the panel's pixel size
(``pixel_nm``), every outline kept within a pixel of its own. ``shapes`` are
the simulation's shapes on the chip, which the view can draw over the layers.

``read_lyp`` reads a KLayout layer-properties file (``.lyp``) for a PDK's layer
names, colors and hatches.
"""
from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import ConfigDict, field_validator, model_validator

from .plotting import AxesInfo, MenuState, Region
from .types import ArgumentError, TaggedModel, register_type


@register_type
class GDSLayer(TaggedModel):
    """One GDS layer and datatype: its name and its polygons in nm.

    ``xy`` holds every polygon's vertices, ``(n, 2)`` as ``(z, x)`` on the chip,
    and polygon ``i`` is ``xy[offsets[i]:offsets[i + 1]]``, each ring open (its
    first vertex not repeated). ``name`` is the simulation shapes the layer is
    mapped to, empty when it is mapped to none.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    layer: int
    datatype: int = 0
    name: str = ""
    xy: np.ndarray
    offsets: np.ndarray

    @field_validator("xy", mode="before")
    @classmethod
    def _xy(cls, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape(-1, 2)

    @field_validator("offsets", mode="before")
    @classmethod
    def _offsets(cls, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.int64).ravel()

    @model_validator(mode="after")
    def _covers(self) -> "GDSLayer":
        offsets = self.offsets
        if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != len(self.xy) or np.any(np.diff(offsets) < 0):
            raise ValueError("offsets run from 0 to the number of vertices, never decreasing")
        return self

    @property
    def key(self) -> str:
        """``'layer/datatype'``, as KLayout names a layer."""
        return f"{self.layer}/{self.datatype}"

    @property
    def polygon_count(self) -> int:
        return int(self.offsets.size - 1)

    def polygons(self) -> List[np.ndarray]:
        """Each polygon's vertices, ``(n, 2)`` in nm."""
        return [self.xy[a:b] for a, b in zip(self.offsets[:-1], self.offsets[1:])]


@register_type
class GDSView(TaggedModel):
    """The device's GDS layers on the chip, for ``plot(view='gds')``.

    - ``layers``: one per (layer, datatype), ordered by layer then datatype.
    - ``shapes``: the simulation's shapes on the chip, ``name`` as
      ``'section.shape'``, for the overlay.
    - ``axes``: the panel's axes, ``z`` along and ``x`` across, in nm.
    - ``pixel_nm``: the panel's pixel size the polygons were decimated to, and
      ``decimated`` whether any were.
    - ``cells``: each GDS section's source cell, by section name.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    layers: List[GDSLayer] = []
    shapes: List[Region] = []
    axes: AxesInfo
    pixel_nm: float = 0.0
    decimated: bool = False
    cells: Dict[str, str] = {}
    menu: MenuState = MenuState()


# ------------------------------------------------------------------ KLayout layer properties

#: Plotly's fill patterns (and matplotlib's hatches) nearest each of KLayout's
#: standard dither patterns, ``I0`` to ``I45``. ``I0`` is solid, ``I1`` hollow.
#: Unlisted patterns are drawn solid.
DITHER_HATCH: Dict[int, str] = {
    **{i: "." for i in (2, 3, 14, 16, 17, 18, 41, 42)},
    **{i: "\\" for i in (4, 5, 6, 7, 25)},
    **{i: "/" for i in (8, 9, 10, 11, 26)},
    **{i: "x" for i in (12, 13, 15, 27, 28)},
    **{i: "+" for i in (19, 20, 21, 30, 31)},
    **{i: "-" for i in (22, 23, 34, 35, 37, 38)},
    **{i: "|" for i in (24, 32, 33, 36, 39, 40)},
}

#: The fill opacity of a filled layer read from a ``.lyp`` file, and of a hollow one.
LYP_OPACITY = 0.5


def _text(element: Optional[ElementTree.Element], tag: str) -> Optional[str]:
    child = element.find(tag) if element is not None else None
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _source_key(source: str) -> Optional[str]:
    """``'1/0@1'`` (KLayout's source) as ``'1/0'``, None for a wildcard or a name."""
    spec = source.split("@")[0].strip()
    layer, _, datatype = spec.partition("/")
    if not layer.isdigit() or (datatype and not datatype.isdigit()):
        return None
    return f"{int(layer)}/{int(datatype or 0)}"


def read_lyp(source: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """The layer styles of a KLayout layer-properties file, by ``'layer/datatype'``.

    `source` is the file's path, or its text. Each entry has the layer's
    ``name``, its fill ``color``, the ``hatch`` nearest its dither pattern
    (``''`` for solid), ``visible``, and the fill ``opacity`` (0 for a hollow
    layer). Entries whose source is a wildcard (``*/*``) name no layer and are
    skipped, and grouped layers are read as the others.
    """
    text = str(source)
    if not text.lstrip().startswith("<"):
        path = Path(text)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ArgumentError(f"cannot read the layer properties file {path}: {exc}", "plot", "layer_properties") from exc
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ArgumentError(f"not a KLayout layer properties file: {exc}", "plot", "layer_properties") from exc
    if root.tag != "layer-properties":
        raise ArgumentError(
            f"not a KLayout layer properties file: its root is <{root.tag}>, not <layer-properties>",
            "plot",
            "layer_properties",
        )
    styles: Dict[str, Dict[str, Any]] = {}
    for props in root.iter("properties"):
        key = _source_key(_text(props, "source") or "")
        if key is None:
            continue
        color = _text(props, "fill-color") or _text(props, "frame-color")
        dither = (_text(props, "dither-pattern") or "I0").lstrip("I")
        pattern = int(dither) if dither.isdigit() else 0
        styles[key] = {
            "name": _text(props, "name") or "",
            "color": color.lower() if color else None,
            "hatch": DITHER_HATCH.get(pattern, ""),
            "visible": (_text(props, "visible") or "true").lower() != "false",
            "opacity": 0.0 if pattern == 1 else LYP_OPACITY,
        }
    return styles
