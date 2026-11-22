"""EMode's colormap family, which every plot of this client and of EMode draws with.

The family has three colormaps and a line palette:

- ``sequential``, the brand's "emode" map, for magnitudes, intensities and the index,
- ``diverging``, the brand's "emode_div" map, for real and imaginary parts (signed
  values about zero, drawn with a symmetric range so zero sits at the center, #1F2D50),
- ``cyclic``, for phase, whose two ends meet (derived, since no cyclic map was delivered),
- the line palette (``SERIES``), for curves, the schematic and legend entries.

A panel never names a colormap. Its rasters say which kind they want, and
this module is the one place a kind becomes colors: stop lists for the browser
(``plotly_colorscale``) and colormaps registered with matplotlib under
``NAMES`` for static images (``get``), built from the same stops so the two
agree. ``plot(colormap=...)`` picks another map by name (``check_colormap``).
"emode" and "emode_div" are registered too, as aliases of the first two, so a
script's own matplotlib figure can name them as the brand's files do.

**The colors.** ``BRAND``, ``SERIES``, ``SERIES_DARK``, ``GUI`` and ``PLOT`` are the
brand palette (``emode_palette.py``) and the two maps are the brand's colormap files,
copied verbatim. Nothing in EMode invents or mixes a color: every color a plot or the
browser interface draws is one of these. ``tests/test_colormaps.py`` compares them with the
brand files when those are present. ``delivered()`` is the one function that gathers
the delivery, and ``family()`` fills in what is not delivered.

**Series.** Curves take the line colors in order, slot 1, 2, 3 and so on, and never
cycle (``series_color``). At most 8 are drawn in a color: every further series is folded
into one "Other" series drawn in Mist, with a single legend entry (``series_label``). A
series keeps its color when others are filtered out, as its slot is its own index. A
scatter plot, a map or a small multiple uses only the first 3 (``SCATTER_SERIES``).
Status colors (success, error) are never series colors, and the GUI is light mode only.
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from typing import Any

from .types import ArgumentError, EModeError

KINDS = ("sequential", "diverging", "cyclic")

#: The matplotlib name each kind is registered under.
MPL_NAMES = {kind: f"emode-{kind}" for kind in KINDS}

#: Every registered name, in the order of ``KINDS``.
NAMES = tuple(MPL_NAMES.values())

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

# ------------------------------------------------------------------ the brand's colors

#: The brand's "emode" colormap (emode_field_colormap.py), 256 stops from Night Ink to Chalk, verbatim.
_EMODE = (
    "#10151F", "#141623", "#111725", "#0F1826", "#151829", "#14192A", "#14192D", "#141A2F",
    "#131B30", "#111C32", "#0F1D34", "#161D36", "#141E38", "#131F3A", "#12203C", "#11213E",
    "#0F2240", "#162242", "#132344", "#192248", "#182249", "#17234B", "#15254D", "#14264F",
    "#192752", "#182753", "#152855", "#132857", "#19295A", "#1A295C", "#1B2A5E", "#182C60",
    "#162C62", "#132D65", "#192C68", "#182E69", "#182F6B", "#182F6E", "#163070", "#133172",
    "#1A3175", "#1F3076", "#243179", "#23327B", "#22337E", "#213380", "#1E3481", "#213686",
    "#1C3787", "#1F3787", "#253889", "#243A8B", "#243B8C", "#243D8D", "#253F88", "#243F8E",
    "#22418F", "#294191", "#2D4092", "#304091", "#383F93", "#3B3F92", "#3F3F92", "#453D95",
    "#453F95", "#424097", "#444095", "#494099", "#4D4099", "#504098", "#514097", "#534095",
    "#584099", "#5A3F99", "#5C3F99", "#5F3F99", "#613F97", "#674099", "#683F9A", "#6A3F98",
    "#6D3F98", "#6E4097", "#6F3F98", "#723F95", "#754199", "#774098", "#793F97", "#7B3F97",
    "#7D4096", "#7F4096", "#823F94", "#843E92", "#853F91", "#884096", "#8A4095", "#8C4095",
    "#8E4194", "#914093", "#933F92", "#963E8D", "#973F88", "#9D3D86", "#9F3D86", "#A13D85",
    "#A23E84", "#A44189", "#A54089", "#A74087", "#A84185", "#AA4085", "#AC4085", "#AE4184",
    "#B04283", "#B24182", "#B44180", "#B6407C", "#B84077", "#BA4076", "#BC4175", "#BE4175",
    "#BF4274", "#C14273", "#C24371", "#C5426E", "#C74267", "#C84365", "#CA4463", "#CB4364",
    "#CD4463", "#CE4661", "#CF4760", "#D1475F", "#D34964", "#D54B62", "#D74C60", "#D84D5F",
    "#DA4E67", "#DC5066", "#DD5165", "#DF5363", "#E15461", "#E2555F", "#E45764", "#E55862",
    "#E65960", "#E75B5B", "#E85C58", "#EA5E57", "#EB5F57", "#EB6056", "#EC6254", "#ED6353",
    "#EE6551", "#EE6650", "#EF6857", "#F06A56", "#F16B55", "#F16D55", "#F26F54", "#F47159",
    "#F47258", "#F47456", "#F47654", "#F67859", "#F5795A", "#F57B59", "#F57D5B", "#F57F5F",
    "#F88163", "#F88361", "#F78560", "#F7865F", "#F88964", "#F88B64", "#F88D64", "#F88F64",
    "#F99269", "#FB926B", "#FC926D", "#FE926F", "#FD9674", "#FD9872", "#FD9971", "#FC9B71",
    "#FC9D71", "#FD9E71", "#FDA071", "#FDA170", "#FCA377", "#FCA575", "#FCA773", "#FCA872",
    "#FCAA71", "#FCAC71", "#FCAD71", "#FCAF71", "#FCB170", "#FCB26F", "#FCB476", "#FCB674",
    "#FCB773", "#FCBA71", "#FBBC71", "#FDBC71", "#FDBE71", "#FDC071", "#FDC170", "#FDC26F",
    "#FFC26F", "#FDC674", "#FDC872", "#FDC971", "#FDCB71", "#FCCC71", "#FACF71", "#F9D26B",
    "#FBD26B", "#FED26D", "#FFD26F", "#F9D575", "#FCD674", "#FCD980", "#FED881", "#FDDA89",
    "#FDDB8B", "#FDDC8F", "#FDDE8F", "#FCDF98", "#FAE098", "#F7E19C", "#F9E4A4", "#FCE4A4",
    "#FBE8AA", "#FAE8AF", "#FCE7B1", "#FFE6B2", "#FEE8B9", "#FFE8BC", "#FFE8BF", "#FDEBC8",
    "#FCEDC8", "#F8EFCA", "#F7F1CD", "#F7F1D0", "#F9F2D6", "#F5F3DB", "#F9F6E2", "#F9F7E1",
    "#FCF6E2", "#FFF6E2", "#FDF8E9", "#FEF8EC", "#FFF8F0", "#FFF9F8", "#FFFBF8", "#FDFCF8",
)

#: The brand's "emode_div" colormap (emode_field_colormap_diverging.py), 511 stops, verbatim. Zero is the middle
#: stop, the favicon blue #1F2D50.
_EMODE_DIV = (
    "#F8FAFF", "#F8F9FF", "#F8F9FE", "#F1F7FF", "#F2F6FF", "#F2F6FE", "#F3F5FF", "#F3F5FE",
    "#F4F4FE", "#E7F1FF", "#E7F1FE", "#E8F0FF", "#E8F0FE", "#E8EFFE", "#E8EEFE", "#E8EDFE",
    "#E8ECFD", "#E8EBFD", "#E0E8FD", "#E0E8FC", "#DFE8FD", "#DEE8FD", "#DCE8FD", "#DBE8FD",
    "#D9E8FE", "#D4E4FC", "#D5E3FB", "#D6E2FB", "#D6E2FA", "#D7E1FA", "#D1E0F7", "#D0DFF8",
    "#D0DEF8", "#D0DDF8", "#CFDCF8", "#CEDBF8", "#CDDAF8", "#CCDAF8", "#CBD9F8", "#CAD8F9",
    "#C9D8F9", "#C6D6F5", "#C6D5F6", "#C5D5F7", "#C6D4F7", "#C2D2F4", "#C1D2F5", "#C2D1F5",
    "#C1D1F6", "#BED0F8", "#BDCFF8", "#BCCEF8", "#B8CCF0", "#B8CBF0", "#B7CBF1", "#B7CAF1",
    "#B7C9F1", "#B7C8F2", "#B6C8F3", "#B2C6F0", "#B3C5F0", "#B3C5EF", "#B3C4F1", "#B2C4F2",
    "#ADC4F4", "#A7C1EF", "#A8C0EE", "#A8C0ED", "#A8BFED", "#A8BEEC", "#A8BDEC", "#A8BCEB",
    "#A8BBEB", "#A8BAEA", "#A8B9E9", "#A1B8E8", "#A2B7E8", "#A1B7E9", "#A2B6E9", "#9FB4E4",
    "#9EB3E5", "#9CB3E5", "#9BB2E6", "#9AB2E6", "#99B1E7", "#99B0E8", "#94AFE4", "#94AEE4",
    "#94ADE4", "#94ACE4", "#94ABE4", "#94AAE4", "#8FA9E0", "#90A8E0", "#8FA8E1", "#8EA7E2",
    "#8CA7E2", "#8BA6E3", "#89A6E3", "#86A3E0", "#86A3DF", "#87A2DE", "#88A1DE", "#89A0DD",
    "#899FDC", "#809FDD", "#809EDC", "#809DDC", "#809CDB", "#809BDA", "#809ADA", "#8099DA",
    "#8099D9", "#7D96D4", "#7C95D5", "#7A95D5", "#7895D6", "#7994D6", "#7894D7", "#7492D4",
    "#7591D4", "#7491D5", "#7590D5", "#708FD2", "#708ED2", "#708DD2", "#708CD2", "#6E8CD3",
    "#6D8BD3", "#6C8AD3", "#6B8AD3", "#6A89D3", "#6587CE", "#6686CD", "#6686CC", "#6785CC",
    "#6884CB", "#6884CA", "#6282C8", "#6381C8", "#6281C9", "#6380C9", "#5F7FC5", "#5F7EC5",
    "#5F7DC5", "#5E7DC5", "#5E7CC5", "#5D7BC5", "#5C7AC5", "#5B7AC5", "#5A79C5", "#5777C0",
    "#5777BF", "#5876BF", "#5975BE", "#5975BD", "#5273BD", "#5372BC", "#5372BB", "#5471BA",
    "#5471B9", "#5570B9", "#4F6FB7", "#4F6EB7", "#506DB6", "#506CB6", "#4E6CB7", "#4D6BB7",
    "#4C6AB7", "#4C67B0", "#4B67B0", "#4A67B0", "#4967B0", "#4A66B1", "#4966B1", "#4364AE",
    "#4463AD", "#4562AD", "#4562AC", "#4661AB", "#4760AA", "#405FA8", "#405EA8", "#415EA7",
    "#415DA7", "#415CA7", "#415BA7", "#425AA6", "#3E58A1", "#3D58A1", "#3C58A1", "#3B57A2",
    "#3A57A2", "#3957A2", "#3455A0", "#35549F", "#36539E", "#36539D", "#37529C", "#38519B",
    "#30509A", "#314F99", "#324F98", "#324E98", "#334D97", "#334C97", "#344B96", "#2F4B92",
    "#2F4A92", "#2E4992", "#2D4992", "#2C4893", "#2B4893", "#26468F", "#27458F", "#27458E",
    "#28448D", "#29438C", "#29438B", "#29438A", "#244187", "#254087", "#263F87", "#273F86",
    "#283E85", "#223E82", "#233D81", "#233C81", "#243C80", "#253B7E", "#253A7C", "#25397B",
    "#253979", "#1F3877", "#203876", "#223775", "#233774", "#1F3670", "#20366F", "#21356D",
    "#21356B", "#213569", "#233567", "#1F3463", "#223361", "#233360", "#24335D", "#25325B",
    "#253259", "#203057", "#213056", "#223055", "#232F55", "#1F2E50", "#1E2E50", "#1F2D50",
    "#202D4F", "#202E4F", "#262F52", "#253052", "#263051", "#273050", "#27314F", "#2A3354",
    "#2C3354", "#2E3453", "#303452", "#313451", "#323450", "#353654", "#363653", "#373652",
    "#383651", "#3A3650", "#3B374F", "#3D374F", "#403955", "#413954", "#413A54", "#423A53",
    "#433B52", "#443B51", "#483C56", "#493C55", "#493D55", "#4B3D55", "#4C3E55", "#4E3E55",
    "#4F3F55", "#503F54", "#513F53", "#554156", "#564155", "#574154", "#584153", "#5A4152",
    "#5B4152", "#5C4458", "#5E4458", "#604457", "#614456", "#624455", "#634454", "#664658",
    "#674657", "#684656", "#6A4655", "#6B4655", "#6C4754", "#6E4754", "#6F4754", "#724958",
    "#734957", "#744956", "#754955", "#754A55", "#764A54", "#784A52", "#7B4A51", "#804B59",
    "#804C59", "#814C58", "#814D58", "#824D57", "#834D56", "#844D55", "#894E59", "#894F59",
    "#8A4F59", "#8C4F59", "#8E4F59", "#8F5058", "#905057", "#924F56", "#934F55", "#965159",
    "#975158", "#985157", "#9A5156", "#9B5156", "#9D5156", "#9E5255", "#A05255", "#A35459",
    "#A3545A", "#A45458", "#A55457", "#A65456", "#A95759", "#AA5759", "#AB5759", "#AD5759",
    "#AF5759", "#B05759", "#B25757", "#B35756", "#B45755", "#B75959", "#B85958", "#BA5957",
    "#BB5957", "#BD5957", "#BE5A57", "#C05A57", "#C15A56", "#C25A55", "#C35B54", "#C75C59",
    "#C85C58", "#C95C57", "#CA5D57", "#CC5D57", "#CE5D57", "#CF5E57", "#D15E56", "#D25E55",
    "#D35F54", "#D45F53", "#D65F51", "#D86156", "#D96155", "#DA6254", "#DC6254", "#DD6353",
    "#DF6353", "#E16352", "#E26450", "#E3644F", "#E4634F", "#E66752", "#E76751", "#E86750",
    "#E8684E", "#E76A4F", "#E76B4F", "#E76C4F", "#E76D4F", "#EA7055", "#EB7154", "#EC7253",
    "#EC7352", "#ED7352", "#EE7451", "#EE7757", "#EE7856", "#EF7955", "#EF7A55", "#EF7C55",
    "#F07D55", "#F07E55", "#F07F55", "#F28259", "#F1835A", "#F0845A", "#F08558", "#F08657",
    "#F4895C", "#F48A5D", "#F38C60", "#F38D60", "#F38E60", "#F38F60", "#F3905F", "#F49364",
    "#F39464", "#F39563", "#F39662", "#F49967", "#F49A67", "#F49B67", "#F49C67", "#F49D67",
    "#F39F68", "#F3A068", "#F3A167", "#F7A46C", "#F6A56D", "#F5A66E", "#F4A76F", "#F3A86F",
    "#F5AB75", "#F5AC75", "#F4AE76", "#F4AF76", "#F4B076", "#F3B176", "#F3B275", "#F5B57A",
    "#F4B67B", "#F3B77D", "#F2B87E", "#F1BA7F", "#F1BB7F", "#F3BD86", "#F3BE86", "#F2C087",
    "#F1C187", "#F0C287", "#F0C386", "#F3C68C", "#F2C78E", "#F1C88F", "#F0C98F", "#F0CA8F",
    "#F1CC97", "#F1CD97", "#F0CE98", "#F0CF98", "#EFD098", "#F0D098", "#F2D3A3", "#F1D4A3",
    "#F0D5A3", "#F0D6A2", "#F0D8A9", "#F1D8A8", "#F0D9A8", "#F0DAA8", "#F0DBA8", "#F4DDB4",
    "#F3DEB5", "#F2DFB6", "#F2E0B6", "#F1E1B6", "#F5E3BA", "#F5E3BC", "#F4E4BE", "#F3E5C0",
    "#F2E6C0", "#F5E7C5", "#F4E8C5", "#F8E9C9", "#F8EACB", "#F8EBCD", "#F7ECD1", "#F6EDD2",
    "#F5EED3", "#F8EFD9", "#F8F0DA", "#F7F1DD", "#F6F2E0", "#F9F4E4", "#FAF4E4", "#FBF4E4",
    "#F9F8EC", "#FAF8EE", "#FBF8F0", "#FCF8F0", "#FBFBF8", "#FCFBF8", "#FDFCF8",
)

#: The brand palette (emode_palette.py), verbatim. Every hex follows the brand's digit rule.
BRAND = {
    # Neutrals
    "Chalk":     "#FDFCF8",  # page background
    "Haze":      "#E5E3DD",  # hairlines, borders, light panels
    "Mist":      "#9B9A98",  # disabled, placeholder (2.7:1 on Chalk: not for text that must be read)
    "Charcoal":  "#535256",  # secondary text (7.6:1 on Chalk)
    "Ink":       "#201A23",
    "Night Ink": "#10151F",  # body text, dark backgrounds
    # Brand colors
    "Blue":      "#253F88",  # wordmark, links, primary buttons
    "Coral":     "#F03A5F",  # logo gradient start
    "Orange":    "#FE5F3E",  # logo gradient middle, highlights (3.0:1 on Chalk: large text or marks only)
    "Amber":     "#FCBA71",
    "Gold":      "#F9D575",  # logo gradient end (not a text color on light)
    "Favicon":   "#1F2D50",  # favicon square, field-plot background (emode_div centre)
    # Status (reserved: state only, always with an icon or label)
    "Success":   "#227636",
    "Error":     "#8E1719",
}

SERIES_NAMES = ["blue", "orange", "teal", "coral", "violet", "ochre", "plum", "green"]

SERIES = [          # light backgrounds (Chalk / white)
    "#3957C2",      # 1 blue    (Blue, lifted for lines)
    "#FE5F3E",      # 2 orange  (brand Orange, exact)
    "#23A3A0",      # 3 teal
    "#CC2150",      # 4 coral   (brand Coral, deepened for contrast)
    "#A979F1",      # 5 violet
    "#B88523",      # 6 ochre   (Gold"s hue, darkened to read on light)
    "#A93680",      # 7 plum
    "#3FA050",      # 8 green
]

SERIES_DARK = [     # dark backgrounds (Night Ink / Favicon blue)
    "#395FC9",      # 1 blue
    "#D33F20",      # 2 orange
    "#2AA6A0",      # 3 teal
    "#CA2150",      # 4 coral
    "#A377E9",      # 5 violet
    "#BC8828",      # 6 ochre
    "#B03986",      # 7 plum
    "#52A955",      # 8 green
]

GUI = {  # light mode (the site is light-only for now)
    "background":     "#FDFCF8",  # Chalk
    "surface":        "#FDFCF8",  # cards and panels sit on Chalk, separated by a Haze border
    "surface_alt":    "#E5E3DD",  # Haze: table stripes, sidebars, selected rows
    "border":         "#E5E3DD",  # Haze
    "text":           "#10151F",  # Night Ink, 17.8:1
    "text_secondary": "#535256",  # Charcoal, 7.6:1
    "text_disabled":  "#9B9A98",  # Mist
    "link":           "#253F88",  # Blue, 9.5:1
    "primary":        "#253F88",  # Blue buttons
    "primary_text":   "#FDFCF8",  # Chalk on Blue
    "primary_hover":  "#263870",
    "focus":          "#253F88",
    "highlight":      "#FE5F3E",  # Orange: active tab underline, badges, icons (not small text)
    "success":        "#227636",  # 5.5:1
    "error":          "#8E1719",  # 9.0:1
    "warning_fill":   "#FCBA71",  # Amber fill with Night Ink text (10.8:1)
}

PLOT = {
    "light": {"background": "#FDFCF8", "text": "#10151F", "axes": "#535256",
              "grid": "#E5E3DD", "series": SERIES},
    "dark":  {"background": "#10151F", "text": "#E5E3DD", "axes": "#9B9A98",
              "grid": "#2B303E", "series": SERIES_DARK},
}

#: The kind each brand colormap name is an alias of.
ALIASES = {"emode": "sequential", "emode_div": "diverging"}

#: The color of every series past the 8th, folded into one "Other" series (Mist).
OTHER = BRAND["Mist"]

#: How many series a line plot colors before the rest fold into "Other".
MAX_SERIES = 8

#: How many series a scatter plot, a map or a small multiple colors, as any two colors can touch.
SCATTER_SERIES = 3


@dataclass(frozen=True)
class Family:
    """The family's colors as ``'#rrggbb'`` stops, evenly spaced from 0 to 1.

    ``diverging`` and ``cyclic`` may be None, and are then derived from
    ``sequential``: the diverging map runs it backward to its dark end at
    zero and forward again, so a value's magnitude has the same color on
    either side of zero, and the cyclic map runs it forward and back so its
    ends meet. The brand delivered a diverging map but no cyclic one.
    """

    sequential: tuple[str, ...]
    diverging: tuple[str, ...] | None = None
    cyclic: tuple[str, ...] | None = None
    lines: tuple[str, ...] = tuple(SERIES)


def delivered() -> Family:
    """The brand's colors: the "emode" and "emode_div" stops and the 8 series colors.

    The cyclic map is not delivered, so it derives from the sequential one
    (``family()``). ``tests/test_colormaps.py`` holds what is delivered to the
    brand files and to what a plot needs: the sequential map monotonic in
    lightness, the diverging map symmetric about its center, and the line colors
    distinguishable as the brand palette states it.
    """
    return Family(sequential=_EMODE, diverging=_EMODE_DIV, lines=tuple(SERIES))


# ------------------------------------------------------------------ the family


def family() -> Family:
    """The delivered colors with every kind filled in, checked."""
    given = delivered()
    sequential = _checked(given.sequential, "sequential")
    diverging = given.diverging or sequential[::-1] + sequential[1:]
    cyclic = given.cyclic or sequential + sequential[-2::-1]
    return Family(
        sequential=sequential,
        diverging=_checked(diverging, "diverging"),
        cyclic=_checked(cyclic, "cyclic"),
        lines=_checked(given.lines, "lines", minimum=1),
    )


def _checked(colors: tuple[str, ...], what: str, minimum: int = 2) -> tuple[str, ...]:
    colors = tuple(colors)
    if len(colors) < minimum:
        raise ValueError(f"the {what} colors need at least {minimum}, got {len(colors)}")
    for color in colors:
        if not isinstance(color, str) or not _HEX.match(color):
            raise ValueError(f"the {what} colors are '#rrggbb' strings, got {color!r}")
    return colors


def kind_of(name: str) -> str | None:
    """The kind `name` is (a kind or a registered name), or None for any other map."""
    if name in KINDS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    for kind, registered in MPL_NAMES.items():
        if name == registered:
            return kind
    return None


def _even(colors: tuple[str, ...]) -> list[tuple[float, str]]:
    n = len(colors) - 1
    return [(i / n, c) for i, c in enumerate(colors)]


def stops(kind: str) -> list[tuple[float, str]]:
    """The ``(position, '#rrggbb')`` stops of `kind` (or its registered name), positions 0 to 1."""
    resolved = kind_of(kind)
    if resolved is None:
        raise ValueError(f"unknown colormap kind {kind!r}, expected one of {KINDS}")
    return _even(getattr(family(), resolved))


def line_palette() -> tuple[str, ...]:
    """The line colors, in the order curves take them."""
    return family().lines


def series_color(index: int, limit: int | None = None) -> str:
    """The color of the series at `index` (0 for the first): its slot's, never cycling.

    A series at or past `limit` (``MAX_SERIES`` unless a scatter plot, a map or small
    multiples ask for ``SCATTER_SERIES``) is part of the one "Other" series, in Mist.
    """
    lines = line_palette()
    cap = min(len(lines), MAX_SERIES if limit is None else limit)
    return lines[index] if 0 <= index < cap else OTHER


def series_label(index: int, label: str | None, limit: int | None = None) -> str | None:
    """`label` for a series that has its own color, "Other" for the first folded one, else None.

    The folded series share one legend entry, and a series without a label has none.
    """
    cap = min(len(line_palette()), MAX_SERIES if limit is None else limit)
    if index < cap:
        return label
    return "Other" if index == cap and label is not None else None


def figure_style(active: bool = True) -> Any:
    """A context in which a new matplotlib figure is light (``plot_style``), or one that changes nothing.

    The rc parameters change only inside the ``with`` block, and are restored when it ends.
    """
    if not active:
        return contextlib.nullcontext()
    return _matplotlib().rc_context(plot_style())


def plot_style() -> dict[str, Any]:
    """The matplotlib rc parameters of a light figure: Chalk, Night Ink text, Charcoal axes, Haze grid.

    Apply it with ``matplotlib.rc_context``, never to the global ``rcParams``, since this
    client runs in the user's own Python.
    """
    light = PLOT["light"]
    return {
        "figure.facecolor": light["background"],
        "axes.facecolor": light["background"],
        "savefig.facecolor": light["background"],
        "text.color": light["text"],
        "axes.labelcolor": light["text"],
        "axes.titlecolor": light["text"],
        "xtick.color": light["axes"],
        "ytick.color": light["axes"],
        "xtick.labelcolor": light["text"],
        "ytick.labelcolor": light["text"],
        "axes.edgecolor": light["axes"],
        "grid.color": light["grid"],
        "legend.frameon": False,
    }


def plotly_colorscale(kind: str) -> list[list[Any]]:
    """`kind` (or its registered name) as a Plotly colorscale, ``[[position, color], ...]``."""
    return [[p, c] for p, c in stops(kind)]


def color_at(kind: str, fraction: float) -> str:
    """The color `fraction` (0 to 1) of the way along `kind`, linearly between its stops."""
    points = stops(kind)
    f = min(1.0, max(0.0, float(fraction)))
    for (p0, c0), (p1, c1) in zip(points, points[1:]):
        if f <= p1:
            t = 0.0 if p1 == p0 else (f - p0) / (p1 - p0)
            a, b = _rgb(c0), _rgb(c1)
            return "#" + "".join(f"{round(x + t * (y - x)):02x}" for x, y in zip(a, b))
    return points[-1][1]


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def schematic_colors() -> dict[str, str]:
    """The schematic's colors, all palette colors: blocks, launch, monitors, connections."""
    return {
        "text": GUI["text"],
        "block_face": BRAND["Haze"],
        "block_edge": SERIES[0],
        "launch": SERIES[1],
        "monitor": SERIES[2],
        "connection": BRAND["Charcoal"],
    }


def overlay_colors() -> dict[str, str]:
    """The colors of the lines drawn over a field panel and between schematic blocks.

    The overlays sit on the field maps, whose background is the dark end, so they take the
    series colors stepped for dark backgrounds: the port is slot 3 (teal), the facet slot 1
    (blue), the monitor slot 4 (coral) and the slice slot 2 (orange). The waveguide outline is
    white, as in the brand's reference mode plot, and the connections are Haze.
    """
    return {
        "outline": "#ffffff",
        "port": SERIES_DARK[2],
        "facet": SERIES_DARK[0],
        "monitor": SERIES_DARK[3],
        "slice": SERIES_DARK[1],
        "connection": BRAND["Haze"],
        "loop": BRAND["Haze"],
        "stub": BRAND["Haze"],
    }


def all_stops() -> dict[str, Any]:
    """Every kind's stops, the registered names and the line palette, as the browser tab asks for them."""
    return {
        "kinds": {k: stops(k) for k in KINDS},
        "names": dict(MPL_NAMES),
        "lines": list(line_palette()),
    }


# ------------------------------------------------------------------ matplotlib


def _matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as error:
        raise EModeError(
            "Drawing a figure here needs matplotlib. Install it with `pip install matplotlib`."
        ) from error
    return matplotlib


def register() -> tuple[str, ...]:
    """Register the family with matplotlib under ``NAMES`` (once), and return the names."""
    matplotlib = _matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    for kind, name in MPL_NAMES.items():
        if name not in matplotlib.colormaps:
            points = stops(kind)
            cmap = LinearSegmentedColormap.from_list(name, points, N=max(256, len(points)))
            matplotlib.colormaps.register(cmap, name=name)
    for alias, kind in ALIASES.items():
        # the brand's own files register these names too: the first to arrive wins
        if alias not in matplotlib.colormaps:
            matplotlib.colormaps.register(matplotlib.colormaps[MPL_NAMES[kind]], name=alias)
    return NAMES


def get(name: str = "sequential") -> Any:
    """The matplotlib colormap of a kind, a registered name, or any colormap matplotlib knows.

    For a figure of your own in EMode's colors, pass
    ``cmap=emode.colormaps.get('sequential')``.
    """
    matplotlib = _matplotlib()
    kind = kind_of(name)
    if kind is not None:
        register()
        return matplotlib.colormaps[name if name in ALIASES else MPL_NAMES[kind]]
    return matplotlib.colormaps[name]


def mpl_colormap(kind: str, name: str | None = None) -> Any:
    """The matplotlib colormap of `kind`, or of `name` when one is given."""
    return get(kind if name is None else name)


def resolve(kind: str, colormap: str | None = None) -> str:
    """The name a raster of `kind` is drawn with: `colormap` when given, else the family's."""
    return colormap if colormap is not None else MPL_NAMES[kind]


def check_colormap(name: Any, function: str = "plot") -> str:
    """`name` if a plot can draw with it, else an ``ArgumentError`` naming the choices.

    The family's names and kinds need nothing else. Any other name must be a
    colormap matplotlib has registered.
    """
    choices = (
        f"one of EMode's colormaps ({', '.join(NAMES + tuple(ALIASES))}), a kind ({', '.join(KINDS)}), "
        "or the name of a colormap matplotlib has registered"
    )
    if isinstance(name, str) and kind_of(name) is not None:
        return name
    if not isinstance(name, str) or name not in _matplotlib().colormaps:
        raise ArgumentError(f"colormap must be {choices}, got {name!r}", function, "colormap")
    return name
