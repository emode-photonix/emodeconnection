"""EMode's colormap family (``emode/colormaps.py``): its properties, its names, ``colormap=``.

The family's colors are a drop-in (``colormaps.delivered()``). These tests hold
whatever is delivered to what a plot needs: a sequential map that only gets
lighter, a diverging map as light at -x as at +x, a cyclic map whose ends
meet, line colors far enough apart to tell apart, and the same stops in the
browser (Plotly) and in a static image (matplotlib).
"""
from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from emode import colormaps
from emode.types import ArgumentError
from emode.ui.figure import PlotOptions

from color_science import color_difference, hex_to_rgb, oklab, oklab_lightness

mpl = pytest.importorskip("matplotlib")

#: The smallest CIEDE2000 distance between two line colors. 20 is well past
#: the about 2 at which two flat patches look different, since a thin line
#: needs much more than a patch to be told apart.
MIN_LINE_DISTANCE = 20.0

#: The brand's own validation of its maps: OKLab lightness of the diverging map's +v and -v match
#: within 0.012 (its docstring), and no step between neighbors passes the visible threshold, 0.02.
SYMMETRY_TOLERANCE = 0.012
MAX_STEP = 0.02


def _sampled_lightness(name: str) -> np.ndarray:
    """OKLab L of every entry of the registered matplotlib colormap's lookup table."""
    cmap = colormaps.get(name)
    return np.array([oklab_lightness(cmap(i)) for i in range(cmap.N)])


def _stop_lightness(kind: str) -> np.ndarray:
    return np.array([oklab_lightness(hex_to_rgb(c)) for _, c in colormaps.stops(kind)])


# ------------------------------------------------------------------ the family's properties


def test_the_sequential_map_gets_lighter_at_every_step():
    """In OKLab, the brand's own space: in CIELAB L* the map's hex rounding steps back by up to 0.054."""
    for values in (_stop_lightness("sequential"), _sampled_lightness("emode-sequential")):
        assert np.all(np.diff(values) > 0), f"L steps back by {-np.diff(values).min():.5f}"
        assert values[-1] - values[0] > 0.7, "the map spans too little lightness to read"


def test_the_diverging_map_is_symmetric_in_lightness_about_its_center():
    for values in (_stop_lightness("diverging"), _sampled_lightness("emode-diverging")):
        assert np.abs(values - values[::-1]).max() <= SYMMETRY_TOLERANCE
        # each half is monotonic toward the dark center, so a value's magnitude reads the same either side
        middle = len(values) // 2
        assert np.all(np.diff(values[: middle + 1]) < 0)
        assert np.all(np.diff(values[middle:]) > 0)


def test_no_map_has_a_visible_step_between_neighbors():
    for kind in colormaps.KINDS:
        samples = np.array([oklab(hex_to_rgb(c)) for _, c in colormaps.stops(kind)])
        assert np.linalg.norm(np.diff(samples, axis=0), axis=1).max() < MAX_STEP, kind


def test_the_cyclic_maps_ends_meet():
    points = colormaps.stops("cyclic")
    assert points[0][1] == points[-1][1]
    cmap = colormaps.get("emode-cyclic")
    np.testing.assert_allclose(cmap(0.0), cmap(1.0), atol=1 / 255)


def test_the_line_colors_are_distinguishable():
    """As the brand palette states it: neighbors in the order curves take them, and the first 3 as all pairs."""
    lines = colormaps.line_palette()
    assert len(lines) == 8
    pairs = list(zip(lines, lines[1:])) + list(itertools.combinations(lines[:3], 2))
    worst = min((color_difference(a, b), a, b) for a, b in pairs)
    assert worst[0] >= MIN_LINE_DISTANCE, f"{worst[1]} and {worst[2]} are only {worst[0]:.1f} apart"


# ------------------------------------------------------------------ one set of stops, two renderers


@pytest.mark.parametrize("kind", colormaps.KINDS)
def test_matplotlib_and_plotly_draw_from_the_same_stops(kind):
    name = colormaps.MPL_NAMES[kind]
    scale = colormaps.plotly_colorscale(name)
    assert scale == [[p, c] for p, c in colormaps.stops(kind)]
    assert colormaps.plotly_colorscale(kind) == scale
    cmap = colormaps.get(name)
    # matplotlib reads a position from its 256-entry table, so a stop between
    # two entries is drawn as the nearer one, within one step of the table
    step = np.abs(np.diff(cmap(np.arange(cmap.N))[:, :3], axis=0)).max()
    for position, color in scale:
        np.testing.assert_allclose(cmap(position)[:3], hex_to_rgb(color), atol=step + 0.5 / 255)


def test_every_name_is_registered_with_matplotlib():
    names = colormaps.register()
    assert names == colormaps.NAMES == ("emode-sequential", "emode-diverging", "emode-cyclic")
    for name in names:
        assert mpl.colormaps[name].name == name
    assert colormaps.register() == names  # a second call is harmless


def test_get_gives_the_registered_colormap_by_kind_or_name():
    assert colormaps.get("sequential").name == "emode-sequential"
    assert colormaps.get("emode-cyclic").name == "emode-cyclic"
    assert colormaps.get("viridis").name == "viridis"


def test_the_browser_gets_every_kind_the_names_and_the_lines():
    stops = colormaps.all_stops()
    assert set(stops["kinds"]) == set(colormaps.KINDS)
    assert stops["names"] == dict(colormaps.MPL_NAMES)
    assert stops["lines"] == list(colormaps.line_palette())


# ------------------------------------------------------------------ the drop-in


# ------------------------------------------------------------------ the delivered brand colors

BRAND_DIR = Path.home() / "My_Drive/Assets/Logos/EMode_styles_2026/00_Colors"
needs_brand_files = pytest.mark.skipif(not BRAND_DIR.is_dir(), reason="the brand color files are not on this machine")


def _brand_assignment(file_name: str, name: str):
    """The literal `name` assigned in a brand file, read without importing (and so registering) it."""
    tree = ast.parse((BRAND_DIR / file_name).read_text())
    namespace: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # the plain data assignments only (PLOT names SERIES), never an import or a call
            try:
                exec(compile(ast.Module([node], []), file_name, "exec"), namespace)
            except NameError:
                continue
    if name not in namespace:
        raise AssertionError(f"{file_name} assigns no {name}")
    return namespace[name]


def _diverging_file() -> str:
    return next(f.name for f in BRAND_DIR.glob("emode_field_colormap_diverging*.py") if "example" not in f.name)


def test_the_delivery_is_the_brands_256_and_511_stops_and_8_series_colors():
    given = colormaps.delivered()
    assert len(given.sequential) == 256  # the warm half of emode_div: stops 255 to 510 of the 511
    assert given.diverging is not None and len(given.diverging) == 511
    assert given.cyclic is None  # no cyclic map was delivered, so it derives from the sequential one
    assert len(given.lines) == 8
    assert given.diverging[255].upper() == "#1F2D50"  # zero sits on the favicon blue
    assert given.sequential[0].upper() == "#1F2D50"  # and the sequential map starts there
    assert given.sequential[-1].upper() == "#FDFCF8"  # and ends on Chalk


def test_the_sequential_map_is_the_warm_half_of_emode_div():
    """On data >= 0 this draws what the brand's symmetric-range rule draws, with no empty half on the colorbar."""
    given = colormaps.delivered()
    assert given.sequential == given.diverging[255:]
    colormaps.register()
    div, seq = mpl.colormaps["emode_div"], colormaps.get("sequential")
    for v in np.linspace(0.0, 1.0, 41):  # value v on vmin=-1, vmax=1 is position (1 + v) / 2
        np.testing.assert_allclose(seq(v)[:3], div((1 + v) / 2)[:3], atol=2 / 255)


def test_the_retired_emode_map_is_not_registered():
    colormaps.register()
    assert "emode" not in mpl.colormaps
    assert "emode" not in colormaps.ALIASES
    assert colormaps.kind_of("emode") is None
    with pytest.raises(ArgumentError):
        colormaps.check_colormap("emode")


@needs_brand_files
def test_the_delivery_equals_the_brand_files_exactly():
    given = colormaps.delivered()
    diverging = _brand_assignment(_diverging_file(), "COLORS")
    assert list(given.diverging) == diverging
    assert list(given.sequential) == diverging[255:]
    assert list(given.lines) == _brand_assignment("emode_palette.py", "SERIES")


@needs_brand_files
def test_the_palette_constants_equal_the_brand_palette():
    for name in ("BRAND", "SERIES", "GUI", "PLOT"):
        assert getattr(colormaps, name) == _brand_assignment("emode_palette.py", name), name
    assert colormaps.SERIES_DARK == colormaps.SERIES  # a kept alias: one set for light and dark


@needs_brand_files
def test_the_generated_brand_module_is_current_with_the_brand_files():
    """``emode/_brand.py`` is what ``scripts/sync_brand_colors.py`` writes from today's brand files."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("sync_brand_colors", root / "scripts/sync_brand_colors.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    assert (root / "emode/_brand.py").read_text() == sync.render(BRAND_DIR), "rerun scripts/sync_brand_colors.py"


def test_the_palette_is_loaded_from_the_generated_brand_module():
    from emode import _brand

    for name in ("BRAND", "SERIES", "SERIES_NAMES", "GUI", "PLOT", "RULES"):
        assert getattr(colormaps, name) is getattr(_brand, name), name
    given = colormaps.delivered()
    assert given.diverging is _brand.EMODE_DIV and given.sequential == _brand.EMODE_DIV[255:]
    assert not hasattr(_brand, "EMODE") and not hasattr(_brand, "SERIES_DARK")


def test_no_module_names_a_matplotlib_colormap():
    """Every map is the brand's: no module of the package names plasma, viridis or another stock map."""
    stock = {"plasma", "viridis", "inferno", "magma", "cividis", "jet", "turbo", "hsv", "twilight",
             "coolwarm", "seismic", "bwr", "RdBu", "RdBu_r", "Blues", "Reds", "Greens", "gray"}
    package = Path(colormaps.__file__).parent
    found = []
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in stock:
                found.append(f"{path.relative_to(package)}:{node.lineno} {node.value!r}")
    assert not found, found


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_use_style_registers_the_brand_maps_and_sets_the_mode(mode):
    with mpl.rc_context():
        colormaps.use_style(mode)
        params = mpl.rcParams
        assert {"emode_div", *colormaps.NAMES} <= set(mpl.colormaps)
        assert params["image.cmap"] == "emode_div"
        assert params["axes.prop_cycle"].by_key()["color"] == colormaps.PLOT[mode]["series"]
        assert params["figure.facecolor"] == colormaps.PLOT[mode]["background"]
        assert params["text.color"] == colormaps.PLOT[mode]["text"]
    assert mpl.rcParams["image.cmap"] == mpl.rcParamsDefault["image.cmap"]


def test_the_brand_maps_are_the_delivered_stops():
    colormaps.register()
    cmap = mpl.colormaps["emode_div"]
    given = colormaps.delivered().diverging
    assert mpl.colors.to_hex(cmap(0.0)).upper() == given[0].upper()
    assert mpl.colors.to_hex(cmap(1.0)).upper() == given[-1].upper()
    assert mpl.colors.to_hex(mpl.colormaps["emode_div"](0.5)).upper() == "#1F2D50"


def test_plot_style_refuses_an_unknown_mode():
    with pytest.raises(ArgumentError):
        colormaps.plot_style("sepia")


def test_the_diverging_map_has_an_odd_number_of_stops_so_zero_has_one():
    stops = colormaps.stops("diverging")
    assert stops[len(stops) // 2][0] == 0.5


def test_the_brand_names_are_aliases_of_the_family():
    names = colormaps.register()
    assert "emode_div" in set(mpl.colormaps)
    assert colormaps.get("emode_div").name == "emode_div"
    for alias, kind in (("emode_div", "diverging"),):
        np.testing.assert_allclose(mpl.colormaps[alias](np.linspace(0, 1, 511)), colormaps.get(kind)(np.linspace(0, 1, 511)))
        assert colormaps.kind_of(alias) == kind
        assert colormaps.check_colormap(alias) == alias
        assert colormaps.plotly_colorscale(alias) == colormaps.plotly_colorscale(kind)
    assert names == colormaps.NAMES  # the aliases are extra, not family members
    assert colormaps.register() == names  # a second call neither fails nor re-registers


def test_a_brand_file_imported_first_does_not_break_registration():
    pytest.importorskip("matplotlib")
    from matplotlib.colors import ListedColormap

    name = "emode_div"
    if name in mpl.colormaps:
        mpl.colormaps.unregister(name)
    mpl.colormaps.register(ListedColormap(colormaps.delivered().diverging, name=name))
    colormaps.register()
    assert mpl.colormaps[name].N == 511


# ------------------------------------------------------------------ series colors, in order and never cycling


def test_series_take_their_colors_in_order_and_never_cycle():
    lines = colormaps.line_palette()
    assert [colormaps.series_color(i) for i in range(8)] == list(lines)
    assert colormaps.series_color(8) == colormaps.BRAND["Mist"] == colormaps.OTHER
    assert colormaps.series_color(30) == colormaps.OTHER


def test_scatter_and_map_series_use_only_the_first_3_colors():
    lines = colormaps.line_palette()
    assert [colormaps.series_color(i, limit=colormaps.SCATTER_SERIES) for i in range(5)] == list(lines[:3]) + [colormaps.OTHER] * 2


def test_the_extra_series_share_one_other_legend_entry():
    labels = [colormaps.series_label(i, f"curve {i}") for i in range(11)]
    assert labels[:8] == [f"curve {i}" for i in range(8)]
    assert labels[8] == "Other" and labels[9:] == [None, None]
    assert colormaps.series_label(9, None) is None
    assert colormaps.series_label(0, None) is None


def test_the_figure_style_is_the_light_plot_style():
    style = colormaps.plot_style()
    light = colormaps.PLOT["light"]
    assert style["figure.facecolor"] == style["axes.facecolor"] == style["savefig.facecolor"] == light["background"]
    assert style["text.color"] == light["text"] == colormaps.GUI["text"]
    assert style["axes.edgecolor"] == light["axes"]
    assert style["grid.color"] == light["grid"]


def test_the_drop_in_is_one_function_and_the_others_derive_from_it(monkeypatch):
    """Replacing ``delivered()`` alone changes every kind, the browser's and matplotlib's."""
    gray = ("#000000", "#808080", "#ffffff")
    monkeypatch.setattr(colormaps, "delivered", lambda: colormaps.Family(sequential=gray))
    assert [c for _, c in colormaps.stops("sequential")] == list(gray)
    assert [c for _, c in colormaps.stops("diverging")] == ["#ffffff", "#808080", "#000000", "#808080", "#ffffff"]
    assert [c for _, c in colormaps.stops("cyclic")] == ["#000000", "#808080", "#ffffff", "#808080", "#000000"]
    assert colormaps.plotly_colorscale("emode-sequential")[1] == [0.5, "#808080"]
    # with no line colors delivered the brand's series stay
    assert colormaps.line_palette() == tuple(colormaps.SERIES)


def test_a_delivered_diverging_or_cyclic_map_is_used_as_given(monkeypatch):
    family = colormaps.Family(
        sequential=("#000000", "#ffffff"),
        diverging=("#0000ff", "#ffffff", "#ff0000"),
        cyclic=("#ff0000", "#00ff00", "#0000ff", "#ff0000"),
        lines=("#111111", "#eeeeee"),
    )
    monkeypatch.setattr(colormaps, "delivered", lambda: family)
    assert [c for _, c in colormaps.stops("diverging")] == list(family.diverging)
    assert [c for _, c in colormaps.stops("cyclic")] == list(family.cyclic)
    assert colormaps.line_palette() == family.lines


def test_a_bad_delivery_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(colormaps, "delivered", lambda: colormaps.Family(sequential=("#000000", "red")))
    with pytest.raises(ValueError, match="'red'"):
        colormaps.stops("sequential")


# ------------------------------------------------------------------ colormap=


@pytest.mark.parametrize("name", colormaps.NAMES + colormaps.KINDS)
def test_colormap_takes_every_name_of_the_family(name):
    assert PlotOptions.from_arguments({"colormap": name}).colormap == name


def test_colormap_takes_every_colormap_matplotlib_has_registered():
    for name in list(mpl.colormaps):
        assert PlotOptions.from_arguments({"colormap": name}).colormap == name


def test_an_unknown_colormap_is_refused_naming_the_choices():
    with pytest.raises(ArgumentError) as info:
        PlotOptions.from_arguments({"colormap": "plasmaa"})
    message = str(info.value)
    assert "'plasmaa'" in message
    for name in colormaps.NAMES:
        assert name in message
    assert "matplotlib" in message
    assert info.value.argument == "colormap"


def test_a_colormap_that_is_not_a_string_is_refused():
    with pytest.raises(ArgumentError, match="colormap"):
        PlotOptions.from_arguments({"colormap": 3})
