"""Write ``emode/_brand.py`` from the EMode Photonix brand color files.

Usage::

    python scripts/sync_brand_colors.py <brand color folder>

The folder holds the brand's ``emode_palette.json`` and
``emode_field_colormap_diverging_511.csv``. The diverging file may carry a
download suffix such as `` (1)``. The folder is only read. The generated module is
plain Python, so the package and the compiled EMode server carry the colors without
data files. Rerun this whenever the brand files change, and commit the result.
"""
from __future__ import annotations

import csv
import json
import pprint
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "emode" / "_brand.py"

HEADER = '''"""The EMode Photonix brand colors, generated from the brand's color files. Do not edit.

Regenerate with ``python scripts/sync_brand_colors.py <brand color folder>``.
``emode.colormaps`` is the module to import: it builds the colormaps from these.
"""
'''


def _one(folder: Path, pattern: str) -> Path:
    found = sorted(p for p in folder.glob(pattern) if p.is_file())
    if len(found) != 1:
        raise SystemExit(f"expected one {pattern} in {folder}, found {[p.name for p in found]}")
    return found[0]


def _hexes(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return [row["hex"].upper() for row in csv.DictReader(f)]


def _literal(name: str, value: object, comment: str) -> str:
    return f"#: {comment}\n{name} = {pprint.pformat(value, width=100, sort_dicts=False)}\n"


def render(folder: Path) -> str:
    """The text of ``emode/_brand.py`` generated from the brand files in `folder`."""
    palette = json.loads(_one(folder, "emode_palette.json").read_text())
    emode_div = _hexes(_one(folder, "emode_field_colormap_diverging_511*.csv"))
    if len(emode_div) != 511:
        raise SystemExit(f"expected 511 stops, got {len(emode_div)}")
    plot = {mode: {**palette["plot"][mode], "series": palette["series"]} for mode in ("light", "dark")}
    parts = [
        HEADER,
        _literal("BRAND", palette["brand"], "Named brand colors. Success and Error are for status only."),
        _literal("SERIES_NAMES", palette["series_names"], "The series colors' names, in slot order."),
        _literal("SERIES", palette["series"], "Series colors, the same on light and dark backgrounds, slot 1 first."),
        _literal("GUI", palette["gui"], "Light-mode interface roles."),
        _literal("PLOT", plot, "Figure colors for light and dark backgrounds."),
        _literal("RULES", palette["rules"], "The brand's color rules, as the palette states them."),
        _literal("EMODE_DIV", tuple(emode_div),
                 'The "emode_div" colormap: 511 stops, -1 to 1, zero the middle stop (Favicon).'),
    ]
    return "\n\n".join(parts)


def main(folder: Path) -> None:
    OUT.write_text(render(folder))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).expanduser())
