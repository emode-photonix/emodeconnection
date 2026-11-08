"""The figure a headless ``plot()`` returns.

When the solver draws a plot without a display (no screen, ``file_name`` given,
or ``EMODE_HEADLESS=1``), it answers with a ``PlotImage`` instead of opening a
window. In a Jupyter or IPython session the client shows it in the cell and
``plot()`` returns ``None``. Elsewhere ``plot()`` returns the ``PlotImage``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import ConfigDict

from .types import EModeError, TaggedModel, register_type


@register_type
class PlotImage(TaggedModel):
    """A rendered plot.

    - ``file``: where the solver saved the figure, when ``plot()`` was given
      ``file_name``, else ``None``.
    - ``png``: the figure as PNG bytes in a ``uint8`` array, or ``None`` when
      only a file was saved. Sent when no file was asked for, and always to a
      notebook, which shows it inline.
    - ``dpi``: the resolution ``png`` was rendered at. A saved file keeps its
      own resolution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    file: str | None = None
    png: np.ndarray | None = None
    dpi: int = 150

    def _repr_png_(self) -> bytes | None:
        """The PNG bytes, which is how IPython draws this object in a cell."""
        if self.png is None:
            return None
        return self.png.tobytes()

    def save(self, path: str | Path) -> Path:
        """Write the PNG to ``path`` and return it."""
        data = self._repr_png_()
        if data is None:
            raise EModeError(f"this plot carries no image. It was saved to {self.file}")
        out = Path(path)
        out.write_bytes(data)
        return out
