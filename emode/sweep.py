"""What ``sweep()`` and ``chi2_spectrum()`` answer with: a ``SweepResult``.

A sweep's result has always been a dict: ``'key'`` (the swept key, or a list of
keys for a grid), ``'values'`` (what it took, or one list per key) and one entry
per requested result, plus what the kind of sweep reports beside them
(``'mode_order'``, ``'num_nodes'``, ``'interpolation_residual'``,
``'node_wavelengths'``, ``'solved_wavelengths'``, ``'unconverged'``).
``SweepResult`` is that dict, still indexed by key, iterated, tested with
``in`` and changed as before, which also plots itself::

    result = em.sweep(key='wavelength', values=wavelengths, result='effective_index')
    result['effective_index']     # as before
    result.plot()                 # one trace per mode against wavelength, in the tab
    em.plot(result)               # the same
    result.plot(file_name='neff') # a static image under EM_figures/

On the wire its entries sit at the top level beside the type tag, so a client
that predates the type reads the reply as the plain dict it always did.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Iterator, List, Literal, Union

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer, model_validator

from .types import TaggedModel, register_type

#: The entries a sweep reports beside its results, which are not plotted as results.
REPORTED = (
    "key",
    "values",
    "mode_order",
    "num_nodes",
    "interpolation_residual",
    "interpolation_residuals",
    "node_wavelengths",
    "solved_wavelengths",
    "unconverged",
)


@register_type
class SweepResult(TaggedModel, MutableMapping):
    """The result of ``sweep()`` or ``chi2_spectrum()``: a dict that plots itself.

    Index it by result name (``result['values']``), iterate it, call
    ``.keys()``, ``.values()``, ``.items()`` or ``.get()``, change an entry,
    or turn it into a plain ``dict`` with ``dict(result)``, as with the dict
    it replaces. ``layout`` says what was swept: ``'line'`` for one key,
    ``'grid'`` for several. ``results`` names the entries that are results
    rather than reports of the sweep. ``plot()`` draws it (see
    ``SweepResult.plot``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, values: Any) -> Any:
        """Accept the flat wire form, ``{'key': ..., 'values': ..., <result>: ...}``, as well as ``data=``."""
        if not isinstance(values, Mapping):
            return values
        if set(values) == {"data"} and isinstance(values["data"], Mapping):
            return {"data": dict(values["data"])}
        return {"data": dict(values)}

    @model_serializer(mode="wrap")
    def _to_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        dumped = handler(self)
        return {"__data_type__": dumped["__data_type__"], **dumped["data"]}

    # ------------------------------------------------------------ the dict

    def __getitem__(self, name: str) -> Any:
        return self.data[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.data[name] = value

    def __delitem__(self, name: str) -> None:
        del self.data[name]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, name: object) -> bool:
        return name in self.data

    def __or__(self, other: Mapping) -> "SweepResult":
        return SweepResult(data={**self.data, **dict(other)})

    def copy(self) -> "SweepResult":
        return SweepResult(data=dict(self.data))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SweepResult):
            other = other.data
        if not isinstance(other, Mapping):
            return NotImplemented
        if set(self.data) != set(other):
            return False
        return all(_same(self.data[k], other[k]) for k in self.data)

    # Unhashable, like the dict it stands in for.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"SweepResult({self.data!r})"

    def __str__(self) -> str:
        return repr(self)

    # ------------------------------------------------------------ what it holds

    @property
    def layout(self) -> Literal["line", "grid"]:
        """``'grid'`` for a sweep over several keys at once, ``'line'`` for one key."""
        return "grid" if isinstance(self.data.get("key"), (list, tuple)) else "line"

    @property
    def swept(self) -> List[str]:
        """The swept keys, one for a line, several for a grid."""
        key = self.data.get("key")
        return [str(k) for k in key] if isinstance(key, (list, tuple)) else [str(key)]

    @property
    def results(self) -> List[str]:
        """The entries that are results, in the order the sweep reported them."""
        return [k for k in self.data if k not in REPORTED]

    def plot(
        self,
        em: Any = None,
        block: bool = False,
        inline: Union[str, None] = None,
        file_name: Union[str, None] = None,
        file_type: str = "png",
        **arguments: Any,
    ) -> Any:
        """Plot the sweep in an EMode session's browser tab, or as a static image.

        One key draws each result against it, one trace per mode for a
        per-mode result (following the modes by their ``mode_order`` when the
        sweep tracked them), and the S-matrix from the first port to the last,
        one trace per output mode. Two keys draw a map, with line cuts against
        either key. An interpolated EME sweep marks its nodes and the held-out
        residual, an adaptive scan or ``chi2_spectrum()`` the wavelengths it
        solved.

        The tab's menus change the selection, and so do these arguments:
        ``result`` (a result's name, or a list of them drawn as traces of one
        panel), ``component`` (``'abs^2'``, ``'dB'``, ``'phase'``, ``'real'``
        or ``'imag'`` of a complex result), ``panel`` (``'map'`` or
        ``'against <key>'`` for a grid), ``mode`` (a grid's mode), ``input``
        and ``output`` (the S-matrix's ports, by name or index),
        ``input_mode`` and ``output_mode`` (one mode, or ``'all'`` of the
        output port's). A power result (a coupling, a conversion) is drawn
        ``'linear'`` or in ``'dB'``. The figure options of ``plot()`` apply
        (``title``, ``xlabel``, ``ylabel``, ``units='um'`` for a length key,
        ``font_size``, ``figure_size``, ``dpi``, ``colormap``, and ``vmin``
        and ``vmax``, the y range of a line panel or the color range of a
        map).

        `em` is the session whose tab shows it, by default the newest one still
        running. Returns a ``PlotHandle``, or a ``PlotImage`` with
        `file_name`, ``EMODE_HEADLESS=1`` or ``inline='png'``.
        """
        from .ui.session import show_data

        return show_data(
            self, em=em, block=block, inline=inline, file_name=file_name, file_type=file_type, **arguments
        )

    show = plot


def _same(a: Any, b: Any) -> bool:
    """Whether two entries hold the same values, arrays compared elementwise."""
    import numpy as np

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, BaseModel) or isinstance(b, BaseModel):
        # a model holding arrays (an SMatrix) compares field by field
        return type(a) is type(b) and _same(dict(a.__dict__), dict(b.__dict__))
    try:
        same = a == b
    except (TypeError, ValueError):
        return False
    if isinstance(same, bool):
        return same
    try:
        return bool(np.all(same))
    except (TypeError, ValueError):
        return False
