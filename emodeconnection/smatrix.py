"""Client-side scattering-matrix type.

Simplified relative to the server's internal S-matrix representation: no
combine(), assemble_vector(), scatter_vector(), or reorder_ports() in the
public API -- those are solver-internal circuit-assembly plumbing. This is
the read-only, inspect-and-plot object handed back by EME/EM_get results.
"""
from functools import cached_property
from typing import Literal

import numpy as np
from pydantic import ConfigDict

from .types import TaggedModel, register_type


@register_type
class SMatrix(TaggedModel):
    """Dense, port-structured scattering matrix returned by EME results.

    Ports are indexed 0..num_ports-1. Port i spans a contiguous range of
    modes whose count is port_sizes[i]. When available, port_names and
    wavelengths carry one entry per port (parallel to port_sizes) so ports
    can be addressed by name instead of bare index, e.g.
    ``S.transmission(from_port='left', to_port='right')``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    port_sizes: tuple[int, ...]
    data: np.ndarray  # complex, shape (total_modes, total_modes)
    port_names: tuple[str, ...] | None = None
    wavelengths: tuple[float, ...] | None = None

    @property
    def num_ports(self) -> int:
        return len(self.port_sizes)

    @property
    def total_modes(self) -> int:
        return sum(self.port_sizes)

    @cached_property
    def port_slices(self) -> list[slice]:
        """Precompute the mode-index slice for each port."""
        slices = []
        offset = 0
        for size in self.port_sizes:
            slices.append(slice(offset, offset + size))
            offset += size
        return slices

    def port_index(self, name: str, wavelength: float | None = None) -> int:
        """Resolve a port name (optionally disambiguated by wavelength) to its integer index."""
        if self.port_names is None:
            raise ValueError(
                f'this SMatrix has no port_names metadata; use an integer port index instead of {name!r}'
            )
        matches = [i for i, n in enumerate(self.port_names) if n == name]
        if wavelength is not None:
            if self.wavelengths is None:
                raise ValueError('this SMatrix has no wavelength metadata; cannot disambiguate by wavelength')
            matches = [i for i in matches if self.wavelengths[i] == wavelength]
        if not matches:
            suffix = f' at wavelength {wavelength}' if wavelength is not None else ''
            raise ValueError(f'no port named {name!r}{suffix}')
        if len(matches) > 1:
            wls = [self.wavelengths[i] for i in matches] if self.wavelengths is not None else None
            raise ValueError(
                f'port name {name!r} is ambiguous ({len(matches)} matches'
                + (f', wavelengths={wls}' if wls is not None else '')
                + ') -- pass wavelength= to disambiguate, or use an integer index'
            )
        return matches[0]

    def _resolve_port(self, port: int | str) -> int:
        """Resolve a single port reference (index or name) to an integer index."""
        if isinstance(port, str):
            return self.port_index(port)
        return port

    def _gather_indices(self, ports: int | str | list[int | str] | np.ndarray) -> np.ndarray:
        """Gather global mode indices for one or more ports (by index or name)."""
        if isinstance(ports, (int, str)):
            ports = [ports]
        indices: list[int] = []
        for p in ports:
            resolved = self._resolve_port(p)
            s = self.port_slices[resolved]
            indices.extend(range(s.start, s.stop))
        return np.array(indices, dtype=np.intp)

    def block(
        self,
        to_ports: int | str | list[int | str] | np.ndarray,
        from_ports: int | str | list[int | str] | np.ndarray,
    ) -> np.ndarray:
        """Extract the dense submatrix mapping from_ports -> to_ports (by index or name)."""
        row_idx = self._gather_indices(to_ports)
        col_idx = self._gather_indices(from_ports)
        return self.data[np.ix_(row_idx, col_idx)]

    def transmission(self, from_port: int | str = 0, to_port: int | str = 1) -> np.ndarray:
        if self.num_ports != 2:
            raise ValueError('transmission() only valid for 2-port S-matrices')
        return self.block(to_port, from_port)

    def reflection(self, port: int | str = 0) -> np.ndarray:
        if self.num_ports != 2:
            raise ValueError('reflection() only valid for 2-port S-matrices')
        return self.block(port, port)

    def unitarity_error(self) -> float:
        """||S^H S - I||_fro."""
        data = self.block(list(range(self.num_ports)), list(range(self.num_ports)))
        return float(np.linalg.norm(data.conj().T @ data - np.eye(self.total_modes)))

    def insertion_loss_dB(self, from_port: int | str, to_port: int | str, mode: int = 0) -> float:
        col = self.block(to_port, from_port)[:, mode]
        return float(-10 * np.log10(np.sum(np.abs(col) ** 2)))

    def plot(self, kind: Literal['abs**2', 'phase'] = 'abs**2', ax=None):
        """
        Plot the scattering matrix as an N x N grid of block heatmaps.

        Parameters
        ----------
        kind : 'abs**2' or 'phase'
            What to plot: squared magnitude or phase.
        ax : array of Axes, optional
            Pre-existing (num_ports, num_ports) array of Axes to draw into.
            If None, a new figure is created.

        Returns
        -------
        The matplotlib Figure.
        """
        from matplotlib import pyplot as plt
        from matplotlib.colors import Normalize

        N = self.num_ports

        if kind == 'abs**2':
            def transform(M):
                return np.abs(M) ** 2
            norm = Normalize(vmin=0.0, vmax=1.0)
            cmap = 'viridis'
            clabel = r'$|S|^2$'
        elif kind == 'phase':
            def transform(M):
                return np.mod(np.angle(M), 2 * np.pi)
            norm = Normalize(vmin=0.0, vmax=2 * np.pi)
            cmap = 'hsv'
            clabel = r'$\arg S$'
        else:
            raise ValueError(f'Unknown kind: {kind}')

        if ax is None:
            fig, ax = plt.subplots(
                nrows=N, ncols=N,
                figsize=(3 * N + 1, 3 * N + 1),
                constrained_layout=True,
                squeeze=False,
            )
        else:
            ax = np.atleast_2d(ax)
            fig = ax.flat[0].get_figure()

        im = None
        for i in range(N):
            for j in range(N):
                data = transform(self.block(i, j))
                im = ax[i, j].imshow(data, norm=norm, cmap=cmap)
                if self.port_names is not None:
                    ax[i, j].set_title(f'S[{self.port_names[i]}, {self.port_names[j]}]')
                else:
                    ax[i, j].set_title(f'S[{i},{j}]')

        if im is not None:
            fig.colorbar(im, ax=ax.ravel().tolist(), label=clabel)

        return fig
