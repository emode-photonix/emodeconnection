"""Turn client results into JSON for the agent, saving large arrays to files."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

# Arrays with more elements than this are saved to a file and returned by path.
INLINE_LIMIT = 1000


class FileStore:
    """Where large results go, created on first use, and the paths written so far."""

    def __init__(self, directory: Path | Callable[[], Path]) -> None:
        self._directory = directory
        self._owned = callable(directory)
        self._counter = itertools.count()
        self.written: list[str] = []

    @property
    def directory(self) -> Path:
        if callable(self._directory):
            self._directory = self._directory()
        self._directory.mkdir(parents=True, exist_ok=True)
        return self._directory

    def cleanup(self) -> None:
        """Delete the array files written, and the directory if this store made it and it is empty.

        Plots are kept, since the agent may have shown them to the user.
        """
        for name in self.written:
            if name.endswith(".npy"):
                Path(name).unlink(missing_ok=True)
        if self._owned and not callable(self._directory):
            try:
                self._directory.rmdir()
            except OSError:
                pass  # plots remain in it

    def new_path(self, stem: str, suffix: str) -> Path:
        return self.directory / f"{stem}-{next(self._counter):03d}{suffix}"

    def save_array(self, array: np.ndarray) -> dict[str, Any]:
        path = self.new_path("array", ".npy")
        np.save(path, array)
        self.written.append(str(path))
        return {
            "file": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }


def _array(array: np.ndarray, store: FileStore) -> Any:
    if array.dtype == object:
        return [to_jsonable(item, store) for item in array.tolist()]
    if array.size > INLINE_LIMIT:
        return store.save_array(array)
    if np.iscomplexobj(array):
        return {"real": array.real.tolist(), "imag": array.imag.tolist()}
    return array.tolist()


def to_jsonable(value: Any, store: FileStore) -> Any:
    """The JSON form of a client result.

    NumPy arrays become lists (complex ones a ``{"real", "imag"}`` pair), or a
    ``{"file", "shape", "dtype"}`` reference to a ``.npy`` file when large.
    Mappings, including the client's mode and wavelength dictionaries, become
    objects with string keys. Client types become objects of their fields.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), store)
    if isinstance(value, np.ndarray):
        return _array(value, store)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v, store) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v, store) for v in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, BaseModel):
        return {
            name: to_jsonable(getattr(value, name), store)
            for name in type(value).model_fields
        }
    return repr(value)
