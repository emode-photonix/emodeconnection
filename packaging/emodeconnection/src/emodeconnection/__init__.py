"""The old name of the `emode` package, kept so existing code keeps working.

The client's import package is ``emode``. This shim re-exports it and warns
once per process that the name changed::

    pip install emode
    import emode

``emodeconnection.<module>`` is the very module ``emode.<module>``, not a copy,
so classes, the wire-type registry, and pickles behave the same under both
names.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType

import emode as _emode
from emode import *  # noqa: F401, F403
from emode import __all__ as __all__  # noqa: PLC0414
from emode import __version__ as __version__  # noqa: PLC0414

warnings.warn(
    'The emodeconnection package is now called emode. Run "pip install emode" and replace '
    '"import emodeconnection" with "import emode". The emodeconnection name keeps working '
    "for now.",
    DeprecationWarning,
    stacklevel=2,
)

# The launch module was emodeconnection.emodeconnection. Inside the emode package it is emode.emode.
_RENAMED = {"emodeconnection": "emode"}


class _Alias(importlib.abc.Loader):
    """Hands the import system the emode module itself, so both names are one module."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module
        self._spec = module.__spec__

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return self._module

    def exec_module(self, module: ModuleType) -> None:
        # The import system stamps the alias spec on the module it was handed. Put the real
        # spec back, so the module still says it is emode.<name> (reload, relative imports).
        module.__spec__ = self._spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves ``emodeconnection.<name>`` to ``emode.<name>``, for any module at any depth."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("emodeconnection."):
            return None
        parts = fullname.split(".")[1:]
        parts[0] = _RENAMED.get(parts[0], parts[0])
        try:
            module = importlib.import_module(".".join(["emode", *parts]))
        except ModuleNotFoundError as err:
            if err.name and err.name.startswith("emode"):
                return None
            raise
        is_package = hasattr(module, "__path__")
        spec = importlib.machinery.ModuleSpec(
            fullname, _Alias(module), is_package=is_package
        )
        return spec


# Imports of emodeconnection.<name> (a submodule, however deep) resolve through the finder.
# The empty __path__ makes emodeconnection a package whose submodules only the finder supplies.
__path__: list[str] = []
if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name: str) -> object:
    """Forward anything not re-exported above, such as a submodule or a private name."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(_emode, name)
    except AttributeError:
        pass
    try:
        return importlib.import_module(f"emodeconnection.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_emode)))
