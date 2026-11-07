"""EMode's Python client under its short name.

``pip install emode`` installs emodeconnection, and this package re-exports
it, so ``import emode`` works today::

    import emode

    em = emode.EMode()
    em.shape(name='core', material='Si', width=600, height=400)
    em.FDM()

Every name emodeconnection exports is available here, and ``emode.<module>``
reaches its modules. A later release of EMode moves the client itself into
this package.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version as _distribution_version

import emodeconnection as _emodeconnection
from emodeconnection import *  # noqa: F401, F403
from emodeconnection import __all__  # noqa: F401

__version__ = _distribution_version('emodeconnection')


def __getattr__(name: str) -> object:
    """Forward anything not re-exported above to emodeconnection, modules included."""
    try:
        return getattr(_emodeconnection, name)
    except AttributeError:
        pass
    try:
        return importlib.import_module(f'emodeconnection.{name}')
    except ModuleNotFoundError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_emodeconnection)))
