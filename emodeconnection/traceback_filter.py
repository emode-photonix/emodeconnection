"""
Hides internal emodeconnection frames (wrapper/call/recv, etc.) from
tracebacks of EModeError and its subclasses, so users see only their own
call site and the error message. These errors are server-validated and
expected (bad argument, wrong license, missing file, ...), not bugs in the
connection library itself, so the internal plumbing frames are just noise.

Python appends a traceback frame for every stack frame an exception unwinds
through as part of normal interpreter unwinding -- this can't be prevented
at the original `raise` site, only stripped after the fact from the
traceback object. That's done here, once, at import time.
"""
import os
import sys
import types

from .types import EModeError

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_internal(tb: types.TracebackType) -> bool:
    return tb.tb_frame.f_code.co_filename.startswith(_PACKAGE_DIR)


def _filter_traceback(tb):
    frames = []
    while tb is not None:
        if not _is_internal(tb):
            frames.append(tb)
        tb = tb.tb_next

    new_tb = None
    for frame_tb in reversed(frames):
        new_tb = types.TracebackType(
            new_tb, frame_tb.tb_frame, frame_tb.tb_lasti, frame_tb.tb_lineno
        )
    return new_tb


def _filter_chain(exc_value, _seen=None):
    if _seen is None:
        _seen = set()
    if exc_value is None or id(exc_value) in _seen:
        return
    _seen.add(id(exc_value))

    if isinstance(exc_value, EModeError):
        exc_value.__traceback__ = _filter_traceback(exc_value.__traceback__)

    _filter_chain(exc_value.__cause__, _seen)
    _filter_chain(exc_value.__context__, _seen)


def _make_excepthook(previous_hook):
    def _excepthook(exc_type, exc_value, tb):
        if isinstance(exc_value, EModeError):
            # The default hook reads exc_value.__traceback__ rather than the
            # `tb` argument, so it must be mutated in place, not just passed
            # through filtered.
            tb = _filter_traceback(tb)
            exc_value.__traceback__ = tb
            _filter_chain(exc_value.__cause__)
            _filter_chain(exc_value.__context__)
        previous_hook(exc_type, exc_value, tb)

    return _excepthook


def _make_ipython_handler():
    def _handler(shell, exc_type, exc_value, tb, tb_offset=None):
        tb = _filter_traceback(tb)
        exc_value.__traceback__ = tb
        _filter_chain(exc_value.__cause__)
        _filter_chain(exc_value.__context__)
        shell.showtraceback((exc_type, exc_value, tb), tb_offset=tb_offset)

    return _handler


_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True

    sys.excepthook = _make_excepthook(sys.excepthook)

    try:
        ipython = get_ipython()  # type: ignore
    except NameError:
        ipython = None

    if ipython is not None:
        ipython.set_custom_exc((EModeError,), _make_ipython_handler())
