import json
import struct
import socket
import base64
import numpy as np
import time
import os
import sys
import platform
from typing import Any, Optional, Union
from logging import getLogger
from .file_utils import Cache
from .types import EModeError, reconstruct, serialize

logger = getLogger(__name__)

# The wire protocol version this client reads. The solver stamps it on every
# dictionary reply as "__version__".
PROTOCOL_VERSION = 1


def obj_hook(dct: dict) -> Optional[Union[np.ndarray, dict]]:
    if "__ndarray__" in dct:
        data = base64.b64decode(dct["__ndarray__"])
        if dct["dtype"] == "object":
            return None  # placeholder value
        else:
            return np.frombuffer(data, dct["dtype"]).reshape(dct["shape"])
    return dct


def detect_ide() -> str:
    """Return the name of the IDE or 'command_line'."""
    env = os.environ
    mods = sys.modules

    if "VSCODE_PID" in env:
        return "vscode"
    if env.get("PYCHARM_HOSTED") == "1" or "pydevd" in mods:
        return "pycharm"
    if "SPYDER_ENV" in env or "SPYDER_ARGS" in env or "spyder_kernels" in mods:
        return "spyder"
    if any(name.startswith("idlelib") for name in mods):
        return "idle"
    # Jupyter variants
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython().__class__.__name__
        if shell in ("ZMQInteractiveShell", "IPKernelApp"):
            return "jupyter"
    except Exception:
        pass
    return "command_line"


def detect_platform() -> dict[str, str]:
    """Return OS name, version, and architecture."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def detect_python() -> dict[str, Any]:
    """Return Python version info and IPython flag."""
    info = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_version_info": tuple(sys.version_info),
        "in_ipython": False,
    }
    try:
        from IPython import get_ipython  # type: ignore

        info["in_ipython"] = get_ipython() is not None
    except ImportError:
        pass
    return info


def collect_environment() -> dict[str, Any]:
    """Aggregate all environment info."""
    return {
        "ide": detect_ide(),
        "platform": detect_platform(),
        "python": detect_python(),
    }


def wait_for_port(cache: Cache, timeout: float = 60) -> int:
    """Wait for the solver to publish its port, then remove the port file.

    The file is removed only after a port has been read from it. Removing it
    after a failed read would delete a file the solver wrote in between, and
    the solver would then wait for a connection that never comes. A file that
    exists but doesn't hold a port yet was caught mid-write and is retried, and
    so is one Windows refuses to open (PermissionError) while the solver's
    atomic rename is moving it into place.
    """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with open(cache.port_path, "r") as f:
                port = int(f.read())
        except (FileNotFoundError, PermissionError, ValueError):
            time.sleep(0.05)
            continue
        cache.port_path.unlink(missing_ok=True)
        return port
    raise ConnectionError("EMode connection error!")


class EModeClient:
    def __init__(self, cache: Cache, host: str = "127.0.0.1"):
        self.cache = cache
        self.host = host
        # Whether arrays go as `__ndarray__` (an EMode of 1.0.8 or newer reads it).
        self.ndarray_wire = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(60)

        try:
            self.port = wait_for_port(cache)
        except ConnectionError:
            self.sock.close()
            raise

        time.sleep(0.1)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(None)
        self.sock.sendall(self.get_connection_string().encode("utf-8"))
        time.sleep(0.1)
        self._connected = True

    @property
    def connected(self) -> bool:
        if not self._connected:
            return False
        try:
            self.sock.setblocking(False)
            peek = self.sock.recv(1, socket.MSG_PEEK)
            if peek == b"":
                self._connected = False  # peer sent FIN (orderly close)
        except BlockingIOError:
            pass  # no data waiting -- still alive, just idle
        except OSError:
            self._connected = False  # e.g. connection reset
        finally:
            self.sock.setblocking(True)  # restore blocking mode (settimeout(None) equiv.)
        return self._connected

    def get_connection_string(self):
        try:
            _ = get_ipython()  # type: ignore
            conn_msg = "connected with iPython"
            self.ipython = True
        except NameError:
            conn_msg = "connected with Python"
            self.ipython = False

        if any("VSCODE" in name for name in os.environ):
            ide = "-VSCode"
        elif any("SPYDER" in name for name in os.environ):
            ide = "-Spyder"
        elif any("JPY_" in name for name in os.environ):
            ide = "-Jupyter"
        elif any("PYCHARM" in name for name in os.environ):
            ide = "-PyCharm"
        elif any("HOME" == name for name in os.environ):
            ide = "-IDLE"
        else:
            ide = "-cmd"

        return conn_msg + ide

    def recv(self) -> Any:
        """Read one reply from the solver.

        A dictionary reply is stamped with the protocol version in
        ``__version__``. A reply without a stamp (a scalar, a list, an array
        or None) is a plain value and is returned as is. A stamp other than
        `PROTOCOL_VERSION` raises an `EModeError`, and so does an error of a
        type this client does not know, with the solver's message.
        """
        logger.debug("")
        data = self._recv_msg()
        json_data = json.loads(data.decode("utf-8"), object_hook=obj_hook)
        if not (isinstance(json_data, dict) and "__version__" in json_data):
            return json_data
        version = json_data.pop("__version__")
        if version != PROTOCOL_VERSION:
            raise EModeError(
                f"this emodeconnection does not understand protocol version {version} "
                f"(it reads version {PROTOCOL_VERSION}). Upgrade it with "
                "'pip install --upgrade emodeconnection'."
            )
        o = reconstruct(json_data)
        if isinstance(o, EModeError):
            raise o
        name = o.get("__data_type__") if isinstance(o, dict) else None
        if isinstance(name, str) and name.endswith("Error") and "msg" in o:
            # An error type a newer solver added and this client predates.
            raise EModeError(f"{name}: {o['msg']}")
        return o

    def send(self, data):
        logger.debug(f"sending {data=}")
        data = serialize(data, ndarray=self.ndarray_wire)
        logger.debug(f"serialized {data=}")
        sendjson = json.dumps(data)
        self._send_raw(sendjson)

    def _send_raw(self, data: str):
        logger.debug(f"sending raw data: {data}")
        msg = bytes(data, encoding="utf-8")
        msg = struct.pack(">I", len(msg)) + msg
        self.sock.sendall(msg)

    def _recv_msg(self):
        raw_msglen = self._recv_raw(4)
        msglen = struct.unpack(">I", raw_msglen)[0]
        ret = self._recv_raw(msglen)
        return ret

    def _recv_raw(self, n: int):
        data = bytearray()
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                logger.info("expected data but didn't get it")
                raise ConnectionError(
                    f"connection interrupted? expected {n} bytes, but only got {len(data)}"
                )
            data.extend(packet)
        if n <= 200:
            logger.debug(f"exiting: received: {data=}")
        else:
            logger.debug(f"exiting: received {len(data)} bytes")
        return data

    def close(self):
        try:
            self.send({"function": "exit"})
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
        except Exception:
            pass
        self._connected = False
