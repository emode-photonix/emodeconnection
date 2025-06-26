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
from .types import EModeError, object_from_dict, TaggedModel

logger = getLogger(__name__)


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


class EModeClient:
    def __init__(self, cache: Cache, host: str = "127.0.0.1"):
        self.cache = cache
        self.host = host
        self.port: int = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(60)

        t = time.perf_counter()
        wait_time = 60
        while wait_time:
            try:
                with open(cache.port_path, "r") as f:
                    self.port = int(f.read())
            except FileNotFoundError:
                pass
            finally:
                cache.port_path.unlink(missing_ok=True)

            if self.port != 0:
                break
            else:
                wait_time -= time.perf_counter() - t
                t = time.perf_counter()
            time.sleep(0.05)

        if wait_time <= 0:
            self.sock.close()
            raise ConnectionError("EMode connection error!")

        time.sleep(0.1)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(None)
        self.sock.sendall(self.get_connection_string().encode("utf-8"))
        time.sleep(0.1)
        self.connected = True

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

    def recv(self) -> dict:
        logger.debug("")
        data = self._recv_msg()
        json_data = json.loads(data.decode("utf-8"), object_hook=obj_hook)
        if isinstance(json_data, dict):
            if version := json_data.pop("__version__", None):
                if version == 1:
                    try:
                        o = object_from_dict(json_data)
                    except KeyError:
                        return json_data

                    if isinstance(o, EModeError):
                        raise o
                    else:
                        return o
        return json_data

    def send(self, data):
        logger.debug(f"sending {data=}")
        for kw in data:
            if isinstance(data[kw], np.ndarray):
                data[kw] = np.squeeze(data[kw]).tolist()

            if isinstance(data[kw], list) and len(data[kw]) == 1:
                data[kw] = data[kw][0]

            if isinstance(data[kw], TaggedModel):
                data[kw] = data[kw].model_dump()
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
        self.connected = False
