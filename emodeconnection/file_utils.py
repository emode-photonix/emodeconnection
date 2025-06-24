import errno
import platform
from pathlib import Path
import os
import time
from typing import Optional


def config_dir(ensure_exists=True) -> Path:
    h = Path.home()
    s = platform.system()
    base = (
        os.environ.get("XDG_CONFIG_HOME")
        if s != "Windows"
        else os.environ.get("APPDATA")
    ) or (h / "Library" / "Application Support" if s == "Darwin" else h / ".config")
    p = Path(base) / "emode"

    if ensure_exists:
        p.mkdir(parents=True, exist_ok=True)
    return Path(base) / "emode"


def cache_dir(ensure_exists=True) -> Path:
    h = Path.home()
    s = platform.system()
    base = (
        os.environ.get("XDG_CACHE_HOME")
        if s != "Windows"
        else os.environ.get("LOCALAPPDATA")
    ) or (h / "Library" / "Caches" if s == "Darwin" else h / ".cache")
    p = Path(base) / "emode"
    if ensure_exists:
        p.mkdir(parents=True, exist_ok=True)
    return Path(base) / "emode"


class Cache:
    def __init__(self, port_file_label: Optional[str] = None):
        self.port_file_label = port_file_label
        self.cache = cache_dir(True)

    @property
    def port_path(self):
        if self.port_file_label is None:
            raise ValueError("Need to set port_file_label first")
        return self.cache / f"port_{self.port_file_label}.txt"

    @property
    def print_path(self):
        if self.port_file_label is None:
            raise ValueError("Need to set port_file_label first")
        return self.cache / f"emode_output_{self.port_file_label}.txt"

    def write_port_file(self, port):
        with open(self.port_path, "w") as f:
            f.write(str(port))

    def read_port_file(self, port):
        with open(self.port_path, "r") as f:
            return f.read()

    def cleanup(self, tries=20):
        for i in range(tries):
            try:
                self.port_path.unlink(missing_ok=True)
                self.print_path.unlink(missing_ok=True)
                break
            except OSError as e:
                if e.errno == errno.EBUSY:
                    time.sleep(0.5)

    def __del__(self):
        self.cleanup()
