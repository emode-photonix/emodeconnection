from .emodeconnection import EMode
from .eph_utils import open_file, get, inspect
from .types import MaterialSpec, MaterialProperties
from subprocess import Popen
from platform import system

__all__ = ["EMode", "open_file", "get", "inspect", "MaterialSpec", "MaterialProperties", "EModeLogin"]


def EModeLogin():
    emodecmd = "emode"
    if system() == "Windows":
        emodecmd = "EMode.exe"
    Popen([emodecmd])
