from .emodeconnection import EMode
from .eph_utils import open_file, get, inspect
from .types import MaterialSpec, MaterialProperties, Grid, Field, GridSet, FieldSet
from .geometry import Pose, Line, Arc, BSpline, Path, Curve
from .port import Bounds, Port, AngledFacetMap, JunctionOverride, make_angled_facet_port
from .nonlinear import Chi2Process, SHGProcess, SFGProcess, DFGProcess, Source, Excitation, CircuitResponse
from .smatrix import SMatrix
from .traceback_filter import install as _install_traceback_filter
from subprocess import Popen
from platform import system

__all__ = [
    "EMode", "open_file", "get", "inspect",
    "MaterialSpec", "MaterialProperties",
    "Grid", "Field", "GridSet", "FieldSet",
    "Pose", "Line", "Arc", "BSpline", "Path", "Curve",
    "Bounds", "Port", "AngledFacetMap", "JunctionOverride", "make_angled_facet_port",
    "Chi2Process", "SHGProcess", "SFGProcess", "DFGProcess",
    "Source", "Excitation", "CircuitResponse",
    "SMatrix",
    "EModeLogin",
]

_install_traceback_filter()


def EModeLogin():
    emodecmd = "emode"
    if system() == "Windows":
        emodecmd = "EMode.exe"
    Popen([emodecmd])
