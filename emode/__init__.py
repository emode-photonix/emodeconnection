from .emode import EMode
from .eph_utils import open_file, get, inspect
from .types import MaterialSpec, MaterialProperties, Grid, Field, GridSet, FieldSet, WavelengthDict, Confinement, ModeDict
from .geometry import Pose, Line, Arc, BSpline, Path, Curve, LayoutReport
from .port import Bounds, Port, AngledFacetMap, JunctionOverride, make_angled_facet_port
from .nonlinear import Chi2Process, SHGProcess, SFGProcess, DFGProcess, Source, Excitation, CircuitResponse
from .smatrix import SMatrix
from .plot import PlotImage
from . import colormaps
from .plotting import MaterialView, PlotView, Schematic, SessionInfo, SMatrixView
from .free_space import FarField, FreeSpaceProjection, Screen, ScreenField
from .gds import GDSView, read_lyp
from .pulse import PulsedSource, PulsedProcess, Poling, PulseResponse, PulseMeasurement, PulseFit, RankReport
from .monitor import Monitor, PlaneField, BeamCoupling
from .grating import grating_sections
from .optimize import (
    Parameter, Objective, Constraint, Evaluation, OptimizationResult,
    ConvergenceStep, ConvergenceStage, ConvergenceResult,
)
from .traceback_filter import install as _install_traceback_filter
from .types import client_version as _client_version
from subprocess import Popen
from platform import system

__all__ = [
    "EMode", "open_file", "get", "inspect",
    "MaterialSpec", "MaterialProperties",
    "Grid", "Field", "GridSet", "FieldSet", "WavelengthDict", "Confinement", "ModeDict",
    "Pose", "Line", "Arc", "BSpline", "Path", "Curve", "LayoutReport",
    "Bounds", "Port", "AngledFacetMap", "JunctionOverride", "make_angled_facet_port",
    "Chi2Process", "SHGProcess", "SFGProcess", "DFGProcess",
    "Source", "Excitation", "CircuitResponse",
    "PulsedSource", "PulsedProcess", "Poling", "PulseResponse", "PulseMeasurement", "PulseFit", "RankReport",
    "Monitor", "PlaneField", "BeamCoupling",
    "grating_sections",
    "Parameter", "Objective", "Constraint", "Evaluation", "OptimizationResult",
    "ConvergenceStep", "ConvergenceStage", "ConvergenceResult",
    "SMatrix",
    "PlotImage", "PlotView", "Schematic", "colormaps",
    "FreeSpaceProjection", "FarField", "Screen", "ScreenField",
    "EModeLogin",
]

__version__ = _client_version()

_install_traceback_filter()


def EModeLogin():
    emodecmd = "emode"
    if system() == "Windows":
        emodecmd = "EMode.exe"
    Popen([emodecmd])
