from collections.abc import Iterator, Mapping
from typing import Any, Type, TypeVar, Optional, Union, get_origin, get_args
import sys

if sys.version_info >= (3, 10):
    from types import UnionType
else:
    UnionType = None

from enum import Enum
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    computed_field,
    field_validator,
)
import math
import numpy as np


def serialize(data: Any):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = serialize(value)
        return data
    elif isinstance(data, list):
        return [serialize(item) for item in data]
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return serialize(np.squeeze(data).tolist())
    elif isinstance(data, TaggedModel):
        return data.model_dump()
    if type(data).__module__ == np.__name__:
        # final catchall
        data = np.squeeze(data).tolist()
    else:
        return data

def _allows_none(tp: Any) -> bool:
    """
    Return True iff the type annotation *explicitly* admits `None`.
    Handles Optional[T], Union[..., None], and Annotated[…].
    """
    if tp is None or tp is type(None):
        return True

    origin = get_origin(tp)
    if origin is Union or origin is UnionType:  # Optional[T] is just Union[T, None]
        return type(None) in get_args(tp)
    if origin is getattr(__import__("typing"), "Annotated", None):
        # drill into Annotated[T, ...]  (first arg is the real annotation)
        return _allows_none(get_args(tp)[0])

    return False

class TaggedModel(BaseModel):
    @computed_field(repr=False)
    @property
    def __data_type__(self) -> str:
        return self.__class__.__name__

    @field_validator("*", mode="before")
    @classmethod
    def _nan_to_none(cls, v: Any, info: ValidationInfo):
        # this is necessary because matlab doesn't support None, so they are all
        # changed to NaNs in serialization.
        if not (isinstance(v, float) and math.isnan(v)):
            return v  # nothing to do
        if info.field_name is None:
            # this should never happen...
            return v

        field_type = cls.model_fields[info.field_name].annotation
        if _allows_none(field_type):
            return None  # safe to coerce
        return v  # leave `nan` as-is

T = TypeVar("T")

_TYPE_REGISTRY = {}

def register_type(cls: Type[T]) -> Type[T]:
    """Class decorator to register a wire type by class name."""
    name = cls.__name__
    if name in _TYPE_REGISTRY:
        raise ValueError(f"{name} already registered")
    _TYPE_REGISTRY[name] = cls
    return cls

def get_type(name: str) -> Type:
    try:
        return _TYPE_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown exception type: {name}")

class LicenseType(Enum):
    _2D = "2d"
    _3D = "3d"
    default = "default"

    def to_dict(self):
        return {"__type__": "LicenseType", "value": self.value}

    def __str__(self):
        return self.value

    @classmethod
    def coerce(cls, value: Any) -> Optional["LicenseType"]:
        """A LicenseType from whichever form of one turned up.

        The server holds live members; the wire carries `to_dict()`'s
        `{"__type__": "LicenseType", "value": ...}`, and `reconstruct` does not
        rebuild it -- it only knows the `__data_type__` tag -- so the client
        side sees that dict. Normalizing at the one place a license type
        enters means holders have a single declared type to reason about
        rather than three: `LicenseError.__str__` had a branch for each and
        got both of them wrong.
        """
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            value = value.get("value")
        try:
            return cls(value)
        except ValueError:
            return None

TensorType = Union[float, list[float], list[list[float]]]
DTensorType = Union[list[list[float]]]

class MaterialProperties(TaggedModel):
    """A user-defined material.

    `n`, `eps` or `mu` give its linear response and `d` (m/V) its chi(2). For the
    pulsed nonlinear solver:

    - `n2`: the Kerr nonlinear index in m^2/W.
    - `raman`: the delayed Kerr response as `(fraction, lines)`, with `lines` a
      list of `(weight, position, half_width)` damped oscillators, position and
      half width in cm^-1. The weights are normalized to sum to one.
    - `tpa`: the two-photon absorption coefficient in m/W.
    - `bandgap`: the bandgap in eV. Two-photon absorption is zero for light
      whose two photons carry less than the bandgap.
    """

    n: Optional[TensorType] = None
    eps: Optional[TensorType] = None
    mu: Optional[TensorType] = None
    d: Optional[DTensorType] = None
    n2: Optional[float] = None
    raman: Optional[tuple[float, list[tuple[float, float, float]]]] = None
    tpa: Optional[float] = None
    bandgap: Optional[float] = None

    # this is necessary to support np.ndarrays here...
    model_config = ConfigDict(arbitrary_types_allowed=True)

class MaterialSpec(TaggedModel):
    """A material and the orientation of its crystal axes in the simulation frame.

    `phi`, `theta` and `psi` are z-y-z Euler angles in radians. The rotation is
    active: it turns the crystal, and a material tensor T given on the crystal
    axes appears in the simulation frame as R T R^T, with
    R = Rz(phi) Ry(theta) Rz(psi). Read right to left, the crystal is first
    turned by `psi` about its own z axis, then by `theta` about y, then by `phi`
    about z. The crystal z axis ends up along
    (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)). `psi` only matters
    for a biaxial crystal (or a nonlinear tensor without rotational symmetry
    about z). Omitted angles are 0.
    """

    material: Union[str, MaterialProperties]
    theta: Optional[float] = None
    phi: Optional[float] = None
    psi: Optional[float] = None
    x: Optional[float] = None
    loss: Optional[float] = None  # dB/m

# Not decorated with @register_type for historical reasons (predates the decorator).
_TYPE_REGISTRY["MaterialSpec"] = MaterialSpec
_TYPE_REGISTRY["MaterialProperties"] = MaterialProperties

@register_type
class Grid(TaggedModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: np.ndarray
    y: np.ndarray

    is_pml: bool = False
    is_expanded: bool = False
    is_bc: bool = False

@register_type
class Field(TaggedModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    field: np.ndarray
    field_names: list[str]
    num_modes: int
    grid: Grid

@register_type
class GridSet(TaggedModel):
    """A `Grid` for each wavelength of a multi-wavelength profile."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    grids: dict[float, Grid]

    def __getitem__(self, wavelength: float) -> Grid:
        return self.grids[wavelength]

    def __iter__(self):
        return iter(self.grids)

    def __len__(self) -> int:
        return len(self.grids)

    def __contains__(self, wavelength: float) -> bool:
        return wavelength in self.grids

    def keys(self):
        return self.grids.keys()

    def values(self):
        return self.grids.values()

    def items(self):
        return self.grids.items()

    def get(self, wavelength: float, default=None):
        return self.grids.get(wavelength, default)

@register_type
class FieldSet(TaggedModel):
    """A `Field` for each wavelength of a multi-wavelength profile."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    fields: dict[float, Field]

    def __getitem__(self, wavelength: float) -> Field:
        return self.fields[wavelength]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, wavelength: float) -> bool:
        return wavelength in self.fields

    def keys(self):
        return self.fields.keys()

    def values(self):
        return self.fields.values()

    def items(self):
        return self.fields.items()

    def get(self, wavelength: float, default=None):
        return self.fields.get(wavelength, default)

@register_type
class WavelengthDict(TaggedModel, Mapping):
    """A read-only mapping from wavelength [nm] to one result per wavelength.

    Returned for a multi-wavelength profile by `get()` of a per-profile
    result, and by `effective_area()`, `group_index()`, `orthogonality()`,
    `scattering()`, `confinement()` and `report()`. It behaves like a `dict`
    keyed by float wavelength: index it, iterate it, call `.keys()`,
    `.values()`, `.items()` or `.get()`, or turn it into a plain `dict` with
    `dict(result)`. Any number equal to a wavelength names it, so
    `result[1550]` and `result[1550.0]` are the same entry.

    It exists because JSON object keys are always strings. A plain dict keyed
    by wavelength would arrive here keyed by `"1550.0"`. This type carries its
    keys through validation, which turns them back into floats.
    """

    model_config = ConfigDict(frozen=True)
    data: dict[float, Any]

    def __getitem__(self, wavelength: float) -> Any:
        return self.data[wavelength]

    def __iter__(self) -> Iterator[float]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, wavelength: object) -> bool:
        return wavelength in self.data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WavelengthDict):
            return self.data == other.data
        if isinstance(other, Mapping):
            return self.data == dict(other)
        return NotImplemented

    # Unhashable, like the dict it stands in for.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"WavelengthDict({self.data!r})"

    def __str__(self) -> str:
        return repr(self)

def object_from_dict(data: dict[str, Any]) -> Any:
    """
    Reconstructs an exception from its serialized form.
    Expects data["__data_type__"] to be the class name.
    """
    name = data.pop("__data_type__")
    if not name:
        raise KeyError("Missing 'type' in exception data")

    ExcClass = get_type(name)

    return ExcClass(**data)

def reconstruct(data: Any) -> Any:
    """
    Recursively rebuild registered wire types anywhere in a decoded payload.

    Any dict carrying a known "__data_type__" tag becomes an instance of the
    registered class; dicts with an unknown tag are left as plain dicts.
    Containers are walked bottom-up so nested tagged objects are rebuilt
    before their parent.
    """
    if isinstance(data, dict):
        data = {key: reconstruct(value) for key, value in data.items()}
        name = data.get("__data_type__")
        if name in _TYPE_REGISTRY:
            fields = {key: value for key, value in data.items() if key != "__data_type__"}
            return _TYPE_REGISTRY[name](**fields)
        return data
    if isinstance(data, list):
        return [reconstruct(item) for item in data]
    return data

@register_type
class EModeError(Exception):
    def __init__(self, msg: str = '', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self._custom_fields = []
        self.msg = msg

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_custom_fields":
            self._custom_fields.append(name)
        return super().__setattr__(name, value)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "__data_type__": self.__class__.__name__,
        }
        for f in self._custom_fields:
            d[f] = getattr(self, f, None)

        return d

@register_type
class ArgumentError(EModeError):
    def __init__(self, msg: str, function: Optional[str], argument: Optional[str]):
        super().__init__(msg)
        self.msg = msg
        self.function = function
        self.argument = argument

    def __str__(self):
        return f'ArgumentError: the argument: ({self.argument}) to function: ({self.function}) had error: "{self.msg}"'

@register_type
class EPHKeyError(EModeError):
    def __init__(self, msg: str, filename: str, key: str):
        super().__init__(msg)
        self.msg = msg
        self.filename = filename
        self.key = key

    def __str__(self):
        return f'EPHKeyError: the key: ({self.key}) doesn\'t exist in the file: ({self.filename}), "{self.msg}"'

@register_type
class FileError(EModeError):
    def __init__(self, msg: str, filename: str):
        super().__init__(msg)
        self.msg = msg
        self.filename = filename

    def __str__(self):
        return f'FileError: the file: "{self.filename}" had error: "{self.msg}"'

@register_type
class LicenseError(EModeError):
    def __init__(self, msg: str, license_type: Any):
        # Coerced here, so `self.license_type` is a LicenseType (or None) no
        # matter which side built this error -- the server passes a live
        # member, `reconstruct` passes the wire dict.
        license_type = LicenseType.coerce(license_type)
        super().__init__(msg, license_type)
        self.msg = msg
        self.license_type = license_type

    def __str__(self):
        emLicense = 'EMode3D' if self.license_type is LicenseType._3D else 'EMode2D'
        return f'LicenseError: current license: "{emLicense}", error msg: "{self.msg}"'

@register_type
class ShapeError(EModeError):
    def __init__(self, msg: str, shape_name: str):
        super().__init__(msg)
        self.msg = msg
        self.shape_name = shape_name

    def __str__(self):
        return f'ShapeError: error: "{self.msg}" with shape: {self.shape_name}'

@register_type
class NameError(EModeError):
    def __init__(self, msg: str, type: str, name: str):
        super().__init__(msg)
        self.msg = msg
        self.type = type
        self.name = name

    def __str__(self):
        return f'NameError: error: "{self.msg}" for type: {self.type} and name: {self.name}'

@register_type
class NotImplementedError(EModeError):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg

    def __str__(self):
        return f"NotImplementedError: {self.msg}"
