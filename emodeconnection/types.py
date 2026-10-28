from collections.abc import Iterator, Mapping
from typing import Any, Type, TypeVar, Optional, Union, get_origin, get_args
import inspect
import sys
from logging import getLogger

if sys.version_info >= (3, 10):
    from types import UnionType
else:
    UnionType = None

from enum import Enum
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
import math
import numpy as np
from pydantic.fields import FieldInfo

logger = getLogger(__name__)


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
        return serialize(wire_dump(data))
    if type(data).__module__ == np.__name__:
        # final catchall
        data = np.squeeze(data).tolist()
    else:
        return data

def _is_default(value: Any, info: FieldInfo) -> bool:
    """Whether a field holds its default, compared without raising on arrays."""
    if info.is_required():
        return False
    default = info.get_default(call_default_factory=True)
    if value is default:
        return True
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return (
            isinstance(value, np.ndarray)
            and isinstance(default, np.ndarray)
            and value.shape == default.shape
            and value.dtype == default.dtype
            and bool(np.array_equal(value, default))
        )
    try:
        same = value == default
    except Exception:
        return False
    return same is True or (isinstance(same, np.bool_) and bool(same))


def non_default_fields(model: BaseModel) -> Iterator[tuple[str, Any]]:
    """The fields of `model` that differ from their defaults, as (name, value)."""
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        if not _is_default(value, info):
            yield name, value


def _prune_defaults(value: Any, dumped: Any) -> None:
    """Remove from `dumped`, in place, every field `value` holds at its default."""
    if isinstance(value, BaseModel):
        if not isinstance(dumped, dict):
            return
        for name, info in type(value).model_fields.items():
            if name not in dumped:
                continue  # a custom serializer laid the model out differently
            field_value = getattr(value, name)
            if _is_default(field_value, info):
                del dumped[name]
            else:
                _prune_defaults(field_value, dumped[name])
    elif isinstance(value, (list, tuple)) and isinstance(dumped, (list, tuple)):
        if len(value) == len(dumped):
            for item, dumped_item in zip(value, dumped):
                _prune_defaults(item, dumped_item)
    elif isinstance(value, dict) and isinstance(dumped, dict):
        for key, item in value.items():
            if key in dumped:
                _prune_defaults(item, dumped[key])


def wire_dump(model: BaseModel) -> dict:
    """`model` as it crosses the wire: its tag and the fields that differ from their defaults.

    A field left at its default is not sent, at any depth. The other side
    fills it in from its own class. That is what lets an older EMode or
    emodeconnection read a wire type that has gained fields since (several
    wire types refuse fields they do not know), and what keeps a default that
    changed between versions (None for "the solver's choice") from reaching a
    side that cannot hold it.
    """
    dumped = model.model_dump()
    _prune_defaults(model, dumped)
    return dumped


def from_wire(cls: type, fields: dict) -> Any:
    """Build `cls` from what the other side sent, ignoring what this version does not know.

    A newer sender may add fields. A type that refuses unknown fields
    (``extra='forbid'``) does so to catch a user's typo, which still raises
    when the type is built directly, and only here are unknown fields dropped.
    An error type takes the arguments its constructor knows, with None for any
    the sender left out, and keeps the rest as attributes.
    """
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return _error_from_wire(cls, fields)
    model_fields = getattr(cls, "model_fields", None)
    if model_fields is not None and cls.model_config.get("extra") == "forbid":  # type: ignore[attr-defined]
        unknown = [key for key in fields if key not in model_fields]
        if unknown:
            logger.debug(f"ignored the fields {unknown} of {cls.__name__}, which this version does not know")
            fields = {key: value for key, value in fields.items() if key in model_fields}
    return cls(**fields)


def _error_from_wire(cls: type, fields: dict) -> BaseException:
    params = [
        p
        for p in list(inspect.signature(cls.__init__).parameters.values())[1:]
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    names = {p.name for p in params}
    kwargs = {key: value for key, value in fields.items() if key in names}
    for p in params:
        if p.name not in kwargs and p.default is p.empty:
            kwargs[p.name] = None
    error = cls(**kwargs)
    for key, value in fields.items():
        if key not in names and not key.startswith("_"):
            setattr(error, key, value)
    return error

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
        # A .mat file has no null, so None is written there as NaN. Read a NaN
        # back as None wherever the field allows None.
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
    # The time-limited session of the "Try EMode now" notebook. EMode grants
    # it to a demo account whichever type was asked for.
    demo = "demo"

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

@register_type
class Confinement(TaggedModel, Mapping):
    """A read-only mapping from shape name to the confinement factor of each mode.

    Returned by `confinement()`. Each shape name (or ``'vs'`` for a polygon
    given by ``vertices``) maps to a `dict` from mode index to that mode's
    confinement factor, so ``result['core'][0]`` is the fundamental mode's
    factor in the shape ``'core'``. It behaves like a `dict` keyed by shape
    name: index it, iterate it, call ``.keys()``, ``.values()``, ``.items()``
    or ``.get()``, or turn it into a plain `dict` with ``dict(result)``.

    It exists because JSON object keys are always strings. A plain dict keyed
    by mode index would arrive here keyed by ``'0'``. This type carries its
    mode indices through validation, which turns them back into integers.

    On the wire the shape names sit at the top level beside the type tag, so
    an emodeconnection that predates this type reads the reply as the plain
    dict it always did (keyed by ``'0'``, with an extra ``'__data_type__'``
    entry).
    """

    model_config = ConfigDict(frozen=True)
    factors: dict[str, dict[int, float]]

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, values: Any) -> Any:
        """Accept the flat wire form, ``{shape: {mode: factor}}``, as well as ``factors=``."""
        if not isinstance(values, Mapping):
            return values
        factors = values.get("factors")
        if (
            set(values) == {"factors"}
            and isinstance(factors, Mapping)
            and all(isinstance(modes, Mapping) for modes in factors.values())
        ):
            return values
        return {"factors": dict(values)}

    @model_serializer(mode="wrap")
    def _to_wire(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        dumped = handler(self)
        return {"__data_type__": dumped["__data_type__"], **dumped["factors"]}

    def __getitem__(self, shape: str) -> dict[int, float]:
        return self.factors[shape]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self.factors)

    def __len__(self) -> int:
        return len(self.factors)

    def __contains__(self, shape: object) -> bool:
        return shape in self.factors

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Confinement):
            return self.factors == other.factors
        if isinstance(other, Mapping):
            return self.factors == dict(other)
        return NotImplemented

    # Unhashable, like the dict it stands in for.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Confinement({self.factors!r})"

    def __str__(self) -> str:
        return repr(self)

@register_type
class ModeDict(TaggedModel, Mapping):
    """A read-only mapping from mode index to one result per mode.

    A shape returned by ``get_shape()`` carries one under
    ``metadata['confinement']``, the confinement factor of each mode in that
    shape, so ``shape['metadata']['confinement'][0]`` is the fundamental mode's.
    It behaves like a `dict` keyed by integer mode index: index it, iterate it,
    call ``.keys()``, ``.values()``, ``.items()`` or ``.get()``, or turn it into
    a plain `dict` with ``dict(result)``.

    It exists because JSON object keys are always strings. A plain dict keyed
    by mode index would arrive here keyed by ``'0'``. This type carries its
    mode indices through validation, which turns them back into integers.

    On the wire the mode indices sit at the top level beside the type tag, so
    an emodeconnection that predates this type reads the reply as the plain
    dict it always did (keyed by ``'0'``, with an extra ``'__data_type__'``
    entry).
    """

    model_config = ConfigDict(frozen=True)
    data: dict[int, Any]

    @model_validator(mode="before")
    @classmethod
    def _from_wire(cls, values: Any) -> Any:
        """Accept the flat wire form, ``{mode: value}``, as well as ``data=``."""
        if not isinstance(values, Mapping):
            return values
        if set(values) == {"data"} and isinstance(values["data"], Mapping):
            return values
        return {"data": dict(values)}

    @model_serializer(mode="wrap")
    def _to_wire(self, handler: SerializerFunctionWrapHandler) -> dict[Any, Any]:
        dumped = handler(self)
        return {"__data_type__": dumped["__data_type__"], **dumped["data"]}

    def __getitem__(self, mode: int) -> Any:
        return self.data[mode]

    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, mode: object) -> bool:
        return mode in self.data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModeDict):
            return self.data == other.data
        if isinstance(other, Mapping):
            return self.data == dict(other)
        return NotImplemented

    # Unhashable, like the dict it stands in for.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"ModeDict({self.data!r})"

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
    registered class, ignoring fields this version does not know (`from_wire`).
    Dicts with an unknown tag, or with values the class refuses, are left as
    plain dicts.
    Containers are walked bottom-up so nested tagged objects are rebuilt
    before their parent.
    """
    if isinstance(data, dict):
        data = {key: reconstruct(value) for key, value in data.items()}
        name = data.get("__data_type__")
        if name in _TYPE_REGISTRY:
            cls = _TYPE_REGISTRY[name]
            fields = {key: value for key, value in data.items() if key != "__data_type__"}
            if isinstance(cls, type) and issubclass(cls, BaseException):
                return from_wire(cls, fields)
            try:
                return from_wire(cls, fields)
            except (ValidationError, EModeError, TypeError, ValueError) as e:
                # A newer sender may hold a value this version's class refuses
                # (a new choice of a Literal, say). The plain dict keeps the
                # rest of the reply readable.
                logger.warning(
                    f"this emodeconnection cannot read the {name} it received, so it is "
                    f"returned as a plain dict. Upgrading emodeconnection may help. ({e})"
                )
                return data
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
        if self.license_type is LicenseType._3D:
            emLicense = 'EMode3D'
        elif self.license_type is LicenseType.demo:
            emLicense = 'EModeDEMO'
        else:
            emLicense = 'EMode2D'
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
