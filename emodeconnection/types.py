from typing import Any, Type, TypeVar, Optional, Union, get_origin, get_args
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
        if len(data) == 1:
            return serialize(data[0])
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
    if origin is Union:  # Optional[T] is just Union[T, None]
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

class LicenseType(Enum):
    _2D = "2d"
    _3D = "3d"
    default = "default"

    def to_dict(self):
        return {"__type__": "LicenseType", "value": self.value}

    def __str__(self):
        return self.value

TensorType = Union[float, list[float], list[list[float]]]
DTensorType = Union[list[list[float]]]

class MaterialProperties(TaggedModel):
    n: Optional[TensorType] = None
    eps: Optional[TensorType] = None
    mu: Optional[TensorType] = None
    d: Optional[DTensorType] = None

    # this is necessary to support np.ndarrays here...
    model_config = ConfigDict(arbitrary_types_allowed=True)

class MaterialSpec(TaggedModel):
    material: Union[str, MaterialProperties]
    theta: Optional[float] = None
    phi: Optional[float] = None
    x: Optional[float] = None
    loss: Optional[float] = None  # dB/m

class Grid(TaggedModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: np.ndarray
    y: np.ndarray

    is_pml: bool = False
    is_expanded: bool = False
    is_bc: bool = False

class Field(TaggedModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    field: np.ndarray
    field_names: list[str]
    num_modes: int
    grid: Grid

T = TypeVar("T")

_TYPE_REGISTRY = {
    "MaterialSpec": MaterialSpec,
    "MaterialProperties": MaterialProperties,
    "Field": Field,
    "Grid": Grid,
}

def register_type(cls: Type[T]) -> Type[T]:
    """Class decorator to register an Exception subclass."""
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
    def __init__(self, msg: str, license_type: LicenseType | None):
        super().__init__(msg, license_type)
        self.msg = msg
        self.license_type = license_type

    def __str__(self):
        if isinstance(self.license_type, LicenseType):
            lt = self.license_type['value'].lower()  # type: ignore
        else:
            lt = str(self.license_type)  # guard against None
        if lt == '3d':
            emLicense = 'EMode3D'
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
