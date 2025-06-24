from typing import Any, Type, TypeVar, Optional, Union
from enum import Enum
from pydantic import BaseModel, ConfigDict, computed_field
import numpy as np


class TaggedModel(BaseModel):
    @computed_field(repr=False)
    @property
    def __data_type__(self) -> str:
        return self.__class__.__name__


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


T = TypeVar("T")

_TYPE_REGISTRY = {
    "MaterialSpec": MaterialSpec,
    "MaterialProperties": MaterialProperties,
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_fields = []

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
    def __init__(self, msg: str, license_type: LicenseType):
        super().__init__(msg, license_type)
        self.msg = msg
        self.license_type = license_type

    def __str__(self):
        return f'LicenseError: you are using license: {self.license_type!s}, error msg: "{self.msg}"'


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
