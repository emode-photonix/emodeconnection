"""The typed arguments of the tools. Their JSON schemas are what the agent sees."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SettingsSpec(BaseModel):
    """Solver settings, passed to ``EMode.settings``. Unset fields keep EMode's defaults."""

    model_config = ConfigDict(extra="forbid")

    wavelength: float | None = Field(None, description="Wavelength in nm.")
    num_modes: int | None = Field(None, description="Number of modes to solve for.")
    x_resolution: float | None = Field(
        None, description="Grid cell size along x in nm."
    )
    y_resolution: float | None = Field(
        None, description="Grid cell size along y in nm."
    )
    window_width: float | None = Field(
        None, description="Simulation window width in nm."
    )
    window_height: float | None = Field(
        None, description="Simulation window height in nm."
    )
    boundary_condition: str | None = Field(
        None, description="'0' for none, or 'S'/'A' symmetry pairs, or 'TE'/'TM'."
    )
    field_to_solve: Literal["Et", "Ht"] | None = Field(
        None, description="Field to solve for."
    )
    max_effective_index: float | None = Field(
        None,
        description="Guess for the largest effective index. 0 uses the largest material index.",
    )
    tolerance: float | None = Field(None, description="Eigensolver tolerance.")
    bend_radius: float | None = Field(
        None, description="Bend radius in nm. 0 means straight."
    )
    pml_thickness: float | None = Field(
        None, description="PML thickness in nm on every side."
    )
    solver: Literal["default", "FDM", "FEM"] | None = Field(
        None, description="Which solver 'default' means."
    )


class ShapeSpec(BaseModel):
    """One layer or ridge of the cross-section, passed to ``EMode.shape``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Shape name, unique within the cross-section.")
    material: str = Field(
        description="A material name from EMode's database, such as 'Si' or 'SiO2'."
    )
    width: float | None = Field(
        None, description="Width in nm. Omit for a layer spanning the window."
    )
    height: float = Field(description="Height (thickness) in nm.")
    position: float | list[float | Literal["auto"]] | None = Field(
        None,
        description="Horizontal position, or [x, y] with 'auto' stacking on the layer below.",
    )
    etch_depth: float | None = Field(None, description="Etch depth in nm, for a rib.")
    sidewall_angle: float | None = Field(None, description="Sidewall angle in degrees.")


class ParameterSpec(BaseModel):
    """One quantity the optimizer varies."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(
        description="Names the quantity, such as 'shape, core, width' or 'section, coupler, length'."
    )
    bounds: tuple[float | None, float | None] | None = Field(
        None, description="[min, max], either end null for no limit."
    )
    x0: float | list[float] | None = Field(
        None, description="Starting value, by default the current one."
    )


class ObjectiveSpec(BaseModel):
    """A quantity to improve (or, in a constraint, to limit)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "S_power",
        "S_phase",
        "monitor_coupling",
        "effective_index",
        "group_index",
        "loss",
        "band_energy",
        "band_average",
        "parameter",
    ]
    key: str | None = Field(
        None, description="For kind 'parameter', the parameter's key."
    )
    out: int | str | None = Field(
        None, description="S-matrix output port, by name or mode index."
    )
    into: int | str | None = Field(
        None, description="S-matrix input port, by name or mode index."
    )
    wavelength: float | None = Field(
        None,
        description="Wavelength in nm, to pick a port of a multi-wavelength circuit.",
    )
    mode: int | None = Field(
        None, description="Mode index for modal quantities, default 0."
    )
    profile: str | None = Field(
        None, description="Profile for modal quantities, default the default profile."
    )
    monitor: str | None = None
    beam: dict[str, Any] | None = None
    band: tuple[float, float] | None = Field(
        None, description="Band in nm for pulse objectives."
    )
    goal: Literal["maximize", "minimize", "target"] | None = Field(
        None, description="Default 'maximize'."
    )
    target: float | None = Field(
        None, description="The value to reach when goal is 'target'."
    )
    weight: float | None = Field(
        None, description="Weight in a weighted sum of objectives."
    )


class ConstraintSpec(BaseModel):
    """A limit: ``quantity op value``."""

    model_config = ConfigDict(extra="forbid")

    quantity: str | ObjectiveSpec = Field(
        description="A parameter key, or an objective whose quantity is limited."
    )
    op: Literal["<=", ">=", "=="]
    value: float
