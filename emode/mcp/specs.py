"""The typed arguments of the tools. Their JSON schemas are what the agent sees."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Bound = tuple[float | None, float | None]

# Every boundary condition EMode takes: a pair of wall conditions, 0 (none), A
# (antisymmetric), S (symmetric) or P (periodic), the TE and TM folds, or 'auto'.
BoundaryCondition = Literal[
    "00", "0A", "0S", "0P", "A0", "AA", "AS", "AP",
    "S0", "SA", "SS", "SP", "P0", "PA", "PS", "PP",
    "TE", "TE-h", "TE-v", "TM", "TM-h", "TM-v", "auto",
]  # fmt: skip


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
    boundary_condition: BoundaryCondition | None = Field(
        None,
        description=(
            "Two characters, one per wall pair, each 0 (none), A (antisymmetric), "
            "S (symmetric) or P (periodic): '00' for no symmetry, or e.g. 'A0' or 'SA'. "
            "Or a polarization fold, 'TE', 'TE-h', 'TE-v', 'TM', 'TM-h' or 'TM-v', "
            "or 'auto' to let EMode choose the walls. 'TE'/'TM' fold about the window's "
            "center lines (y = window_height / 2, x = 0) where the structure is mirror "
            "symmetric about them, not about the window's edges."
        ),
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
    background_material: str | None = Field(
        None,
        description="Material filling the window where no shape is drawn, such as 'SiO2'.",
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
        description=(
            "Horizontal position, or [x, y] with 'auto' stacking on the layer below. The "
            "first shape stacks on the window's floor, and a shape on the window's edge "
            "continues past it, so give a core with no layer under it a y position."
        ),
    )
    etch_depth: float | None = Field(
        None,
        description="Etch depth in nm. The layer is etched outside its mask openings.",
    )
    mask: float | list[float] | None = Field(
        None,
        description="Width in nm of each mask opening an etched layer keeps, such as 500 or [800, 800].",
    )
    mask_offset: float | list[float] | None = Field(
        None,
        description="Center in nm of each mask opening, such as [-750, 750] for a pair.",
    )
    sidewall_angle: float | None = Field(None, description="Sidewall angle in degrees.")


class ParameterSpec(BaseModel):
    """One quantity the optimizer varies."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(
        description=(
            "Names the quantity, such as 'shape, core, width' (a shape of the current "
            "cross-section), 'shape, <profile>, <name>, <property>' (a shape of a labeled "
            "profile, such as 'shape, taper_end, core, mask'), 'section, coupler, length' "
            "or 'process, <section>, poling_period' (the poling period of a CW chi(2) "
            "process, also 'duty_cycle')."
        )
    )
    bounds: Bound | list[Bound] | None = Field(
        None,
        description=(
            "[min, max], either end null for no limit. For a key that takes a list, "
            "one [min, max] pair per entry, such as [[600, 1000], [600, 1000]], "
            "or one [min, max] for every entry."
        ),
    )
    x0: float | list[float] | None = Field(
        None,
        description=(
            "Starting value, by default the current one. A list only for a key that "
            "takes a list of values, 'poling, <section>, domain_lengths'."
        ),
    )

    @field_validator("bounds", mode="before")
    @classmethod
    def _bounds_are_pairs(cls, value: Any) -> Any:
        if (
            isinstance(value, list)
            and value
            and all(isinstance(v, list) for v in value)
            and any(len(v) != 2 for v in value)
        ):
            raise ValueError(
                f"bounds is [min, max], or one [min, max] pair per entry of x0, "
                f"such as [[600, 1000], [600, 1000]], not {value}"
            )
        return value


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
        "conversion",
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
    efficiency: bool | None = Field(
        None,
        description=(
            "For kind 'conversion' (the power a CW chi(2) process generates at port "
            "'out' and 'wavelength'): true, the default, for a fraction of the power "
            "put in, false for the power in W."
        ),
    )
    wavelengths: list[float] | None = Field(
        None,
        description=(
            "For kind 'conversion', in place of 'wavelength': a band of generated "
            "wavelengths in nm, reduced to one number by 'reduce'. A wavelength the "
            "declared device does not generate is reached by scaling every wavelength "
            "of the device by one factor."
        ),
    )
    reduce: Literal["soft_min", "mean"] | None = Field(
        None,
        description=(
            "With 'wavelengths': 'soft_min', a smooth worst case across the band, or "
            "'mean', the band's average."
        ),
    )
    temperature: float | None = Field(
        None,
        description=(
            "With reduce 'soft_min': how close it sits to the true minimum, as a "
            "fraction of the band's mean, default 0.05."
        ),
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
