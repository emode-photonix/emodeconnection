"""Which EMode release first reads each wire type and field added since EMode 1.0.6.

A client and an EMode of different releases each rebuild the other's wire
types with their own classes. This client sends a wire type as the fields
that differ from their defaults (`types.wire_dump`), so an older EMode reads
any argument that leaves the newer options alone. An option an older EMode
cannot read at all is refused here, before anything is sent:
`required_versions` finds the versioned types and fields a call uses, and
`check_emode_version` compares them with the EMode's own version, which
`EMode` asks for once a session with ``version()``. An EMode older than
1.0.7 has no ``version()``, so all that is known of it is that it is older.

Every wire type and field added after EMode 1.0.6 has an entry below, and
a test holds the tables to that, against a record of what EMode 1.0.6 and
emodeconnection 1.1.22 knew. A new field or type gets the version of the
first EMode release that reads it.
"""

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from .types import ArgumentError, non_default_fields

# The first EMode that answers version(). One that does not is older.
FIRST_REPORTING_EMODE = "1.0.7"

# The first EMode release that reads each wire type added after EMode 1.0.6.
TYPE_SINCE: dict[str, str] = {
    "Confinement": "1.0.7",
    "Constraint": "1.0.7",
    "ConvergenceResult": "1.0.7",
    "ConvergenceStage": "1.0.7",
    "ConvergenceStep": "1.0.7",
    "Evaluation": "1.0.7",
    "ModeDict": "1.0.7",
    "Objective": "1.0.7",
    "OptimizationResult": "1.0.7",
    "Parameter": "1.0.7",
    "PlotImage": "1.0.7",
    "PulseFit": "1.0.7",
    "PulseMeasurement": "1.0.7",
    "RankReport": "1.0.7",
}

# The first EMode release that reads each field added after EMode 1.0.6.
FIELD_SINCE: dict[tuple[str, str], str] = {
    ("MaterialSpec", "reference"): "1.0.8",
    ("Poling", "domain_lengths"): "1.0.7",
    ("PulseResponse", "measured_fit"): "1.0.7",
    ("PulseResponse", "rank_report"): "1.0.7",
    ("PulseResponse", "window_gap"): "1.0.7",
    ("PulsedProcess", "harmonic_coupling"): "1.0.7",
    ("PulsedProcess", "max_harmonic_modes"): "1.0.7",
    ("PulsedProcess", "modal_search"): "1.0.7",
    ("PulsedSource", "measurement"): "1.0.7",
    ("SMatrix", "mode_power"): "1.0.7",
}


def parse_version(version: str) -> tuple[int, ...]:
    """The numeric release of a version string: ``'1.0.7.dev3'`` is ``(1, 0, 7)``."""
    numbers = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        numbers.append(int(digits))
        if digits != part:
            break
    return tuple(numbers)


def _walk(value: Any) -> Iterator[tuple[str, str]]:
    """Each versioned type and non-default field in `value`, as (what, version)."""
    if isinstance(value, BaseModel):
        name = type(value).__name__
        if name in TYPE_SINCE:
            yield name, TYPE_SINCE[name]
        for field, field_value in non_default_fields(value):
            since = FIELD_SINCE.get((name, field))
            if since is not None:
                yield f"{name}.{field}", since
            yield from _walk(field_value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def required_versions(arguments: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """The versioned wire types and fields each argument of a call uses.

    Keyed by argument name, each a list of (what, version), where `what` names
    a type (``'Parameter'``) or a field (``'PulsedProcess.max_harmonic_modes'``).
    Arguments that use none are left out, so an empty result needs no version.
    """
    needs = {}
    for argument, value in arguments.items():
        found = list(_walk(value))
        if found:
            needs[argument] = found
    return needs


def check_emode_version(
    needs: dict[str, list[tuple[str, str]]], emode_version: str | None, function: str
) -> None:
    """Raise `ArgumentError` for the first need the EMode cannot read.

    `emode_version` is None for an EMode older than `FIRST_REPORTING_EMODE`,
    which cannot be told apart further, so only needs from that release on are
    refused there.
    """
    name = function[3:] if function.startswith("EM_") else function
    for argument, found in needs.items():
        for what, since in found:
            if emode_version is None:
                too_old = parse_version(since) >= parse_version(FIRST_REPORTING_EMODE)
                this_emode = f"older than {FIRST_REPORTING_EMODE}"
            else:
                too_old = parse_version(emode_version) < parse_version(since)
                this_emode = emode_version
            if not too_old:
                continue
            if "." in what:
                remedy = f"or leave {what.split('.', 1)[1]} at its default"
            else:
                remedy = f"or call {name}() without a {what}"
            raise ArgumentError(
                f"{what} needs EMode {since} or newer, and this EMode is {this_emode}. "
                f"Update EMode, {remedy}.",
                name,
                argument,
            )
