"""`MaterialSpec.reference` picks one of a material's index references (emode#807)."""

from __future__ import annotations

import json

import pytest

from emodeconnection import compat
from emodeconnection.types import (
    ArgumentError,
    EModeError,
    MaterialSpec,
    reconstruct,
    serialize,
    wire_dump,
)


def test_the_reference_defaults_to_none() -> None:
    assert MaterialSpec(material="LN_MgO").reference is None


def test_the_spec_round_trips_over_the_wire() -> None:
    spec = MaterialSpec(material="LN_MgO", reference="Zelmon 1997", theta=0.1)
    wire = json.loads(json.dumps(serialize({"m": spec})))["m"]
    assert wire["reference"] == "Zelmon 1997"
    assert reconstruct(wire) == spec


def test_a_spec_without_a_reference_sends_none_of_it() -> None:
    assert "reference" not in wire_dump(MaterialSpec(material="Si"))


def test_the_reference_names_the_emode_that_reads_it() -> None:
    assert compat.FIELD_SINCE[("MaterialSpec", "reference")] == "1.0.8"


def test_a_reference_is_refused_before_an_older_emode_gets_it() -> None:
    needs = compat.required_versions(
        {"material": MaterialSpec(material="LN_MgO", reference="Zelmon 1997")}
    )
    with pytest.raises(ArgumentError, match=r"MaterialSpec\.reference.*1\.0\.8"):
        compat.check_emode_version(needs, "1.0.7", "EM_shape")
    compat.check_emode_version(needs, "1.0.8", "EM_shape")


def test_a_spec_without_a_reference_needs_no_version() -> None:
    assert compat.required_versions({"material": MaterialSpec(material="Si")}) == {}
    assert issubclass(ArgumentError, EModeError)
