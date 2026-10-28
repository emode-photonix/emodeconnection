"""The wire contract between versions of emodeconnection and of EMode (#86).

A client and a solver of different releases have to keep working together.
Each side rebuilds the other's wire types with its own copy of the classes,
so these tests pin down what each side may send and what it must accept:

- This client sends a wire type as the fields that differ from their
  defaults. An older EMode then reads it with its own, older class, which
  may refuse a field it does not know (several wire types are
  ``extra='forbid'``) or a default that changed.
- An option an older EMode cannot read is refused here, before anything is
  sent, with the EMode version it needs.
- What arrives here is read tolerantly: a field or a type this client does
  not know is dropped or left as a plain dict, never a failed reply. A user's
  own typo in a constructor still raises.

The older releases' side is recorded under ``fixtures/wire`` (no old
checkout is needed): the wire types each released EMode and emodeconnection
knows, field by field, the JSON released clients send, and the replies
released solvers write.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from emodeconnection import compat
from emodeconnection.emodeclient import EModeClient
from emodeconnection.emodeconnection import EMode
from emodeconnection.geometry import Arc, Line
from emodeconnection.geometry import Path as GeometryPath
from emodeconnection.monitor import Monitor
from emodeconnection.nonlinear import SHGProcess, Source
from emodeconnection.optimize import Parameter
from emodeconnection.port import Bounds, Port
from emodeconnection.pulse import Poling, PulsedProcess, PulsedSource, PulseResponse
from emodeconnection.smatrix import SMatrix
from emodeconnection.types import (
    _TYPE_REGISTRY,
    ArgumentError,
    EModeError,
    LicenseError,
    LicenseType,
    MaterialProperties,
    MaterialSpec,
    TaggedModel,
    reconstruct,
    serialize,
)

FIXTURES = Path(__file__).parent / "fixtures" / "wire"
WIRE_TYPES = json.loads((FIXTURES / "wire_types.json").read_text())
CLIENT_PAYLOADS = json.loads((FIXTURES / "client_payloads.json").read_text())
SERVER_REPLIES = json.loads((FIXTURES / "server_replies.json").read_text())

# Arguments that leave every newer option at its default, by the wire types
# they use. Each should reach an older EMode as fields it knows.
_DEFAULT_ARGUMENTS: dict[str, Callable[[], TaggedModel]] = {
    "MaterialSpec": lambda: MaterialSpec(material="Si", theta=0.1),
    "MaterialProperties": lambda: MaterialSpec(material=MaterialProperties(n=2.0)),
    "Bounds": lambda: Bounds(window_width=1000, window_height=800),
    "Port": lambda: Port(name="p", side="left"),
    "Path": lambda: GeometryPath(segments=[Line(length=10.0), Arc(radius=5.0, sweep_angle=1.0)]),
    "SHGProcess": lambda: SHGProcess(pump_wavelength=1550),
    "Source": lambda: Source(wavelength=1550),
    "PulsedSource": lambda: PulsedSource(center_wavelength=1550, duration=100, energy=1.0),
    "PulsedProcess()": lambda: PulsedProcess(),
    "PulsedProcess(max_rank=8)": lambda: PulsedProcess(max_rank=8),
    "PulsedProcess(dispersion_samples=16)": lambda: PulsedProcess(dispersion_samples=16),
    "PulsedProcess(max_harmonic_modes=3)": lambda: PulsedProcess(max_harmonic_modes=3),
    "Poling(period)": lambda: Poling(period=4000.0),
    "PulsedProcess(poling=Poling(period))": lambda: PulsedProcess(poling=Poling(period=4000.0)),
    "Monitor": lambda: Monitor(name="m", w=1000.0, y=200.0),
}


def _tagged(value: Any) -> Iterator[tuple[str, set[str]]]:
    """Every tagged dict in a decoded payload, as (tag, its field names)."""
    if isinstance(value, dict):
        if "__data_type__" in value:
            yield value["__data_type__"], set(value) - {"__data_type__"}
        for item in value.values():
            yield from _tagged(item)
    elif isinstance(value, list):
        for item in value:
            yield from _tagged(item)


def _sent(obj: Any) -> Any:
    """What this client puts on the wire for `obj` as a call's argument."""
    return json.loads(json.dumps(serialize({"function": "EM_x", "arg": obj})))["arg"]


# What this client sends


@pytest.mark.parametrize("server", ["emode 1.0.5", "emode 1.0.6"])
@pytest.mark.parametrize("name", list(_DEFAULT_ARGUMENTS))
def test_an_argument_left_at_its_defaults_carries_only_fields_an_older_emode_knows(
    server: str, name: str
) -> None:
    known = WIRE_TYPES[server]
    for tag, fields in _tagged(_sent(_DEFAULT_ARGUMENTS[name]())):
        if tag not in known:
            # The type itself is newer than this EMode (the pulsed solver
            # before 1.0.6), which no client can change.
            assert server == "emode 1.0.5", (server, tag)
            continue
        unknown = fields - set(known[tag]["fields"])
        assert not unknown, f"{server} would refuse {tag} fields {sorted(unknown)}"


def test_a_default_pulsed_process_is_sent_as_its_tag_alone() -> None:
    assert _sent(PulsedProcess()) == {"__data_type__": "PulsedProcess"}


def test_a_field_given_its_default_value_is_not_sent() -> None:
    assert _sent(PulsedProcess(max_harmonic_modes=3, modal_search="broadened")) == {
        "__data_type__": "PulsedProcess"
    }


def test_a_changed_field_is_sent() -> None:
    assert _sent(PulsedProcess(max_rank=8)) == {"__data_type__": "PulsedProcess", "max_rank": 8}


def test_defaults_are_left_out_of_nested_wire_types_too() -> None:
    sent = _sent(PulsedProcess(poling=Poling(period=4000.0)))
    assert sent == {"__data_type__": "PulsedProcess", "poling": {"__data_type__": "Poling", "period": 4000.0}}


def test_a_default_that_changed_is_not_sent() -> None:
    # solver_atol was 1e-9 before 1.0.7 and is None (the solver's choice) now.
    # An older EMode would refuse None for its float field.
    assert "solver_atol" not in _sent(SHGProcess(pump_wavelength=1550))
    assert "dispersion_samples" not in _sent(PulsedProcess())


def test_array_fields_are_compared_with_their_defaults_safely() -> None:
    with_power = SMatrix(port_sizes=(1, 1), data=np.eye(2, dtype=complex), mode_power=np.ones(2))
    assert "mode_power" in _sent(with_power)
    assert "mode_power" not in _sent(SMatrix(port_sizes=(1, 1), data=np.eye(2, dtype=complex)))


@pytest.mark.parametrize("name", list(_DEFAULT_ARGUMENTS))
def test_what_is_sent_rebuilds_the_same_object(name: str) -> None:
    obj = _DEFAULT_ARGUMENTS[name]()
    assert reconstruct(_sent(obj)) == obj


# Refusing an option an older EMode cannot read


class _FakeClient:
    """The socket client, answering each call with the next recorded reply."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.sent: list[dict] = []

    def send(self, data: dict) -> None:
        self.sent.append(json.loads(json.dumps(serialize(data))))

    def recv(self) -> Any:
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _session(*replies: Any) -> tuple[EMode, _FakeClient]:
    em = EMode.__new__(EMode)
    em._dsim = "emode"
    em._in_flight = False
    client = _FakeClient(list(replies))
    em._client = client  # type: ignore[assignment]
    em.in_ipython = lambda: False  # type: ignore[method-assign]
    return em, client


# What an EMode older than 1.0.7, which has no version(), answers.
_NO_VERSION = EModeError("'version' is not a supported EMode function")


def test_a_new_option_is_refused_before_it_reaches_an_emode_older_than_1_0_7() -> None:
    em, client = _session(_NO_VERSION)
    with pytest.raises(ArgumentError, match=r"max_harmonic_modes.*1\.0\.7") as raised:
        em.section(nonlinear=PulsedProcess(max_harmonic_modes=1))
    assert raised.value.argument == "nonlinear"
    assert [m["function"] for m in client.sent] == ["EM_version"]


def test_a_new_wire_type_is_refused_before_it_reaches_an_emode_older_than_1_0_7() -> None:
    em, client = _session(_NO_VERSION)
    with pytest.raises(ArgumentError, match=r"Parameter.*1\.0\.7"):
        em.optimize(parameters=[Parameter(key="width", bounds=(400, 600))])
    assert [m["function"] for m in client.sent] == ["EM_version"]


def test_a_new_option_reaches_an_emode_that_reads_it() -> None:
    em, client = _session({"emode": "1.0.7", "protocol": 1}, "done")
    assert em.section(nonlinear=PulsedProcess(max_harmonic_modes=1)) == "done"
    assert [m["function"] for m in client.sent] == ["EM_version", "EM_section"]
    assert client.sent[1]["nonlinear"]["max_harmonic_modes"] == 1


def test_defaults_reach_an_older_emode_without_asking_its_version() -> None:
    em, client = _session("done")
    assert em.section(nonlinear=PulsedProcess(poling=Poling(period=4000.0))) == "done"
    assert [m["function"] for m in client.sent] == ["EM_section"]


def test_the_emode_version_is_asked_once_a_session() -> None:
    em, client = _session({"emode": "1.0.7", "protocol": 1}, "a", "b")
    em.section(nonlinear=PulsedProcess(max_harmonic_modes=1))
    em.section(nonlinear=PulsedProcess(harmonic_coupling=0.2))
    assert [m["function"] for m in client.sent] == ["EM_version", "EM_section", "EM_section"]


def test_an_emode_that_reports_its_version_is_held_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(compat.FIELD_SINCE, ("PulsedProcess", "max_rank"), "1.0.8")
    em, client = _session({"emode": "1.0.7", "protocol": 1})
    with pytest.raises(ArgumentError, match=r"max_rank.*1\.0\.8.*1\.0\.7"):
        em.section(nonlinear=PulsedProcess(max_rank=8))
    assert [m["function"] for m in client.sent] == ["EM_version"]


def test_a_license_error_while_asking_the_version_reaches_the_caller() -> None:
    em, _ = _session(LicenseError("the session has ended", LicenseType.demo))
    with pytest.raises(LicenseError):
        em.section(nonlinear=PulsedProcess(max_harmonic_modes=1))


def test_versions_compare_numerically() -> None:
    assert compat.parse_version("1.0.10") > compat.parse_version("1.0.7")
    assert compat.parse_version("1.0.7") == compat.parse_version("1.0.7")
    assert compat.parse_version("1.0.7.dev3") == compat.parse_version("1.0.7")


# Every addition since the newest recorded release says which EMode reads it


def _baseline() -> dict[str, set[str]]:
    """The wire types (and their fields) EMode 1.0.6 and its client know."""
    known: dict[str, set[str]] = {}
    for release in ("emode 1.0.6", "emodeconnection 1.1.22"):
        for tag, info in WIRE_TYPES[release].items():
            known.setdefault(tag, set()).update(info["fields"])
    return known


def _wire_models() -> dict[str, type[BaseModel]]:
    return {
        name: cls
        for name, cls in _TYPE_REGISTRY.items()
        if isinstance(cls, type) and issubclass(cls, TaggedModel)
    }


def test_every_wire_type_and_field_newer_than_emode_1_0_6_names_the_emode_that_reads_it() -> None:
    known = _baseline()
    missing = []
    for name, cls in _wire_models().items():
        if name not in known:
            if name not in compat.TYPE_SINCE:
                missing.append(name)
            continue
        for field in cls.model_fields:
            if field not in known[name] and (name, field) not in compat.FIELD_SINCE:
                missing.append(f"{name}.{field}")
    assert not missing, f"add these to compat.TYPE_SINCE / FIELD_SINCE: {missing}"


def test_every_versioned_entry_names_a_real_wire_type_or_field() -> None:
    models = _wire_models()
    for name in compat.TYPE_SINCE:
        assert name in models, name
    for name, field in compat.FIELD_SINCE:
        assert field in models[name].model_fields, (name, field)


# What this client accepts


def _client(message: str) -> EModeClient:
    client = EModeClient.__new__(EModeClient)
    client._recv_msg = lambda: message.encode("utf-8")  # type: ignore[method-assign]
    return client


def _reply(payload: dict) -> Any:
    return _client(json.dumps({**payload, "__version__": 1})).recv()


def test_a_reply_field_this_client_does_not_know_is_dropped() -> None:
    received = _reply({"__data_type__": "PulsedProcess", "max_rank": 8, "future_option": 5})
    assert received == PulsedProcess(max_rank=8)


def test_a_typo_in_a_constructor_is_still_refused() -> None:
    with pytest.raises(ValidationError, match="max_rnak"):
        PulsedProcess(max_rnak=8)  # type: ignore[call-arg]


def test_an_unknown_type_inside_a_known_one_is_dropped_with_its_field() -> None:
    received = _reply(
        {
            "__data_type__": "PulsedProcess",
            "future_report": {"__data_type__": "FutureReport", "value": 1},
        }
    )
    assert received == PulsedProcess()


def test_an_unknown_type_arrives_as_a_plain_dict() -> None:
    assert _reply({"__data_type__": "FutureResult", "value": 1}) == {
        "__data_type__": "FutureResult",
        "value": 1,
    }


def test_a_reply_value_this_client_cannot_hold_arrives_as_a_plain_dict() -> None:
    # A newer EMode may accept a value this client's class refuses.
    received = _reply({"__data_type__": "PulsedProcess", "raman_mode": "future"})
    assert received == {"__data_type__": "PulsedProcess", "raman_mode": "future"}


def test_an_error_type_this_client_does_not_know_is_raised_as_an_emode_error() -> None:
    with pytest.raises(EModeError, match="the quota is used up"):
        _reply({"__data_type__": "QuotaError", "msg": "the quota is used up", "seats": 3})


def test_an_error_with_a_field_this_client_does_not_know_is_still_raised() -> None:
    with pytest.raises(ArgumentError) as raised:
        _reply(
            {
                "__data_type__": "ArgumentError",
                "msg": "must be positive",
                "function": "EM_init",
                "argument": "x_resolution",
                "hint": "try 10",
            }
        )
    assert raised.value.argument == "x_resolution"


def _released_replies() -> list[tuple[str, str, str]]:
    return [(server, name, raw) for server, replies in SERVER_REPLIES.items() for name, raw in replies.items()]


@pytest.mark.parametrize(
    "server, name, raw", _released_replies(), ids=[f"{s}-{n}" for s, n, _ in _released_replies()]
)
def test_a_released_emodes_reply_reconstructs_into_todays_types(server: str, name: str, raw: str) -> None:
    tag = json.loads(raw)["__data_type__"]
    if tag.endswith("Error"):
        with pytest.raises(_TYPE_REGISTRY[tag]):
            _client(raw).recv()
        return
    received = _client(raw).recv()
    assert type(received).__name__ == tag


def _released_payloads() -> list[tuple[str, str, str]]:
    return [(client, name, raw) for client, sent in CLIENT_PAYLOADS.items() for name, raw in sent.items()]


@pytest.mark.parametrize(
    "client, name, raw", _released_payloads(), ids=[f"{c}-{n}" for c, n, _ in _released_payloads()]
)
def test_a_released_clients_argument_reconstructs_into_todays_types(client: str, name: str, raw: str) -> None:
    sent = json.loads(raw)["arg"]
    received = reconstruct(sent)
    assert type(received).__name__ == sent["__data_type__"]
