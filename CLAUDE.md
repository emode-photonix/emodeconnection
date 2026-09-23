# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**emodeconnection** is the client library for EMode's mode solver. It is the only piece
of the system that runs inside the customer's own Python environment: it is pip-installed
from PyPI, opens a socket to the EMode solver process, and exchanges length-prefixed JSON
messages with it.

The dependency runs one way — the solver depends on this library, never the reverse. This
library must never import from the solver. It has to stand on its own in the customer's
environment, where the solver's source is not present.

Python >= 3.10, kept deliberately low: customers run this in whatever environment they
already have.

This repository is **public**. The solver's own source is not, so nothing here should
name its internal files, classes, or paths — describe the wire contract from this side.

## Branching, versioning, and release

Read this before changing `pyproject.toml`.

### Feature work

- Branch off `dev`. Open the PR against `dev`. Never against `main`.
- **A feature branch never touches the version.** No exceptions, including branches that
  add a new wire type the server is waiting on.

### Version bumps

- A bump happens **only** when the developer explicitly asks for one, or when they make it
  themselves by hand on `dev` immediately before a release.
- It is its own commit on `dev`, made right before `dev` is merged into `main` — not
  earlier, and not as part of any feature PR.

### Release

1. On `dev`: bump the version in `pyproject.toml`. That is the only place the package
   version is declared — the version strings in `README.md` belong to the `emode-install`
   examples and refer to the EMode executable, not to this package.
2. PR `dev` into `main` and merge it.
3. Tag **from `main`**: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. `.github/workflows/publish.yml` fires on the `v*` tag, builds with `uv build`, and
   publishes to PyPI over Trusted Publishing (OIDC — there is no API token anywhere).

Order matters in two ways:

- The workflow **refuses to publish when the tag and the packaged version disagree**, so
  the bump has to be committed before the tag is created.
- A tag-push workflow runs the file **as it exists at the tagged commit**, so the
  `dev` → `main` merge has to land before the tag.

### Why the rule is this strict

The steady state is `PyPI == main == dev`, all at the last published version. The version
only moves during the short window between the bump commit and the publish.

If you find the three out of step, don't renumber backwards to close the gap. Release the
version `dev` is already at; that restores the invariant in one step and rewrites nothing.

## Commands

```bash
# Install for development (editable, into the environment that consumes it)
pip install -e .

# Build the distribution the release workflow builds
uv build
```

### Tests

This repo has **no test suite of its own**. The tests that exercise it live with the
solver, because they launch a real solver process and need a license checkout, which
rules out running them here or in this repo's CI.

So a change to anything that crosses the wire — a new or altered wire type, a change to
how one serializes — cannot be validated from inside this repository. Say so plainly in
the PR rather than implying it was tested, and have it exercised against the solver
before it is released.

### Formatting

**This repo is not `ruff`-formatted, and must not be reformatted wholesale.** Running
`ruff format` over a file here rewrites it end to end and buries a real change in dozens
of lines of unrelated churn. Match the surrounding style by hand instead — and note that
other EMode repositories do gate on `ruff`, so do not carry that habit into this one.

`pyproject.toml` configures `pyright`, deliberately loosened (see `[tool.pyright]`).

## Architecture

### The client/server contract

The socket protocol is not JSON-RPC — there are no request IDs and no method/params
envelope. This side of it is `emodeclient.py`: `recv()`, `obj_hook`, and `reconstruct`.

**Any type that must cross the wire as a rich object — not a bare numpy array or plain
dict — has to be defined *here*** and registered with `@register_type` in `types.py`, so
its `__data_type__` tag round-trips through `reconstruct`. The solver keeps its own richer
internal representations and converts them into these shared wire types before returning
a result, so the types in this repo are deliberately simplified, read-only views: they
carry the data a caller needs and none of the machinery that produced it. `smatrix.py` is
the pattern to copy.

That has a consequence worth stating outright: **this repo defines the public surface.** A
result type cannot be added from the solver side alone — it has to land here first, and a
released version has to contain it before anything can depend on it. Adding one means
extending an existing module rather than inventing a new ad hoc serialization path.

### Modules

- **`emodeconnection.py`** — the `EMode` class the user talks to; wraps every `EM_*` call.
- **`emodeclient.py`** — socket transport, `recv`, `obj_hook`, `reconstruct`.
- **`types.py`** — `@register_type`, `__data_type__`, and the user-facing exception
  hierarchy (`EModeError`, `ArgumentError`, `ShapeError`, `FileError`, `LicenseError`,
  `EPHKeyError`, `NameError`). All EMode errors reach the user as one of these.
- **`port.py`** — `Port`, `Bounds`, `JunctionOverride`, `make_angled_facet_port`.
- **`nonlinear.py`** — chi(2) process wire types (`SHGProcess`, `SFGProcess`,
  `DFGProcess`) and their solver settings.
- **`smatrix.py`** — the read-only `SMatrix` view returned to callers.
- **`geometry.py`** — geometry primitives (`Path`, `Pose`) for section construction.
- **`free_space.py`** — client-side free-space field projection diagnostics.
- **`eph_utils.py`, `file_utils.py`** — `.eph` simulation-file helpers.
- **`installer.py`** — the `emode-install` console script; downloads and installs the
  EMode executable.
- **`traceback_filter.py`** — hides internal wrapper/call/recv frames so a user's
  traceback points at their own script.
- **`constants.py`** — shared protocol constants.

`emodeconnection_matlab/` is the MATLAB unsupported counterpart, kept alongside the Python client.

## Style

- Wire types are **pydantic v2** models. Prefer `ConfigDict(frozen=True)` for anything
  used as a spec or a key — `Port` is frozen, and callers rely on that.
- Modern built-in type syntax (`x: int | None`, `list[str]`), never `typing.Optional`.
- Raise the `types.py` exceptions for anything the user should see; let internal errors
  propagate untouched.
- Keep docstrings proportional: a full one where the behavior is not obvious from the
  name, nothing at all on a trivial accessor.
