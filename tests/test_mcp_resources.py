"""MCP resources (the docs pages and SKILL.md), tidy-ups and packaging files."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.client import Client
from test_mcp import FakeEMode, _reset_fake, session  # noqa: F401

from emodeconnection.mcp.docs import html_to_markdown
from emodeconnection.mcp.server import create_server

ROOT = Path(__file__).parent.parent
SKILL = ROOT / "emodeconnection" / "mcp" / "SKILL.md"


def with_client(fn: Any, **server_kw: Any) -> Any:
    server = create_server(client_factory=FakeEMode, **server_kw)

    async def go() -> Any:
        async with Client(server) as client:
            return await fn(client)

    return asyncio.run(go())


# tidy-ups


def test_tool_descriptions_are_dedented() -> None:
    tools = with_client(lambda c: c.list_tools()).tools
    for tool in tools:
        assert tool.description
        lines = tool.description.splitlines()
        assert not any(line.startswith(" ") for line in lines), tool.name


def test_shutdown_deletes_the_array_files_it_wrote(tmp_path: Path) -> None:
    FakeEMode.returns["get"] = lambda key: np.zeros(5000)
    result = session(
        ("emode_open", {}), ("emode_get", {"key": "Ex"}), output_dir=tmp_path
    )[1]
    written = result.structured_content["files"]
    assert written
    assert not any(Path(p).exists() for p in written)
    assert list(tmp_path.glob("*.npy")) == []


def test_shutdown_removes_its_own_temporary_directory() -> None:
    FakeEMode.returns["get"] = lambda key: np.zeros(5000)
    result = session(("emode_open", {}), ("emode_get", {"key": "Ex"}))[1]
    (path,) = result.structured_content["files"]
    assert not Path(path).parent.exists()


def test_shutdown_keeps_plots(tmp_path: Path) -> None:
    def plot(**kwargs: Any) -> Any:
        from emodeconnection import PlotImage

        Path(kwargs["file_name"]).write_bytes(b"\x89PNG")
        return PlotImage(file=kwargs["file_name"])

    FakeEMode.returns["plot"] = plot
    session(("emode_open", {}), ("emode_plot", {}), output_dir=tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 1


# resources


def test_the_skill_is_served_as_a_resource() -> None:
    async def go(client: Client) -> Any:
        listed = [str(r.uri) for r in (await client.list_resources()).resources]
        body = await client.read_resource("emode://skill")
        return listed, body.contents[0].text

    listed, text = with_client(go)
    assert "emode://skill" in listed
    assert text == SKILL.read_text()


def test_the_skill_says_how_to_use_emode_well() -> None:
    text = SKILL.read_text()
    assert text.startswith("---\nname: emode\n")
    assert "description:" in text.split("---")[1]
    for phrase in ("converge", "interpolated", "passiv"):
        assert phrase in text.lower(), phrase
    assert "—" not in text


def test_a_docs_page_comes_from_the_installed_docs(tmp_path: Path) -> None:
    (tmp_path / "emodeguide").mkdir()
    (tmp_path / "emodeguide" / "setup.md").write_text("# Setup\n\nInstall it.\n")

    async def go(client: Client) -> str:
        return (
            (await client.read_resource("emode://docs/emodeguide/setup"))
            .contents[0]
            .text
        )

    assert (
        with_client(go, docs_dir=tmp_path, docs_fetch=_no_network)
        == "# Setup\n\nInstall it.\n"
    )


def _no_network(url: str) -> str:
    raise AssertionError(f"fetched {url}")


def test_a_page_not_installed_is_fetched_from_the_docs_site_as_markdown(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def fetch(url: str) -> str:
        seen.append(url)
        return (
            "<html><body><h1>Ring</h1><p>A <code>ring</code> design.</p></body></html>"
        )

    async def go(client: Client) -> str:
        return (
            (await client.read_resource("emode://docs/examples/Ring_Design"))
            .contents[0]
            .text
        )

    text = with_client(go, docs_dir=tmp_path, docs_fetch=fetch)
    assert seen == ["https://docs.emodephotonix.com/examples/Ring_Design.html"]
    assert text.startswith("# Ring")
    assert "A `ring` design." in text


def test_docs_pages_cannot_leave_the_docs_directory(tmp_path: Path) -> None:
    secret = tmp_path.parent / "secret.md"
    secret.write_text("secret")

    async def go(client: Client) -> Any:
        return await client.read_resource("emode://docs/..%2Fsecret")

    with pytest.raises(Exception):  # noqa: B017 - the SDK wraps it in an ExceptionGroup
        with_client(go, docs_dir=tmp_path, docs_fetch=_no_network)


def test_a_missing_page_is_an_error_that_names_it(tmp_path: Path) -> None:
    def fetch(url: str) -> str:
        raise OSError("404")

    async def go(client: Client) -> str:
        try:
            await client.read_resource("emode://docs/nope")
        except Exception as err:  # noqa: BLE001 - the error type is the SDK's
            return str(err)
        return ""

    assert "nope" in with_client(go, docs_dir=tmp_path, docs_fetch=fetch)


def test_html_to_markdown_keeps_structure_and_drops_scripts() -> None:
    html = (
        "<html><head><title>x</title><script>bad()</script></head><body>"
        "<h2>Mode solving</h2><p>Use <code>em.FDM()</code>.</p>"
        "<ul><li>one</li><li>two</li></ul><pre>em.close()</pre></body></html>"
    )
    md = html_to_markdown(html)
    assert "## Mode solving" in md
    assert "Use `em.FDM()`." in md
    assert "- one\n- two" in md
    assert "```\nem.close()\n```" in md
    assert "bad()" not in md


# packaging


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_the_claude_code_plugin_manifest_runs_emode_mcp() -> None:
    plugin = ROOT / "packaging" / "claude-plugin"
    manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "emode"
    assert manifest["description"]
    servers = json.loads((plugin / ".mcp.json").read_text())["mcpServers"]
    assert "emode-mcp" in json.dumps(servers["emode"])
    assert "emodeconnection[mcp]" in json.dumps(servers["emode"])
    assert (plugin / "skills" / "emode" / "SKILL.md").read_text() == SKILL.read_text()


def test_the_desktop_extension_manifest_is_complete() -> None:
    manifest = json.loads((ROOT / "packaging" / "mcpb" / "manifest.json").read_text())
    for key in (
        "manifest_version",
        "name",
        "display_name",
        "version",
        "description",
        "author",
        "server",
    ):
        assert key in manifest, key
    server = manifest["server"]
    assert server["type"] == "python"
    assert (ROOT / "packaging" / "mcpb" / server["entry_point"]).is_file()


def test_building_the_extension_stamps_the_package_version(tmp_path: Path) -> None:
    build = _load(ROOT / "packaging" / "build_mcpb.py")
    out = build.build(tmp_path / "emode.mcpb", install_dependencies=False)
    version = next(
        line.split('"')[1]
        for line in (ROOT / "pyproject.toml").read_text().splitlines()
        if line.startswith("version")
    )
    with zipfile.ZipFile(out) as bundle:
        names = bundle.namelist()
        manifest = json.loads(bundle.read("manifest.json"))
    assert manifest["version"] == version
    assert "server/main.py" in names
    assert "server/lib/emodeconnection/mcp/server.py" in names
    assert "server/lib/emodeconnection/mcp/SKILL.md" in names
