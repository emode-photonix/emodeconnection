"""The tab's other views, the export panel's routes, the session page, and their calls.

``plot_S_matrix()`` and ``material_explorer()`` fetch their views from EMode and
register them in the session's plots list, ``FarField.show()`` and
``SMatrix.show()`` register data the client already holds, and the tab changes
what each shows. The export panel saves a view under the working directory
(``EM_figures/`` by default) or sends it back as a file, at a chosen format,
size, resolution and labels. The session page merges EMode's ``SessionInfo``
with this client's own rows. Every new route needs the token, this server's
``Host`` and no foreign ``Origin``.
"""
from __future__ import annotations

import builtins
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

import emode.ui.session as ui
from emode.plotting import MaterialView, SMatrixView
from emode.types import ArgumentError, EModeError
from emode.ui.figure import FarFieldPanel, PlotOptions
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from plot_fixtures import farfield, material_view, session_info, smatrix_view, zx_view
from plot_standin import VERSION_1_0_9, session
from test_ui_server import request

VERSION_GUI_M2 = {**VERSION_1_0_9, "views": ["plot", "smatrix", "material", "session"]}


@pytest.fixture
def pages(tmp_path, monkeypatch):
    """A server with one view of each kind, the stand-in fetch recording what it is asked."""
    monkeypatch.chdir(tmp_path)
    registry = ViewRegistry()
    registry.add(zx_view(), {"plane": "z-x"}, PlotOptions(), (1200, 750))
    registry.add(smatrix_view(), {}, PlotOptions(), (1200, 750))
    registry.add(material_view(), {"material": "Made"}, PlotOptions(), (1200, 750))
    registry.add(FarFieldPanel(farfield(screen=True)), {}, PlotOptions(), (1200, 750))
    fetched = []

    def fetch(view, arguments, pixels):
        fetched.append((view.kind, dict(arguments)))
        return material_view(requested=tuple(arguments.get("wavelengths") or (1000.0, 2000.0)))

    server = UIServer(registry, fetch, session_info=lambda: session_info().model_dump(mode="json")).start()
    server.fetched = fetched  # type: ignore[attr-defined]
    yield server
    server.stop()


def _json(reply) -> Dict[str, Any]:
    status, headers, body = reply
    assert status == 200, (status, body)
    return json.loads(body)


# ------------------------------------------------------------------ the plots list


def test_the_views_route_lists_every_kind_in_order_with_its_title(pages):
    views = _json(request(pages, "GET", "/api/views"))
    assert [(v["id"], v["kind"]) for v in views] == [
        ("p1", "plot"), ("x2", "smatrix"), ("m3", "material"), ("f4", "farfield")
    ]
    assert views[1]["title"] == "S-matrix in to out_a abs^2"
    assert views[2]["title"] == "material Made n_x"
    assert views[3]["title"] == "far field 1550 nm intensity"


def test_the_s_matrix_controls_redraw_without_asking_emode(pages):
    reply = _json(request(pages, "POST", "/api/views/x2/data", body={"controls": {"plot_function": "angle", "output": 2}}))
    assert reply["menu"]["selected"] == {"input": "in", "output": "out_b", "component": "angle"}
    assert reply["title"] == "S-matrix in to out_b angle"
    assert pages.fetched == []
    reply = _json(request(pages, "POST", "/api/views/x2/data", body={"controls": {"vmin": 0.0, "vmax": 0.5}}))
    heatmap = next(t for t in reply["figure"]["data"] if t["type"] == "heatmap")
    assert (heatmap["zmin"], heatmap["zmax"]) == (0.0, 0.5)


def test_the_material_component_is_the_clients_and_a_range_is_emodes(pages):
    reply = _json(request(pages, "POST", "/api/views/m3/data", body={"controls": {"component": "n_z"}}))
    assert reply["menu"]["selected"]["component"] == "n_z"
    assert pages.fetched == []
    reply = _json(
        request(pages, "POST", "/api/views/m3/data", body={"controls": {"wavelengths": [900.0, 2100.0], "step": 5.0}})
    )
    assert pages.fetched == [("material", {"material": "Made", "wavelengths": [900.0, 2100.0], "step": 5.0})]
    # the component chosen in the tab outlives the new data
    assert reply["menu"]["selected"]["component"] == "n_z"
    assert reply["extra"]["requested"] == [900.0, 2100.0]
    assert reply["extra"]["reference"].startswith("A. Author")


def test_the_far_field_switches_to_its_screen(pages):
    reply = _json(request(pages, "POST", "/api/views/f4/data", body={"controls": {"data": "screen"}}))
    assert reply["menu"]["data"] == ["intensity", "screen"]
    assert reply["menu"]["selected"] == {"data": "screen"}
    assert pages.fetched == []


# ------------------------------------------------------------------ the export panel


def _png_size(data: bytes) -> tuple:
    assert data.startswith(b"\x89PNG")
    return struct.unpack(">II", data[16:24])


def _png_dpi(data: bytes) -> float:
    at = data.index(b"pHYs")
    per_meter = struct.unpack(">I", data[at + 4 : at + 8])[0]
    return per_meter * 0.0254


def test_save_writes_a_png_at_the_posted_dpi_and_size_under_em_figures(pages, tmp_path):
    reply = _json(
        request(pages, "POST", "/api/views/x2/save", body={"file_name": "smat", "file_type": "png", "dpi": 200, "figure_size": [100, 60]})
    )
    path = Path(reply["path"])
    assert path == (tmp_path / "EM_figures" / "smat.png").resolve()
    data = path.read_bytes()
    assert _png_dpi(data) == pytest.approx(200, abs=1)
    width, height = _png_size(data)
    # the crop takes the margins off a 100 mm by 60 mm figure at 200 dpi (787 by 472 px)
    assert 400 < width <= 830 and 200 < height <= 520


def test_save_takes_a_relative_path_and_writes_pdf_and_svg(pages, tmp_path):
    pdf = Path(_json(request(pages, "POST", "/api/views/m3/save", body={"file_name": "out/mat", "file_type": "pdf"}))["path"])
    assert pdf == (tmp_path / "out" / "mat.pdf").resolve()
    assert pdf.read_bytes().startswith(b"%PDF")
    svg = Path(_json(request(pages, "POST", "/api/views/p1/save", body={"file_name": "field.svg", "file_type": "svg"}))["path"])
    assert svg.name == "field.svg" and b"<svg" in svg.read_bytes()[:400]


def test_the_labels_posted_with_a_save_are_in_the_file(pages):
    body = {
        "file_name": "labeled",
        "file_type": "svg",
        "title": "Splitter transmission",
        "xlabel": "In",
        "ylabel": "Out",
        "colorbar_label": "|S|²",
        "font_size": 9,
    }
    path = Path(_json(request(pages, "POST", "/api/views/x2/save", body=body))["path"])
    text = path.read_text()
    for label in ("Splitter transmission", "In", "Out"):
        assert label in text
    # the save's labels are for that file: the view keeps its own
    assert pages.registry.get("x2").options.title is None


@pytest.mark.parametrize("name", ["/tmp/escape", "../escape", "a/../../escape"])
def test_save_stays_in_the_working_directory(pages, name):
    reply = _json(request(pages, "POST", "/api/views/p1/save", body={"file_name": name, "file_type": "png"}))
    assert "working directory" in reply["error"]


def test_an_unknown_file_type_is_refused(pages):
    reply = _json(request(pages, "POST", "/api/views/p1/save", body={"file_name": "x", "file_type": "bmp"}))
    assert "file_type" in reply["error"]


def test_export_sends_the_file_back_to_download(pages):
    status, headers, body = request(pages, "POST", "/api/views/m3/export", body={"file_name": "mat", "file_type": "pdf"})
    assert status == 200 and body.startswith(b"%PDF")
    assert headers["content-type"] == "application/pdf"
    assert headers["content-disposition"] == 'attachment; filename="mat.pdf"'
    assert headers["cache-control"] == "no-store"


def test_without_matplotlib_a_save_names_the_install_and_the_tab_still_draws(pages, monkeypatch):
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    for name in [m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    reply = _json(request(pages, "POST", "/api/views/x2/save", body={"file_name": "s", "file_type": "png"}))
    assert "pip install matplotlib" in reply["error"]
    reply = _json(request(pages, "POST", "/api/views/x2/data", body={"controls": {}}))
    assert any(t["type"] == "heatmap" for t in reply["figure"]["data"])


# ------------------------------------------------------------------ the session page


def test_the_session_route_merges_emodes_rows_with_the_clients(pages):
    page = _json(request(pages, "GET", "/api/session"))
    assert page["emode"]["version"] == "1.0.9"
    assert page["emode"]["eph_bytes"] == 12345
    assert page["emode"]["license_type"] == "3d"
    client = page["client"]
    assert client["pid"] > 0 and client["python"] == sys.version.split()[0]
    assert client["ui_port"] == pages.port and client["views"] == 4
    assert client["rss_bytes"] > 0
    assert client["docs"].startswith("https://docs.emodephotonix.com/")


def test_without_psutil_the_clients_memory_row_says_to_install_it(pages, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    client = _json(request(pages, "GET", "/api/session"))["client"]
    assert "pip install psutil" in client["memory"]
    assert "rss_bytes" not in client


def test_after_the_session_ends_the_page_keeps_the_clients_rows(pages):
    pages.session_ended = True
    page = _json(request(pages, "GET", "/api/session"))
    assert page["emode"] is None and page["session_ended"] is True
    assert page["client"]["pid"] > 0


# ------------------------------------------------------------------ refusals on the new routes


NEW_ROUTES = [
    ("GET", "/api/session"),
    ("POST", "/api/views/x2/save"),
    ("POST", "/api/views/x2/export"),
]


@pytest.mark.parametrize("token", [None, "", "wrong"])
@pytest.mark.parametrize("method, path", NEW_ROUTES)
def test_the_new_routes_refuse_a_missing_or_wrong_token(pages, token, method, path):
    reply = request(pages, method, path, token=token, body={"file_name": "x"} if method == "POST" else None)
    assert (reply[0], reply[2]) == (403, b"")


@pytest.mark.parametrize("method, path", NEW_ROUTES)
def test_the_new_routes_refuse_a_foreign_host_or_origin(pages, method, path, tmp_path):
    body = {"file_name": "x"} if method == "POST" else None
    assert request(pages, method, path, host="evil.example:80", body=body)[0] == 403
    assert request(pages, method, path, origin="https://evil.example", body=body)[0] == 403
    assert not (tmp_path / "EM_figures").exists()


def test_no_new_route_carries_the_token(pages):
    for method, path in NEW_ROUTES:
        body = {"file_name": "x", "file_type": "svg"} if method == "POST" else None
        assert pages.token.encode() not in request(pages, method, path, body=body)[2]


# ------------------------------------------------------------------ the calls


def _pages_session(**answers):
    em, client = session(version=VERSION_GUI_M2)
    client.answers.update(
        {"EM_plot_S_matrix": lambda request: smatrix_view(), "EM_material_explorer": lambda request: material_view()}
    )
    client.answers.update(answers)
    return em, client


@pytest.fixture
def quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("EMODE_NONINTERACTIVE", "1")
    monkeypatch.delenv("EMODE_HEADLESS", raising=False)
    monkeypatch.chdir(tmp_path)
    if sys.platform.startswith("linux"):
        monkeypatch.setenv("DISPLAY", ":0")


def test_plot_s_matrix_fetches_once_and_registers_a_view(quiet):
    em, client = _pages_session()
    handle = em.plot_S_matrix(input=0, output="out_b", plot_function="angle", title="T")
    (call,) = client.calls("EM_plot_S_matrix")
    assert call["input"] == 0 and call["output"] == "out_b" and call["plot_function"] == "angle"
    assert "title" not in call and "pixels" in call
    assert handle.id == "x1" and isinstance(handle.view, SMatrixView)
    assert handle.figure.title == "T"


def test_material_explorer_keeps_the_component_in_the_client(quiet):
    em, client = _pages_session()
    handle = em.material_explorer(material="Made", wavelengths=(900.0, 2100.0), component="n_z")
    (call,) = client.calls("EM_material_explorer")
    assert call["material"] == "Made" and list(call["wavelengths"]) == [900.0, 2100.0]
    assert "component" not in call
    assert isinstance(handle.view, MaterialView)
    assert handle.view.menu.selected["component"] == "n_z"


@pytest.mark.parametrize("call", ["plot_S_matrix", "material_explorer"])
def test_an_emode_without_the_view_is_refused_before_sending(quiet, call):
    em, client = session(version=VERSION_1_0_9)
    with pytest.raises(EModeError, match="needs EMode 1.0.9 or newer"):
        getattr(em, call)()
    assert client.calls("EM_plot_S_matrix") == client.calls("EM_material_explorer") == []


def test_an_unknown_argument_is_refused_naming_itself(quiet):
    em, _ = _pages_session()
    with pytest.raises(ArgumentError, match="window"):
        em.plot_S_matrix(window="tk")


def test_a_headless_s_matrix_is_a_plot_image(monkeypatch, tmp_path):
    monkeypatch.setenv("EMODE_HEADLESS", "1")
    monkeypatch.chdir(tmp_path)
    em, _ = _pages_session()
    image = em.plot_S_matrix(file_name="S10")
    assert image.file == str((tmp_path / "EM_figures" / "S10.png").resolve())


def test_farfield_and_smatrix_show_register_without_asking_emode(quiet):
    em, client = _pages_session()
    ui.register_session(em)
    ff = farfield(screen=True).show(data="screen")
    sm = smatrix_view().smatrix.show(output=2)
    assert ff.id == "f1" and sm.id == "x2"
    assert ff.view.panel == "screen"
    assert sm.view.menu.selected["output"] == "out_b"
    assert [c["function"] for c in client.sent] == []


def test_show_without_a_session_draws_a_file_or_says_how(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui, "_SESSIONS", [])
    image = farfield().show(file_name="ff")
    assert Path(image.file).read_bytes().startswith(b"\x89PNG")
    monkeypatch.delenv("EMODE_HEADLESS", raising=False)
    if not sys.platform.startswith("linux"):
        with pytest.raises(EModeError, match="EMode session"):
            farfield().show()


def test_the_session_info_route_asks_emode(quiet):
    em, client = _pages_session(EM_session_info=lambda request: session_info())
    em.plot_S_matrix()
    server = ui.ensure_server(em)
    try:
        page = server.session_page()
        assert page["emode"]["account"] == "user@example.com"
        assert client.calls("EM_session_info")
    finally:
        server.stop()


def test_the_docs_link_carries_the_running_version(pages):
    from emode.types import client_version

    client = _json(request(pages, "GET", "/api/session"))["client"]
    assert client["docs"] == f"{ui.DOCS_URL}?version=1.0.9"
    assert client["version"] == client_version()
