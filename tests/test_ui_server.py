"""The session's local server: every route answers, and nothing else gets in.

The server runs on loopback with a per-session token. A request without the
token, with a wrong one, with a ``Host`` that is not this server's, or with
another site's ``Origin`` is answered 403 with an empty body, as is every
preflight. Static files carry their content types, the page its security
policy, the API ``no-store``, and no response carries the token.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from typing import Any, Dict, Optional, Tuple

import pytest

from emode.ui.figure import PlotOptions
from emode.ui.server import UIServer
from emode.ui.views import ViewRegistry

from plot_fixtures import xy_view, zx_view


def _server(**kwargs: Any) -> UIServer:
    registry = ViewRegistry()
    registry.add(
        zx_view(), {"plane": "z-x", "component": "Ex"}, PlotOptions(), (1200, 750)
    )
    fetched = []

    def fetch(view, arguments, pixels):
        fetched.append((dict(arguments), tuple(pixels)))
        return xy_view()

    server = UIServer(registry, fetch, **kwargs).start()
    server.fetched = fetched  # type: ignore[attr-defined]
    return server


@pytest.fixture
def server():
    s = _server()
    yield s
    s.stop()


def request(
    server: UIServer,
    method: str,
    path: str,
    token: Optional[str] = "header",
    host: Optional[str] = None,
    origin: Optional[str] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{server.port}"}
    if token == "header":
        headers["X-EMode-Token"] = server.token
    elif token is not None:
        headers["X-EMode-Token"] = token
    if origin is not None:
        headers["Origin"] = origin
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    response = conn.getresponse()
    payload = response.read()
    out = response.status, {k.lower(): v for k, v in response.getheaders()}, payload
    conn.close()
    return out


# ------------------------------------------------------------------ routes


def test_the_app_shell_needs_the_token_in_its_url_and_carries_the_policy(server):
    status, headers, body = request(
        server, "GET", f"/?token={server.token}", token=None
    )
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert b"static/app.js" in body
    assert request(server, "GET", "/", token=None)[0] == 403
    assert request(server, "GET", "/?token=wrong", token=None)[0] == 403


@pytest.mark.parametrize(
    "name, content_type",
    [
        ("app.js", "text/javascript"),
        ("emode.css", "text/css"),
        ("plotly-cartesian.min.js", "text/javascript"),
    ],
)
def test_the_static_files_carry_their_content_types(server, name, content_type):
    status, headers, body = request(server, "GET", f"/static/{name}", token=None)
    assert status == 200
    assert headers["content-type"].startswith(content_type)
    assert len(body) > 100


def test_no_path_outside_the_static_directory_is_served(server):
    for path in (
        "/static/../server.py",
        "/static/%2e%2e/server.py",
        "/static/missing.js",
        "/etc/passwd",
    ):
        assert request(server, "GET", path, token=None)[0] == 403


def test_the_views_route_lists_the_session(server):
    status, headers, body = request(server, "GET", "/api/views")
    assert status == 200
    assert headers["cache-control"] == "no-store"
    (view,) = json.loads(body)
    assert view["id"] == "p1" and view["kind"] == "plot"


def test_the_data_route_composes_from_the_registered_view_without_fetching(server):
    status, _, body = request(
        server,
        "POST",
        "/api/views/p1/data",
        body={"controls": {}, "pixels": [1200, 750]},
    )
    reply = json.loads(body)
    assert status == 200
    assert [t["type"] for t in reply["figure"]["data"]].count("heatmap") == 2
    assert reply["menu"]["selected"]["data"] == "Ex"
    assert server.fetched == []  # type: ignore[attr-defined]


def test_a_figure_control_redraws_without_fetching(server):
    body = {"controls": {"index_outline": False, "contours": True}}
    reply = json.loads(request(server, "POST", "/api/views/p1/data", body=body)[2])
    assert reply["controls"]["index_outline"] is False
    names = {t.get("name") for t in reply["figure"]["data"]}
    assert "outline" not in names and "port" not in names
    assert server.fetched == []  # type: ignore[attr-defined]


def test_a_data_control_fetches_the_view_again(server):
    body = {"controls": {"plane": "x-y"}, "pixels": [800, 600]}
    reply = json.loads(request(server, "POST", "/api/views/p1/data", body=body)[2])
    assert server.fetched == [({"plane": "x-y", "component": "Ex"}, (800, 600))]  # type: ignore[attr-defined]
    assert reply["figure"]["layout"]["xaxis"]["title"]["text"] == "x (nm)"
    assert server.registry.get("p1").data.plane == "x-y"


def test_a_much_wider_tab_fetches_a_z_panel_again(server):
    request(
        server,
        "POST",
        "/api/views/p1/data",
        body={"controls": {}, "pixels": [1300, 750]},
    )
    assert server.fetched == []  # type: ignore[attr-defined]
    request(
        server,
        "POST",
        "/api/views/p1/data",
        body={"controls": {}, "pixels": [2400, 1500]},
    )
    assert len(server.fetched) == 1  # type: ignore[attr-defined]


def test_after_the_session_a_data_control_is_refused_and_figure_controls_still_work(
    server,
):
    server.session_ended = True
    reply = json.loads(
        request(
            server, "POST", "/api/views/p1/data", body={"controls": {"plane": "x-y"}}
        )[2]
    )
    assert "session has ended" in reply["error"]
    reply = json.loads(
        request(
            server, "POST", "/api/views/p1/data", body={"controls": {"legend": False}}
        )[2]
    )
    assert "figure" in reply


def test_fetched_ping_closed_and_colormaps(server):
    assert request(server, "POST", "/api/views/p1/fetched", body={})[0] == 200
    assert server.registry.all_fetched()
    assert not server.tab_alive()
    assert request(server, "POST", "/api/ping", body={})[0] == 200
    assert server.tab_alive()
    assert request(server, "POST", "/api/closed", body={})[0] == 200
    assert server.tab_closed and not server.tab_alive()
    status, _, body = request(server, "GET", "/api/colormaps")
    assert status == 200 and set(json.loads(body)["kinds"]) == {
        "sequential",
        "diverging",
        "cyclic",
    }


def test_an_unknown_view_is_404(server):
    assert request(server, "POST", "/api/views/p9/data", body={})[0] == 404


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("token", [None, "", "wrong"])
@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/api/views"),
        ("GET", "/api/events"),
        ("POST", "/api/views/p1/data"),
        ("POST", "/api/ping"),
    ],
)
def test_every_api_route_refuses_a_missing_or_wrong_token(server, token, method, path):
    status, headers, body = request(
        server, method, path, token=token, body={} if method == "POST" else None
    )
    assert (status, body) == (403, b"")


@pytest.mark.parametrize(
    "host", ["evil.example:80", "127.0.0.1:1", "localhost", "attacker.com"]
)
def test_a_foreign_host_is_refused(server, host):
    assert request(server, "GET", "/api/views", host=host)[0] == 403
    assert (
        request(server, "GET", f"/?token={server.token}", token=None, host=host)[0]
        == 403
    )


def test_localhost_is_this_servers_host_too(server):
    assert (
        request(server, "GET", "/api/views", host=f"localhost:{server.port}")[0] == 200
    )


@pytest.mark.parametrize(
    "origin", ["https://evil.example", "http://127.0.0.1:1", "null"]
)
def test_a_foreign_origin_is_refused(server, origin):
    assert request(server, "GET", "/api/views", origin=origin)[0] == 403
    assert request(server, "POST", "/api/ping", origin=origin, body={})[0] == 403


def test_the_servers_own_origin_is_accepted(server):
    assert (
        request(server, "GET", "/api/views", origin=f"http://127.0.0.1:{server.port}")[
            0
        ]
        == 200
    )


def test_a_preflight_is_always_refused(server):
    status, headers, body = request(server, "OPTIONS", "/api/views")
    assert (status, body) == (403, b"")
    assert "access-control-allow-origin" not in headers


def test_no_response_carries_the_token(server):
    for method, path, token in [
        ("GET", f"/?token={server.token}", None),
        ("GET", "/static/app.js", None),
        ("GET", "/api/views", "header"),
        ("POST", "/api/views/p1/data", "header"),
        ("GET", "/api/colormaps", "header"),
    ]:
        _, headers, body = request(
            server, method, path, token=token, body={} if method == "POST" else None
        )
        assert server.token.encode() not in body
        assert server.token not in json.dumps(headers)


# ------------------------------------------------------------------ the event stream


def _read_event(sock: socket.socket, deadline: float) -> str:
    data = b""
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            continue
        if not chunk:
            break
        data += chunk
        if b"event: view-added" in data and data.endswith(b"\n\n"):
            break
    return data.decode()


def test_the_event_stream_delivers_a_new_view(server):
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=0.5)
    sock.sendall(
        (
            f"GET /api/events HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
            f"X-EMode-Token: {server.token}\r\n\r\n"
        ).encode()
    )
    time.sleep(0.2)
    assert server.tab_alive()  # connecting counts as a ping
    server.publish("view-added", {"id": "p2", "title": "x-y Ex"})
    text = _read_event(sock, time.monotonic() + 5)
    sock.close()
    assert "text/event-stream" in text
    assert 'event: view-added\ndata: {"id": "p2", "title": "x-y Ex"}' in text


# ------------------------------------------------------------------ lifecycle


def test_stop_frees_the_port_and_rotates_the_token():
    s = _server()
    port, token = s.port, s.token
    s.stop()
    assert s.token != token
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_two_sessions_get_two_ports_and_two_tokens():
    a, b = _server(), _server()
    try:
        assert a.port != b.port and a.token != b.token
        assert request(a, "GET", "/api/views", token=b.token)[0] == 403
    finally:
        a.stop()
        b.stop()


def test_EMODE_UI_PORT_is_honored(monkeypatch):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("EMODE_UI_PORT", str(port))
    s = _server()
    try:
        assert s.port == port
        assert s.url("p1") == f"http://127.0.0.1:{port}/?token={s.token}#/plots/p1"
    finally:
        s.stop()


def test_EMODE_UI_URL_names_the_address_and_its_host_is_accepted(monkeypatch):
    monkeypatch.setenv("EMODE_UI_URL", "https://hub.example/user/x/proxy/8123/")
    s = _server()
    try:
        assert s.url("p1").startswith("https://hub.example/user/x/proxy/8123/?token=")
        assert (
            request(
                s, "GET", "/api/views", host="hub.example", origin="https://hub.example"
            )[0]
            == 200
        )
    finally:
        s.stop()


def test_the_server_binds_loopback_only(server):
    assert server._httpd.server_address[0] == "127.0.0.1"


def test_serving_runs_on_a_daemon_thread(server):
    assert any(t.name == "emode-ui" and t.daemon for t in threading.enumerate())


# ------------------------------------------------------------------ keep-alive


@pytest.mark.parametrize(
    "path, token",
    [
        ("/api/views/p1/fetched", "header"),
        ("/api/ping", "header"),
        ("/api/closed", "header"),
        ("/api/views/p1/fetched", "wrong"),
    ],
)
def test_a_post_body_never_leaks_into_the_next_request_on_the_same_connection(
    server, path, token
):
    """emodeconnection#156: a route that answered without reading its body left
    ``{}`` on the kept-alive connection, and the next request read ``{}GET``."""
    view_id = json.loads(request(server, "GET", "/api/views")[2])[0]["id"]
    path = path.replace("p1", view_id)
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    host = {"Host": f"127.0.0.1:{server.port}"}
    first = {**host, "Content-Type": "application/json"}
    first["X-EMode-Token"] = server.token if token == "header" else token
    conn.request("POST", path, body=b"{}", headers=first)
    conn.getresponse().read()
    conn.request("GET", "/api/views", headers={**host, "X-EMode-Token": server.token})
    response = conn.getresponse()
    response.read()
    conn.close()
    assert response.status == 200


def test_each_request_on_a_kept_alive_connection_reads_its_own_body(server):
    """A POST after another POST on the same connection still gets its own controls."""
    view_id = json.loads(request(server, "GET", "/api/views")[2])[0]["id"]
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    headers = {
        "Host": f"127.0.0.1:{server.port}",
        "X-EMode-Token": server.token,
        "Content-Type": "application/json",
    }
    conn.request("POST", f"/api/views/{view_id}/fetched", body=b"{}", headers=headers)
    conn.getresponse().read()
    body = json.dumps({"controls": {"component": "Ey"}, "pixels": [1200, 750]}).encode()
    conn.request("POST", f"/api/views/{view_id}/data", body=body, headers=headers)
    response = conn.getresponse()
    reply = json.loads(response.read())
    conn.close()
    assert response.status == 200
    assert server.fetched[-1][0].get("component") == "Ey"
