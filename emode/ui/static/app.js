// The EMode app tab: the session's plots, each with the plot window's controls.
//
// The page draws what the client sends and decides nothing: every figure
// arrives as Plotly JSON from the client, which made it from EMode's data.
// A control change posts the new value, and the client answers with the new
// figure, asking EMode for new data only when the control changes the data.
"use strict";

const params = new URLSearchParams(location.search);
const TOKEN = params.get("token") || "";
const EMBED = params.get("embed") === "1";
const HEADERS = { "X-EMode-Token": TOKEN, "Content-Type": "application/json" };

// the eight settings of the plot window and the argument of plot() each sets
const SETTINGS = [
  ["Index outline", "index_outline"],
  ["Contours", "contours"],
  ["Text display", "text_display"],
  ["Mesh display", "mesh_display"],
  ["Window display", "window_display"],
  ["Forward only", "forward_only"],
  ["Physical space", "view"],
  ["Blend slices", "blend_slices"],
];

const state = { views: [], current: null, reply: null, ended: false, busyTimer: null };

function api(path, body) {
  const options = body === undefined
    ? { headers: HEADERS }
    : { method: "POST", headers: HEADERS, body: JSON.stringify(body) };
  return fetch("api/" + path, options).then((r) => {
    if (!r.ok) throw new Error("EMode answered " + r.status);
    return r.json();
  });
}

function setStatus(text, busy) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.classList.toggle("busy", !!busy);
}

function showMessage(text) {
  const el = document.getElementById("message");
  el.hidden = !text;
  el.textContent = text || "";
}

// ------------------------------------------------------------ the plots list

function routeId() {
  const m = location.hash.match(/^#\/plots\/([\w-]+)/);
  return m ? m[1] : null;
}

async function loadList() {
  state.views = await api("views");
  const ol = document.getElementById("views");
  ol.replaceChildren();
  for (const v of [...state.views].reverse()) {
    const li = document.createElement("li");
    li.dataset.id = v.id;
    li.textContent = v.title || v.id;
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = v.id + ", " + new Date(v.created * 1000).toLocaleTimeString();
    li.appendChild(when);
    li.addEventListener("click", () => { location.hash = "#/plots/" + v.id; });
    ol.appendChild(li);
  }
  markSelected();
}

function markSelected() {
  for (const li of document.querySelectorAll("#views li")) {
    li.classList.toggle("selected", li.dataset.id === state.current);
  }
}

// ------------------------------------------------------------ one view

function pixels() {
  const el = document.getElementById("figure");
  const r = window.devicePixelRatio || 1;
  return [Math.round(el.clientWidth * r), Math.round(el.clientHeight * r)];
}

async function request(id, controls) {
  clearTimeout(state.busyTimer);
  state.busyTimer = setTimeout(() => setStatus("EMode is busy", true), 1000);
  try {
    return await api("views/" + id + "/data", { controls: controls || {}, pixels: pixels() });
  } finally {
    clearTimeout(state.busyTimer);
    setStatus(state.ended ? "session ended" : "");
  }
}

async function show(id, controls) {
  if (!id) return;
  const first = state.current !== id;
  state.current = id;
  markSelected();
  let reply;
  try {
    reply = await request(id, controls);
  } catch (err) {
    showMessage(String(err));
    return;
  }
  if (reply.error) {
    showMessage(reply.error);
    if (!state.reply || state.reply.id !== id) return;
  } else {
    showMessage("");
    state.reply = reply;
    draw(reply);
  }
  if (first) {
    document.title = "EMode: " + (reply.title || id);
    api("views/" + id + "/fetched", {}).catch(() => {});
  }
}

function draw(reply) {
  const fig = document.getElementById("figure");
  Plotly.react(fig, reply.figure.data, reply.figure.layout, {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: { filename: "EM_plot_" + reply.id, scale: 3 },
  });
  fig.removeAllListeners && fig.removeAllListeners("plotly_clickannotation");
  fig.on && fig.on("plotly_clickannotation", (ev) => {
    const target = ev.annotation && ev.annotation.hovertext;
    if (target && target.startsWith("go to ")) change({ section: target.slice(6) });
  });
  buildControls(reply);
}

function change(controls) {
  if (state.current) show(state.current, controls);
}

// ------------------------------------------------------------ the controls

function select(label, values, selected, onchange) {
  const wrap = document.createElement("label");
  if (label) wrap.append(label + ":");
  const s = document.createElement("select");
  for (const v of values) {
    const o = document.createElement("option");
    o.value = o.textContent = v;
    if (String(v) === String(selected)) o.selected = true;
    s.appendChild(o);
  }
  s.addEventListener("change", () => onchange(s.value));
  wrap.appendChild(s);
  return wrap;
}

function textbox(label, options, value, onchange) {
  const wrap = document.createElement("label");
  wrap.append(label + ":");
  const input = document.createElement("input");
  input.type = "text";
  input.size = 9;
  input.value = value || "";
  const listId = "list-" + label.replace(/\W/g, "");
  input.setAttribute("list", listId);
  const dl = document.createElement("datalist");
  dl.id = listId;
  for (const v of options || []) {
    const o = document.createElement("option");
    o.value = v;
    dl.appendChild(o);
  }
  input.addEventListener("change", () => onchange(input.value));
  wrap.append(input, dl);
  return wrap;
}

function buildControls(reply) {
  const box = document.getElementById("controls");
  box.replaceChildren();
  const m = reply.menu || {};
  const sel = m.selected || {};
  const c = reply.controls || {};
  if (m.data && m.data.length) box.append(select("Data", m.data, sel.data, (v) => change({ component: v })));
  if (m.components && m.components.length) {
    box.append(select("", m.components, sel.component, (v) => change({ plot_function: v })));
  }
  if (m.normalizations && m.normalizations.length) {
    box.append(select("Norm", m.normalizations, sel.normalization, (v) => change({ normalization: v })));
  }
  if (m.planes && m.planes.length) box.append(select("Slice", m.planes, sel.plane, (v) => change({ plane: v })));
  if (m.ports && m.ports.length) box.append(select("Port", m.ports, sel.port, (v) => change({ port: v })));
  if (m.sections && m.sections.length) {
    box.append(select("Section", m.sections, sel.section || "all", (v) => change({ section: v === "all" ? null : v })));
  }
  if (m.profiles && m.profiles.length) {
    box.append(select("Profile", m.profiles, sel.profile, (v) => change({ profile: v })));
  }
  if (m.modes && m.modes.length) box.append(select("Mode", m.modes, sel.mode, (v) => change({ mode: v })));
  if (m.slider) {
    box.append(textbox(m.slider.label.replace(":", ""), m.slider.options, m.slider.value,
      (v) => change({ slice_position: v })));
  }
  if (m.z_resolutions && m.z_resolutions.length) {
    box.append(textbox("Z-res (nm)", m.z_resolutions, sel.z_resolution || "Auto",
      (v) => change({ resolution: v === "Auto" || v === "" ? "auto" : Number(v) })));
  }
  if (m.settings && Object.keys(m.settings).length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Settings";
    const list = document.createElement("div");
    const active = new Set(m.active_settings || []);
    for (const [name, arg] of SETTINGS) {
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      const figureValue = c[arg];
      cb.checked = arg in c && typeof figureValue === "boolean" ? figureValue : !!m.settings[name];
      if (arg === "view") cb.checked = (c.view || "layout") === "layout" && !!m.settings[name];
      cb.disabled = !active.has(name);
      cb.addEventListener("change", () => {
        const value = arg === "view" ? (cb.checked ? "layout" : "simulation") : cb.checked;
        change({ [arg]: value });
      });
      label.append(cb, " " + name);
      list.appendChild(label);
    }
    details.append(summary, list);
    box.append(details);
  }
}

// ------------------------------------------------------------ keys: the section zoom

document.addEventListener("keydown", (ev) => {
  if (!state.reply || ev.target.closest("input, select")) return;
  const arrows = (state.reply.figure.layout.annotations || []).filter((a) => a.name === "arrow");
  if (ev.key === "Escape") return change({ section: null });
  const dir = { ArrowLeft: "◀", ArrowRight: "▶", ArrowUp: "▲", ArrowDown: "▼" }[ev.key];
  const hit = dir && arrows.find((a) => a.text === dir);
  if (hit) {
    ev.preventDefault();
    change({ section: hit.hovertext.slice(6) });
  }
});

// ------------------------------------------------------------ the session's events

async function listen() {
  try {
    const r = await fetch("api/events", { headers: { "X-EMode-Token": TOKEN } });
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        let event = "message";
        let data = "";
        for (const line of chunk.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (data) onEvent(event, JSON.parse(data));
      }
    }
  } catch (err) {
    // the stream ends when the session does
  }
  sessionEnded();
}

async function onEvent(event, data) {
  if (event === "view-added") {
    await loadList();
    if (!EMBED) location.hash = "#/plots/" + data.id;
  } else if (event === "session-closed") {
    sessionEnded();
  }
}

function sessionEnded() {
  if (state.ended) return;
  state.ended = true;
  setStatus("session ended");
}

// ------------------------------------------------------------ start

function route() {
  const id = routeId() || (state.views.length ? state.views[state.views.length - 1].id : null);
  if (id && id !== state.current) show(id);
}

window.addEventListener("hashchange", route);
window.addEventListener("pagehide", () => {
  fetch("api/closed", { method: "POST", headers: HEADERS, body: "{}", keepalive: true }).catch(() => {});
});

document.addEventListener("DOMContentLoaded", async () => {
  if (EMBED) document.body.classList.add("embed");
  await loadList().catch((err) => showMessage(String(err)));
  route();
  listen();
  setInterval(() => {
    if (!state.ended) api("ping", {}).catch(sessionEnded);
  }, 2000);
  api("ping", {}).catch(() => {});
});
