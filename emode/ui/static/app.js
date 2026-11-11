// The EMode app tab: the session's plots, each with its controls and the export
// panel, the session page, and a link to the documentation.
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
  const kind = reply.kind || "plot";
  if (kind === "smatrix") buildSMatrixControls(box, reply);
  else if (kind === "material") buildMaterialControls(box, reply);
  else if (kind === "farfield") buildFarFieldControls(box, reply);
  else if (kind === "sweep") buildSweepControls(box, reply);
  else buildPlotControls(box, reply);
  box.append(exportPanel(reply));
}

function numberOrNull(text) {
  const v = String(text).trim();
  return v === "" ? null : Number(v);
}

function buildSMatrixControls(box, reply) {
  const m = reply.menu || {};
  const sel = m.selected || {};
  const c = reply.controls || {};
  box.append(select("S from", m.ports || [], sel.input, (v) => change({ input: v })));
  box.append(select("to", m.ports || [], sel.output, (v) => change({ output: v })));
  box.append(select("", m.components || [], sel.component, (v) => change({ plot_function: v })));
  box.append(textbox("Scale min", [], c.vmin == null ? "" : c.vmin, (v) => change({ vmin: numberOrNull(v) })));
  box.append(textbox("max", [], c.vmax == null ? "" : c.vmax, (v) => change({ vmax: numberOrNull(v) })));
}

function buildMaterialControls(box, reply) {
  const m = reply.menu || {};
  const sel = m.selected || {};
  const x = reply.extra || {};
  box.append(select("Material", x.materials || [], sel.material, (v) => change({ material: v, wavelengths: null, step: null })));
  if (x.x_fraction !== null && x.x_fraction !== undefined) {
    box.append(textbox("x", [], x.x_fraction, (v) => change({ x: numberOrNull(v) })));
  }
  if ((m.components || []).length > 1) {
    box.append(select("", m.components, sel.component, (v) => change({ component: v })));
  }
  const req = x.requested || ["", ""];
  const lo = textbox("Wavelength", [], req[0], () => {});
  const hi = textbox("to", [], req[1], () => {});
  const step = textbox("nm, step", [], x.step, () => {});
  const set = document.createElement("button");
  set.textContent = "Set";
  set.addEventListener("click", () => change({
    wavelengths: [numberOrNull(lo.querySelector("input").value), numberOrNull(hi.querySelector("input").value)],
    step: numberOrNull(step.querySelector("input").value),
  }));
  const home = document.createElement("button");
  home.textContent = "Home";
  home.title = "The model's own wavelength range";
  home.addEventListener("click", () => change({ wavelengths: null, step: null }));
  box.append(lo, hi, step, set, home);
}

function buildFarFieldControls(box, reply) {
  const m = reply.menu || {};
  if ((m.data || []).length > 1) {
    box.append(select("Data", m.data, (m.selected || {}).data, (v) => change({ data: v })));
  }
}

// A sweep's menus come in the reply, each {name, label, options, selected}: which
// result, which part of it, which panel of a grid, which ports of an S-matrix.
// A menu with one choice is not shown.
function buildSweepControls(box, reply) {
  for (const m of (reply.extra || {}).menus || []) {
    if ((m.options || []).length < 2) continue;
    box.append(select(m.label, m.options, m.selected, (v) => change({ [m.name]: v })));
  }
}

// ------------------------------------------------------------ the export panel

const MM_PER = { mm: 1, in: 25.4 };

function field(label, value, size) {
  const wrap = document.createElement("label");
  wrap.append(label + ":");
  const input = document.createElement("input");
  input.type = "text";
  input.size = size || 6;
  input.value = value == null ? "" : value;
  wrap.append(input);
  return [wrap, input];
}

function exportPanel(reply) {
  const c = reply.controls || {};
  const details = document.createElement("details");
  details.className = "export";
  const summary = document.createElement("summary");
  summary.textContent = "Export";
  const panel = document.createElement("div");
  const format = document.createElement("select");
  for (const [value, text] of [["png-browser", "PNG (browser)"], ["svg-browser", "SVG (browser)"],
    ["png", "PNG at the dpi"], ["pdf", "PDF"], ["svg", "SVG (matplotlib)"]]) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = text;
    format.appendChild(o);
  }
  const fmtLabel = document.createElement("label");
  fmtLabel.append("Format:", format);
  const size = c.figure_size || [80, 56.6];
  const [wLabel, width] = field("Width", size[0]);
  const [hLabel, height] = field("Height", size[1]);
  const unit = document.createElement("select");
  for (const u of ["mm", "in"]) {
    const o = document.createElement("option");
    o.value = o.textContent = u;
    unit.appendChild(o);
  }
  unit.addEventListener("change", () => {
    // keep the size, shown in the new unit
    const from = unit.value === "mm" ? MM_PER.in : MM_PER.mm;
    const to = MM_PER[unit.value];
    for (const input of [width, height]) {
      const v = numberOrNull(input.value);
      if (v !== null) input.value = Math.round((v * from / to) * 100) / 100;
    }
  });
  const [dLabel, dpi] = field("dpi", c.dpi || 600, 4);
  const [nLabel, name] = field("File name", "EM_" + reply.id, 16);

  const labels = document.createElement("details");
  const ls = document.createElement("summary");
  ls.textContent = "Labels";
  const lbox = document.createElement("div");
  const entries = {};
  for (const [key, text] of [["title", "Title"], ["xlabel", "x label"], ["ylabel", "y label"],
    ["colorbar_label", "Colorbar label"], ["font_size", "Font size"], ["units", "Units (nm or um)"]]) {
    const [wrap, input] = field(text, c[key], key === "font_size" ? 4 : 18);
    entries[key] = input;
    lbox.append(wrap);
  }
  const apply = document.createElement("button");
  apply.textContent = "Apply";
  apply.addEventListener("click", () => change(labelControls(entries)));
  lbox.append(apply);
  labels.append(ls, lbox);

  const options = () => {
    const mm = MM_PER[unit.value];
    return {
      file_name: name.value,
      figure_size: [numberOrNull(width.value) * mm, numberOrNull(height.value) * mm],
      dpi: numberOrNull(dpi.value),
      ...labelControls(entries),
    };
  };
  const download = document.createElement("button");
  download.textContent = "Download";
  download.addEventListener("click", () => downloadFigure(reply.id, format.value, options()));
  const save = document.createElement("button");
  save.textContent = "Save";
  save.title = "Save under the script's working directory, in EM_figures/ unless the name has a folder";
  save.addEventListener("click", () => saveFigure(reply.id, format.value, options()));
  panel.append(fmtLabel, wLabel, hLabel, unit, dLabel, nLabel, labels, download, save);
  details.append(summary, panel);
  return details;
}

function labelControls(entries) {
  const out = {};
  for (const [key, input] of Object.entries(entries)) {
    const v = input.value.trim();
    out[key] = v === "" ? null : (key === "font_size" ? Number(v) : v);
  }
  return out;
}

async function downloadFigure(id, format, options) {
  if (format.endsWith("-browser")) {
    const px = 96 / 25.4;
    const scale = format.startsWith("png") ? (options.dpi || 96) / 96 : 1;
    return Plotly.downloadImage(document.getElementById("figure"), {
      format: format.split("-")[0],
      width: Math.round(options.figure_size[0] * px),
      height: Math.round(options.figure_size[1] * px),
      scale: scale,
      filename: options.file_name,
    });
  }
  const r = await fetch("api/views/" + id + "/export", {
    method: "POST", headers: HEADERS, body: JSON.stringify({ ...options, file_type: format }),
  });
  if ((r.headers.get("Content-Type") || "").startsWith("application/json")) {
    const reply = await r.json();
    return showMessage(reply.error || "");
  }
  const blob = await r.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = options.file_name.endsWith("." + format) ? options.file_name : options.file_name + "." + format;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function saveFigure(id, format, options) {
  const fileType = format.split("-")[0];
  const reply = await api("views/" + id + "/save", { ...options, file_type: fileType });
  if (reply.error) showMessage(reply.error);
  else {
    showMessage("");
    setStatus("saved " + reply.path);
  }
}

// ------------------------------------------------------------ the session page

const SESSION_ROWS = [
  ["EMode", (e) => e.version],
  ["License", (e) => e.license_type],
  ["Account", (e) => e.account],
  ["Started", (e) => e.started && (new Date(e.started).toLocaleString() + ", " + since(e.started))],
  ["Simulation", (e) => e.simulation_name],
  ["File", (e) => e.eph_path],
  ["File size", (e) => bytes(e.eph_bytes)],
  ["EMode process", (e) => "pid " + e.pid + ", " + (e.threads || "?") + " threads"],
  ["EMode memory", (e) => bytes(e.rss_bytes)],
  ["EMode CPU", (e) => e.cpu_percent == null ? null : e.cpu_percent.toFixed(1) + "%"],
  ["Solve pool", (e) => e.solve_pool_workers + " workers at most"],
  ["Solver cache", (e) => e.cache_entries + " entries"],
  ["Error reports", (e) => e.session_errors + " waiting for the next license check-in"],
];

const CLIENT_ROWS = [
  ["Client", (c) => "emode " + c.version],
  ["Python", (c) => c.python + ", " + c.executable],
  ["Client process", (c) => "pid " + c.pid + (c.threads ? ", " + c.threads + " threads" : "")],
  ["Client memory", (c) => c.memory || bytes(c.rss_bytes)],
  ["Client CPU", (c) => c.cpu_percent == null ? null : c.cpu_percent.toFixed(1) + "%"],
  ["This tab", (c) => "port " + c.ui_port + ", " + c.views + " plots"],
];

function bytes(n) {
  if (n == null) return null;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(i ? 1 : 0) + " " + units[i];
}

function since(iso) {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return "running " + (h ? h + " h " : "") + m + " min";
}

function rows(table, spec, data) {
  for (const [label, get] of spec) {
    let value = null;
    try { value = data ? get(data) : null; } catch (err) { value = null; }
    if (value == null || value === "") continue;
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = label;
    const td = document.createElement("td");
    td.textContent = value;
    tr.append(th, td);
    table.appendChild(tr);
  }
}

async function loadSession() {
  let page;
  try {
    page = await api("session");
  } catch (err) {
    return showSessionMessage(String(err));
  }
  const table = document.getElementById("session-table");
  table.replaceChildren();
  rows(table, SESSION_ROWS, page.emode);
  rows(table, CLIENT_ROWS, page.client);
  showSessionMessage(page.session_ended ? "The EMode session has ended." : (page.error || ""));
  if (page.client && page.client.docs) document.getElementById("docs").href = page.client.docs;
}

function showSessionMessage(text) {
  const el = document.getElementById("session-message");
  el.hidden = !text;
  el.textContent = text || "";
}

function buildPlotControls(box, reply) {
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
    if (!EMBED && !location.hash.startsWith("#/session")) location.hash = "#/plots/" + data.id;
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

let sessionTimer = null;

function route() {
  const onSession = location.hash.startsWith("#/session");
  document.getElementById("app").hidden = onSession;
  document.getElementById("session").hidden = !onSession;
  for (const a of document.querySelectorAll("#nav .tab[data-route]")) {
    a.classList.toggle("active", a.dataset.route === (onSession ? "session" : "plots"));
  }
  clearInterval(sessionTimer);
  if (onSession) {
    loadSession();
    // refreshed every five seconds while the page is in view
    sessionTimer = setInterval(() => {
      if (document.visibilityState === "visible") loadSession();
    }, 5000);
    return;
  }
  const id = routeId() || (state.views.length ? state.views[state.views.length - 1].id : null);
  if (id && id !== state.current) show(id);
  else if (id && state.reply) Plotly.Plots.resize(document.getElementById("figure"));
}

window.addEventListener("hashchange", route);
window.addEventListener("pagehide", () => {
  fetch("api/closed", { method: "POST", headers: HEADERS, body: "{}", keepalive: true }).catch(() => {});
});

document.addEventListener("DOMContentLoaded", async () => {
  if (EMBED) document.body.classList.add("embed");
  await loadList().catch((err) => showMessage(String(err)));
  route();
  // the docs link names the running EMode's version
  api("session").then((page) => {
    if (page.client && page.client.docs) document.getElementById("docs").href = page.client.docs;
  }).catch(() => {});
  listen();
  setInterval(() => {
    if (!state.ended) api("ping", {}).catch(sessionEnded);
  }, 2000);
  api("ping", {}).catch(() => {});
});
