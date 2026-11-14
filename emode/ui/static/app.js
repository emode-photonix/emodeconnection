// The EMode app tab: the session's plots, each with its controls and the export
// panel, the session page, the account page (signing in, the licenses and the
// sessions), and a link to the documentation.
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

// the settings of the plot and the argument of plot() each sets. The view sets
// the space (layout is physical, simulation the simulation space), so there is
// no Physical space setting (emode#1649)
const SETTINGS = [
  ["Index outline", "index_outline"],
  ["Contours", "contours"],
  ["Text display", "text_display"],
  ["Mesh display", "mesh_display"],
  ["Window display", "window_display"],
  ["Forward only", "forward_only"],
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

// ------------------------------------------------------------ the collapsible sidebar

const SIDEBAR_KEY = "emode.sidebar";
const narrowQuery = window.matchMedia("(max-width: 800px)");

function storedSidebar() {
  try {
    return localStorage.getItem(SIDEBAR_KEY);
  } catch (err) {
    return null;
  }
}

function setSidebar(open, remember) {
  const app = document.getElementById("app");
  app.classList.toggle("sidebar-collapsed", !open);
  document.getElementById("sidebar-toggle").setAttribute("aria-expanded", String(open));
  // a wide screen remembers the choice, a narrow one always starts with the drawer shut
  if (remember && !narrowQuery.matches) {
    try {
      localStorage.setItem(SIDEBAR_KEY, open ? "open" : "closed");
    } catch (err) {
      // the choice is only a convenience
    }
  }
  // the plot takes the freed width once the transition is over
  setTimeout(() => {
    fitFigure();
  }, 250);
}

function sidebarOpen() {
  return !document.getElementById("app").classList.contains("sidebar-collapsed");
}

function closeDrawer() {
  if (narrowQuery.matches && sidebarOpen()) {
    setSidebar(false, false);
    document.getElementById("sidebar-toggle").focus();
  }
}

function startSidebar() {
  setSidebar(narrowQuery.matches ? false : storedSidebar() !== "closed", false);
}

document.getElementById("sidebar-toggle").addEventListener("click", () => setSidebar(!sidebarOpen(), true));
narrowQuery.addEventListener("change", startSidebar);
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && narrowQuery.matches && sidebarOpen()) {
    ev.stopImmediatePropagation();
    closeDrawer();
  }
});

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
    li.addEventListener("click", () => { closeDrawer(); location.hash = "#/plots/" + v.id; });
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
    closeDetails();
    document.title = "EMode: " + (reply.title || id);
    api("views/" + id + "/fetched", {}).catch(() => {});
  }
}

// The plot's size (emodeconnection#160): as wide as its container allows,
// never wider (the stage, which a figure grown to Plotly's default 700 px must
// not push out), and as tall as the reply's aspect (the plot box's height over
// its width) makes it, with the margins around the box. A phone's tall frame
// scrolls rather than stretching the plot.
function figureWidth(fig) {
  const room = document.getElementById("stage").clientWidth;
  const own = fig.clientWidth;
  return Math.max(100, own > 0 && room > 0 ? Math.min(own, room) : room || own);
}

function figureSize(fig, reply) {
  const layout = reply.figure.layout || {};
  const m = layout.margin || {};
  const colorbar = (reply.figure.data || []).some((t) => t.showscale || t.colorbar) ? 90 : 0;
  const legend = layout.showlegend ? 30 : 0;
  const width = figureWidth(fig);
  const box = Math.max(100, width - (m.l || 0) - (m.r || 0) - colorbar);
  return { width: width, height: Math.round(box * (reply.aspect || 0.618) + (m.t || 0) + (m.b || 0) + legend) };
}

function sizeFigure(reply) {
  const fig = document.getElementById("figure");
  const size = figureSize(fig, reply);
  fig.style.height = size.height + "px";
  return size;
}

// Refit the shown plot to its container: after a resize, a turn of the phone or the sidebar.
function fitFigure() {
  if (!state.reply || !state.reply.figure) return;
  Plotly.relayout(document.getElementById("figure"), sizeFigure(state.reply));
}

window.addEventListener("resize", fitFigure);

function draw(reply) {
  const fig = document.getElementById("figure");
  const layout = { ...reply.figure.layout, ...sizeFigure(reply), autosize: true };
  Plotly.react(fig, reply.figure.data, layout, {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: { filename: "EM_plot_" + reply.id, scale: 3 },
  });
  fig.removeAllListeners && fig.removeAllListeners("plotly_click");
  fig.on && fig.on("plotly_click", (ev) => {
    // a schematic's block, port or connection opens its details (#159)
    const point = ev.points && ev.points[0];
    const meta = point && point.data && point.data.meta;
    if (typeof meta === "string" && meta.startsWith("component:")) openDetails(meta.slice("component:".length));
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

// The step of a range slider over `span`: a round number near a 200th of it.
function sliderStep(span) {
  if (!(span > 0)) return 1;
  const raw = span / 200;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / p;
  return (n < 2 ? 1 : n < 5 ? 2 : 5) * p;
}

// A position slider (emodeconnection#161): a range from the menu's min to its
// max, with the text box beside it for an exact value or a named entry.
// Dragging moves the text box only, and letting go posts the position once,
// through `onslide` with the number. The text box posts its own text through `ontext`.
function zSlider(label, menu, value, ontext, onslide) {
  const wrap = document.createElement("span");
  wrap.className = "slider";
  const range = document.createElement("input");
  range.type = "range";
  const lo = Number(menu.min), hi = Number(menu.max);
  const step = sliderStep(hi - lo);
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  range.min = String(lo);
  range.max = String(hi);
  range.step = String(step);
  const start = parseFloat(value);
  range.value = String(Number.isFinite(start) ? start : (lo + hi) / 2);
  range.setAttribute("aria-label", label + " (" + menu.unit + ")");
  const box = textbox(label, menu.options, value, ontext);
  const text = box.querySelector("input");
  const shown = () => Number(Number(range.value).toFixed(digits));
  range.addEventListener("input", () => { text.value = shown() + " " + menu.unit; });
  range.addEventListener("change", () => onslide(shown()));
  wrap.append(box, range);
  return wrap;
}

function buildControls(reply) {
  const box = document.getElementById("controls");
  box.replaceChildren();
  const kind = reply.kind || "plot";
  buildLayerPanel(reply);
  if (kind === "smatrix") buildSMatrixControls(box, reply);
  else if (kind === "material") buildMaterialControls(box, reply);
  else if (kind === "farfield") buildFarFieldControls(box, reply);
  else if (kind === "gds") buildGDSControls(box, reply);
  else if (kind === "sweep") buildSweepControls(box, reply);
  else buildPlotControls(box, reply);
  if (kind === "schematic") box.append(componentMenu(reply));
  box.append(exportPanel(reply));
}

// ------------------------------------------------------------ a component's details (#159)
//
// A tap or click on a schematic's block, port or connection, or a choice in the
// Details menu (the keyboard's and a screen reader's way in), asks the client
// for the component's settings and parameters and shows them in the details
// panel: beside the plot on a wide screen, a sheet over its lower part on a phone.

function componentMenu(reply) {
  const wrap = document.createElement("label");
  wrap.append("Details:");
  const s = document.createElement("select");
  s.setAttribute("aria-label", "Component details");
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "choose a component";
  s.appendChild(none);
  for (const [id, label] of (reply.extra || {}).components || []) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = label;
    s.appendChild(o);
  }
  s.addEventListener("change", () => { if (s.value) openDetails(s.value); });
  wrap.appendChild(s);
  return wrap;
}

let detailsOpener = null;

async function openDetails(id) {
  let reply;
  try {
    reply = await api("views/" + state.current + "/details", { component: id });
  } catch (err) {
    return showMessage(String(err));
  }
  if (reply.error) return showMessage(reply.error);
  showMessage("");
  showDetails(reply);
}

function showDetails(details) {
  const panel = document.getElementById("details");
  document.getElementById("details-title").textContent = details.title;
  const body = document.getElementById("details-body");
  body.replaceChildren();
  let table = null;
  let group = null;
  for (const row of details.rows || []) {
    if (row.group !== group) {
      group = row.group;
      table = document.createElement("table");
      const caption = document.createElement("caption");
      caption.textContent = group;
      table.appendChild(caption);
      body.appendChild(table);
    }
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.setAttribute("scope", "row");
    th.textContent = row.name;
    const td = document.createElement("td");
    td.textContent = row.value;
    tr.append(th, td);
    table.appendChild(tr);
  }
  if (panel.hidden) detailsOpener = document.activeElement || null;
  panel.hidden = false;
  panel.focus();
}

function closeDetails() {
  const panel = document.getElementById("details");
  if (panel.hidden) return;
  panel.hidden = true;
  if (detailsOpener && detailsOpener.focus) detailsOpener.focus();
  detailsOpener = null;
}

document.getElementById("details-close").addEventListener("click", closeDetails);

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

// ------------------------------------------------------------ the GDS layer panel
//
// Each change restyles the layer's trace here, in the browser, and is then
// posted to the view's layers route so the client keeps it for the session (a
// later visit and a saved file draw the same). layerStyle() makes what the
// client's renderers/plotly.py layer_style() makes from the same row.

const HATCH_NAMES = { "": "none", "/": "/", "\\": "\\", "x": "x", "-": "-", "|": "|", "+": "+", ".": "." };

function layerRgba(color, alpha) {
  let h = color.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = (i) => parseInt(h.slice(i, i + 2), 16);
  return "rgba(" + n(0) + "," + n(2) + "," + n(4) + "," + Number(alpha) + ")";
}

function layerStyle(row) {
  const fill = layerRgba(row.color, row.opacity);
  return {
    visible: [row.visible],
    name: [row.label],
    fillcolor: [fill],
    "line.color": [row.color],
    "fillpattern.shape": [row.hatch],
    "fillpattern.fgcolor": [row.color],
    "fillpattern.bgcolor": [fill],
  };
}

function layerTrace(key) {
  const fig = document.getElementById("figure");
  return (fig.data || []).findIndex((t) => t.meta === "layer:" + key);
}

function restyleLayer(row) {
  const i = layerTrace(row.key);
  if (i >= 0) Plotly.restyle(document.getElementById("figure"), layerStyle(row), [i]);
}

async function postLayers(change) {
  const id = state.current;
  let reply;
  try {
    reply = await api("views/" + id + "/layers", change);
  } catch (err) {
    showMessage(String(err));
    return null;
  }
  if (reply.error) {
    showMessage(reply.error);
    return null;
  }
  showMessage("");
  if (state.reply && state.reply.id === id) {
    state.reply.extra.layers = reply.layers;
    state.reply.extra.shapes = reply.shapes;
    state.reply.extra.orientation = reply.orientation;
  }
  return reply;
}

function changeLayer(row, change, post = true) {
  Object.assign(row, change);
  restyleLayer(row);
  if (post) postLayers({ layers: { [row.key]: change } });
}

async function changeAllLayers(change, local) {
  const rows = state.reply.extra.layers;
  for (const row of rows) {
    local(row);
    restyleLayer(row);
  }
  buildLayerPanel(state.reply);
  await postLayers(change);
}

function layerRow(row) {
  const tr = document.createElement("tr");
  const visible = document.createElement("input");
  visible.type = "checkbox";
  visible.checked = row.visible;
  visible.title = "Show this layer";
  visible.addEventListener("change", () => changeLayer(row, { visible: visible.checked }));
  const color = document.createElement("input");
  color.type = "color";
  color.value = row.color.length === 4 ? "#" + row.color.slice(1).split("").map((c) => c + c).join("") : row.color;
  color.title = "Color";
  // dragging redraws, letting go keeps it
  color.addEventListener("input", () => changeLayer(row, { color: color.value }, false));
  color.addEventListener("change", () => changeLayer(row, { color: color.value }));
  const opacity = document.createElement("input");
  opacity.type = "range";
  opacity.min = "0";
  opacity.max = "1";
  opacity.step = "0.05";
  opacity.value = String(row.opacity);
  opacity.title = "Fill opacity";
  opacity.addEventListener("input", () => changeLayer(row, { opacity: Number(opacity.value) }, false));
  opacity.addEventListener("change", () => changeLayer(row, { opacity: Number(opacity.value) }));
  const hatch = document.createElement("select");
  hatch.title = "Hatch";
  for (const h of (state.reply.extra.hatches || Object.keys(HATCH_NAMES))) {
    const o = document.createElement("option");
    o.value = h;
    o.textContent = HATCH_NAMES[h] || h;
    if (h === row.hatch) o.selected = true;
    hatch.appendChild(o);
  }
  hatch.addEventListener("change", () => changeLayer(row, { hatch: hatch.value }));
  const solo = document.createElement("button");
  solo.textContent = "only";
  solo.title = "Show only this layer";
  solo.addEventListener("click", () => changeAllLayers({ solo: row.key }, (r) => { r.visible = r.key === row.key; }));
  const label = document.createElement("span");
  label.textContent = row.label;
  label.title = row.polygons + " polygons";
  for (const el of [visible, color, opacity, hatch, label, solo]) {
    const td = document.createElement("td");
    td.appendChild(el);
    tr.appendChild(td);
  }
  return tr;
}

function buildLayerPanel(reply) {
  const panel = document.getElementById("layer-panel");
  if (!panel) return;
  const gds = (reply.kind || "plot") === "gds";
  panel.hidden = !gds;
  panel.replaceChildren();
  if (!gds) return;
  const x = reply.extra || {};
  const head = document.createElement("div");
  head.className = "layer-head";
  const all = document.createElement("button");
  all.textContent = "Show all";
  all.addEventListener("click", () => changeAllLayers({ show_all: true }, (r) => { r.visible = true; }));
  head.appendChild(all);
  if (x.has_shapes) {
    const wrap = document.createElement("label");
    const shapes = document.createElement("input");
    shapes.type = "checkbox";
    shapes.checked = !!x.shapes;
    shapes.addEventListener("change", () => {
      const i = layerTrace("shapes");
      if (i >= 0) Plotly.restyle(document.getElementById("figure"), { visible: [shapes.checked] }, [i]);
      x.shapes = shapes.checked;
      postLayers({ shapes: shapes.checked });
    });
    wrap.append(shapes, " Simulation shapes");
    head.appendChild(wrap);
  }
  // y up as KLayout draws the cell, or the layout view's axes: the y range turns over here
  head.appendChild(select("Axes", ["KLayout (y up)", "layout view (x down)"],
    x.orientation === "layout" ? "layout view (x down)" : "KLayout (y up)", (v) => {
      const orientation = v.startsWith("layout") ? "layout" : "gds";
      const fig = document.getElementById("figure");
      const range = (fig.layout.yaxis.range || []).slice();
      const down = range.length === 2 && range[0] > range[1];
      if (range.length === 2 && down !== (orientation === "layout")) {
        Plotly.relayout(fig, { "yaxis.range": [range[1], range[0]] });
      }
      x.orientation = orientation;
      postLayers({ orientation });
    }));
  const file = document.createElement("label");
  file.className = "lyp";
  file.append("KLayout .lyp: ");
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".lyp,.xml";
  input.addEventListener("change", async () => {
    const f = input.files && input.files[0];
    if (!f) return;
    const reply = await postLayers({ layer_properties: await f.text() });
    if (!reply) return;
    for (const row of reply.layers) restyleLayer(row);
    buildLayerPanel(state.reply);
  });
  file.appendChild(input);
  head.appendChild(file);
  panel.appendChild(head);
  const table = document.createElement("table");
  for (const row of x.layers || []) table.appendChild(layerRow(row));
  panel.appendChild(table);
  if (x.decimated) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = "Large layers drawn to within " + Number(x.pixel_nm).toPrecision(3) + " nm (one pixel).";
    panel.appendChild(note);
  }
}

function buildGDSControls(box, reply) {
  const n = ((reply.extra || {}).layers || []).length;
  const span = document.createElement("span");
  span.textContent = n + (n === 1 ? " GDS layer" : " GDS layers");
  box.append(span);
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
  // PNG and SVG are drawn here by Plotly, PDF by matplotlib in the client (#164)
  for (const [value, text] of [["png", "PNG"], ["svg", "SVG"], ["pdf", "PDF"]]) {
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

// The figure as Plotly draws it: the export panel's size in mm at 96 px per inch,
// a PNG scaled up to the dpi.
function imageOptions(format, options) {
  const px = 96 / 25.4;
  return {
    format: format,
    width: Math.round(options.figure_size[0] * px),
    height: Math.round(options.figure_size[1] * px),
    scale: format === "png" ? (options.dpi || 96) / 96 : 1,
    filename: options.file_name,
  };
}

function downloadFigure(id, format, options) {
  if (format === "pdf") return downloadServer(id, format, options);
  return Plotly.downloadImage(document.getElementById("figure"), imageOptions(format, options));
}

// A PDF: matplotlib draws it in the client, which sends it back as a file.
async function downloadServer(id, format, options) {
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

function saved(reply) {
  if (reply.error) showMessage(reply.error);
  else {
    showMessage("");
    setStatus("saved " + reply.path);
  }
}

// Save under the script's working directory: a PNG or an SVG as Plotly drew it
// here, a PDF drawn by matplotlib in the client.
function saveFigure(id, format, options) {
  const post = (extra) => api("views/" + id + "/save", { ...options, file_type: format, ...extra })
    .then(saved).catch((err) => showMessage(String(err)));
  if (format === "pdf") return post({});
  return Plotly.toImage(document.getElementById("figure"), imageOptions(format, options))
    .then((image) => post({ image: image }))
    .catch((err) => showMessage(String(err)));
}

// ------------------------------------------------------------ the session page

const SESSION_ROWS = [
  ["EMode", (e) => e.version],
  ["License", (e) => e.license_type],
  ["Account", (e) => e.account],
  ["Started", (e) => e.started && (new Date(e.started).toLocaleString() + ", " + since(e.started))],
  ["Simulation name", (e) => e.simulation_name],
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
  if (m.views && m.views.length) {
    // layout and simulation change this plot, the schematic and the GDS view open beside it (#146)
    box.append(select("View", m.views, sel.view || c.view || "layout", (v) => {
      if (v === "layout" || v === "simulation") return change({ view: v });
      openView(v);
    }));
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
    // on the x-y plane the slider is the z of the cut, which a z panel then marks
    const isZ = m.slider.label.startsWith("z");
    const post = (v) => {
      const z = isZ ? parseFloat(v) : NaN;
      change(Number.isFinite(z) ? { slice_position: v, z_position: z } : { slice_position: v });
    };
    box.append(zSlider(m.slider.label.replace(":", ""), m.slider, m.slider.value, post,
      (x) => change(isZ ? { slice_position: x, z_position: x } : { slice_position: x })));
  }
  if (m.position) {
    // on a z panel: the z of the x-y cut, marked on the panel (emode#1625)
    const label = "Cut " + m.position.label.replace(":", "") + " (" + m.position.unit + ")";
    const post = (v) => {
      const z = parseFloat(v);
      if (Number.isFinite(z)) change({ z_position: z });
    };
    box.append(zSlider(label, m.position, parseFloat(m.position.value), post, (x) => change({ z_position: x })));
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
      cb.disabled = !active.has(name);
      cb.addEventListener("change", () => change({ [arg]: cb.checked }));
      label.append(cb, " " + name);
      list.appendChild(label);
    }
    details.append(summary, list);
    box.append(details);
  }
}

async function openView(name) {
  const id = state.current;
  let reply;
  try {
    reply = await api("views/" + id + "/open", { view: name });
  } catch (err) {
    return showMessage(String(err));
  }
  if (reply.error) return showMessage(reply.error);
  await loadList();
  location.hash = "#/plots/" + reply.id;
}

// ------------------------------------------------------------ keys: the section zoom

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !document.getElementById("details").hidden) {
    ev.preventDefault();
    return closeDetails();
  }
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

// ------------------------------------------------------------ the account page
//
// The login window's every function. The page shows what EMode answers and
// holds no secret: the password goes once, in a POST body, and the field is the
// only place it lives. A sign-in through the browser shows the code EMode got
// from the license manager and polls the status at the interval it gives.

const account = { timer: null, pollTimer: null, lastRefresh: 0, lastUse: Date.now(), state: null };
// the window's refresh rule: every 30 s while the page is in view and used in
// the last 15 minutes, on focus when the data is older than 5 s, and at once on Refresh
const ACCOUNT_POLL_MS = 30000;
const ACCOUNT_IDLE_MS = 900000;
const ACCOUNT_STALE_MS = 5000;

function onAccount() {
  return location.hash.startsWith("#/account");
}

function setText(id, text) {
  document.getElementById(id).textContent = text == null ? "" : String(text);
}

function showAccountMessage(text) {
  const el = document.getElementById("account-message");
  el.hidden = !text;
  el.textContent = text || "";
}

function showLoginError(text) {
  const el = document.getElementById("login-error");
  el.hidden = !text;
  el.textContent = text || "";
}

function togglePassword() {
  const input = document.getElementById("login-password");
  const button = document.getElementById("password-toggle");
  const show = input.type === "password";
  input.type = show ? "text" : "password";
  button.textContent = show ? "Hide" : "Show";
  button.setAttribute("aria-pressed", show ? "true" : "false");
}

// the sign-in state: the form, the browser sign-in's code while it is pending,
// and the account panel once signed in
function showLogin(status) {
  account.state = status.state;
  if (status.version) setText("account-version", "Version " + status.version);
  if (status.state === "logged_in") {
    clearTimeout(account.pollTimer);
    return loadAccountInfo();
  }
  document.getElementById("account").classList.add("form-view");
  document.getElementById("signed-out").hidden = false;
  document.getElementById("signed-in").hidden = true;
  if (status.device !== undefined) document.getElementById("device-signin").hidden = !status.device;
  const pending = status.state === "pending";
  const ended = status.state === "denied" || status.state === "expired";
  if (pending || ended) document.getElementById("device-signin").hidden = false;
  document.getElementById("device-panel").hidden = !(pending || ended);
  document.getElementById("device-wait").hidden = !pending;
  document.getElementById("device-retry").hidden = !ended;
  if (pending) {
    setText("device-code", status.user_code);
    const link = document.getElementById("device-link");
    link.setAttribute("href", status.verification_url_complete || status.verification_url || "#");
    clearTimeout(account.pollTimer);
    account.pollTimer = setTimeout(pollLogin, Math.max(1, status.interval || 5) * 1000);
  }
  showLoginError(status.error || "");
}

function pollLogin() {
  if (!onAccount() && account.state !== "pending") return;
  api("login/status").then(showLogin).catch((err) => showAccountMessage(String(err)));
}

function signInWithPassword() {
  const email = document.getElementById("login-email").value.trim();
  const input = document.getElementById("login-password");
  if (!email || !input.value) return showLoginError("Enter your email and password.");
  showLoginError("");
  setText("login-submit", "Signing in…");
  return api("login/password", { email: email, password: input.value }).then((status) => {
    setText("login-submit", "Log In");
    if (status.state === "logged_in") input.value = "";
    showLogin(status);
  }).catch((err) => {
    setText("login-submit", "Log In");
    showLoginError(String(err));
  });
}

function startDeviceSignIn() {
  showLoginError("");
  return api("login/start", {}).then((status) => {
    showLogin(status);
    const url = status.verification_url_complete || status.verification_url;
    // the website's sign-in, in the same browser, where the user is likely signed in already
    if (status.state === "pending" && url && window.open) window.open(url, "_blank", "noopener");
  }).catch((err) => showLoginError(String(err)));
}

// a table cell, labeled with its column for the phone layout, where each row is a card
function cell(text, label) {
  const td = document.createElement("td");
  td.textContent = text == null ? "" : String(text);
  if (label) td.setAttribute("data-label", label);
  return td;
}

function showAccount(view) {
  if (view.state === "signed_out") return loadLogin();
  if (view.error) return showAccountMessage(view.error);
  showAccountMessage("");
  account.lastRefresh = Date.now();
  document.getElementById("account").classList.remove("form-view");
  document.getElementById("signed-out").hidden = true;
  document.getElementById("signed-in").hidden = false;
  setText("account-email", view.email);
  if (view.version) setText("account-version", "Version " + view.version);
  setText("count-2d", view.counts["2D"]);
  setText("count-3d", view.counts["3D"]);
  document.getElementById("sharing-link").hidden = !view.sharing.link;
  setText("sharing-line", view.sharing.line || "");
  const notice = document.getElementById("update-notice");
  notice.hidden = !view.update_notice;
  notice.textContent = view.update_notice || "";
  document.getElementById("error-reporting").checked = !!view.error_reporting;

  const licenses = document.getElementById("license-rows");
  licenses.replaceChildren();
  for (const row of view.licenses) {
    const tr = document.createElement("tr");
    if (row.shared) tr.className = "shared";
    tr.append(cell(row.subscription, "Subscription"), cell(row.status, "Status"),
      cell(row.expiration, "Expiration"), cell(row.owner, "Owner"));
    licenses.appendChild(tr);
  }
  const sessions = document.getElementById("session-rows");
  sessions.replaceChildren();
  for (const row of view.sessions) {
    const tr = document.createElement("tr");
    if (row.shared) tr.className = "shared";
    tr.append(cell(row.type, "Type"), cell(row.user, "User"), cell(row.computer, "Computer"),
      cell(row.started, "Started"), cell(row.owner, "Owner"));
    const last = document.createElement("td");
    last.className = "clear-cell";
    if (row.clearable) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "clear-row";
      button.textContent = "✕";
      button.title = "Clear this session";
      button.setAttribute("aria-label", "Clear the " + row.type + " session of " + row.user + " on " + (row.computer || "an unknown computer"));
      button.addEventListener("click", () => clearSession(row));
      last.appendChild(button);
    }
    tr.appendChild(last);
    sessions.appendChild(tr);
  }
}

function clearSession(row) {
  if (!row.clearable || !window.confirm(row.confirm)) return;
  return api("login/clear", { mode: "session", session_handle: row.handle })
    .then(showAccount).catch((err) => showAccountMessage(String(err)));
}

function clearByMode() {
  const select = document.getElementById("clear-mode");
  const mode = select.value;
  select.value = "none";
  if (mode === "none") return;
  if (!window.confirm("Clear " + mode + " sessions? A simulation still running in a cleared session loses its license.")) return;
  return api("login/clear", { mode: mode }).then(showAccount).catch((err) => showAccountMessage(String(err)));
}

function setErrorReporting() {
  const box = document.getElementById("error-reporting");
  return api("login/settings", { error_reporting: box.checked })
    .then((reply) => { if (reply.error) showAccountMessage(reply.error); else box.checked = !!reply.error_reporting; })
    .catch((err) => showAccountMessage(String(err)));
}

function logOut() {
  return api("login/logout", {}).then(showLogin).catch((err) => showAccountMessage(String(err)));
}

function loadAccountInfo() {
  return api("login/info").then(showAccount).catch((err) => showAccountMessage(String(err)));
}

function loadLogin() {
  return api("login/status").then(showLogin).catch((err) => showAccountMessage(String(err)));
}

function loadAccount() {
  account.lastUse = Date.now();
  return loadLogin();
}

function startAccount() {
  setText("error-reporting-explanation",
    "When a session closes, EMode can send EMode Photonix a short report of each internal error: " +
    "the error type, the EMode function, the EMode version and the platform. It never includes " +
    "your script, geometry or files. The setting applies to every session on this computer, and " +
    "'emode settings --error-reporting on|off' changes it too.");
  document.getElementById("login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    signInWithPassword();
  });
  document.getElementById("password-toggle").addEventListener("click", togglePassword);
  document.getElementById("device-start").addEventListener("click", startDeviceSignIn);
  document.getElementById("device-retry").addEventListener("click", startDeviceSignIn);
  document.getElementById("logout").addEventListener("click", logOut);
  document.getElementById("refresh-sessions").addEventListener("click", () => { account.lastUse = Date.now(); loadAccountInfo(); });
  document.getElementById("clear-sessions").addEventListener("click", clearByMode);
  document.getElementById("error-reporting").addEventListener("change", setErrorReporting);
  const used = () => { account.lastUse = Date.now(); };
  document.addEventListener("keydown", used);
  document.addEventListener("pointerdown", used);
  window.addEventListener("focus", () => {
    if (onAccount() && account.state === "logged_in" && Date.now() - account.lastRefresh >= ACCOUNT_STALE_MS) loadAccountInfo();
  });
  account.timer = setInterval(() => {
    const active = onAccount() && document.visibilityState === "visible" && account.state === "logged_in";
    if (active && Date.now() - account.lastUse < ACCOUNT_IDLE_MS) loadAccountInfo();
  }, ACCOUNT_POLL_MS);
}

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
    if (!EMBED && !location.hash.startsWith("#/session") && !(onAccount() && account.state !== "logged_in")) location.hash = "#/plots/" + data.id;
  } else if (event === "login-changed") {
    if (onAccount()) loadAccount();
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
  const onAccountPage = onAccount();
  const page = onSession ? "session" : (onAccountPage ? "account" : "plots");
  document.getElementById("app").hidden = page !== "plots";
  document.getElementById("session").hidden = !onSession;
  document.getElementById("account").hidden = !onAccountPage;
  document.getElementById("nav").classList.toggle("on-session", page !== "plots");
  for (const a of document.querySelectorAll("#nav .tab[data-route]")) {
    a.classList.toggle("active", a.dataset.route === page);
  }
  clearInterval(sessionTimer);
  if (onAccountPage) {
    loadAccount();
    return;
  }
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
  else if (id && state.reply) fitFigure();
}

window.addEventListener("hashchange", route);
window.addEventListener("pagehide", () => {
  fetch("api/closed", { method: "POST", headers: HEADERS, body: "{}", keepalive: true }).catch(() => {});
});

document.addEventListener("DOMContentLoaded", async () => {
  if (EMBED) document.body.classList.add("embed");
  startSidebar();
  startAccount();
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
