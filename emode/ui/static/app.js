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

const state = { views: [], current: null, reply: null, ended: false, busyTimer: null, autoRender: null, benchTimer: null };

const SESSION_GONE = "This EMode session has ended, so this page can no longer reach it. You can close this tab.";

function api(path, body) {
  const options = body === undefined
    ? { headers: HEADERS }
    : { method: "POST", headers: HEADERS, body: JSON.stringify(body) };
  return fetch("api/" + path, options).then(
    (r) => {
      if (!r.ok) throw new Error("EMode answered " + r.status);
      return r.json();
    },
    () => {
      // The request never reached the client's server: the script that owned this page has ended.
      throw new Error(SESSION_GONE);
    },
  );
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

// #/plots alone is the workbench (emode#1701), the tab's default plot
function routeId() {
  const m = location.hash.match(/^#\/plots\/([\w-]+)/);
  return m ? m[1] : "bench";
}

async function loadList() {
  state.views = await api("views");
  renderList();
}

// The saved views (newest first) grouped by their first tag (GUI-M4d, emode#1704): the tags'
// groups in alphabetical order, then the untagged views. With no tags at all, one group.
// `filter` keeps the views whose name or a tag contains it, ignoring case.
function groupViews(saved, filter) {
  const text = (filter || "").trim().toLowerCase();
  const kept = saved.filter((v) => !text || [v.title || "", ...(v.tags || [])].some((t) => t.toLowerCase().includes(text)));
  const tagged = saved.some((v) => (v.tags || []).length);
  if (!tagged) return kept.length ? [["Saved views", kept]] : [];
  const groups = new Map();
  for (const v of kept) {
    const tag = (v.tags || [])[0] || "";
    if (!groups.has(tag)) groups.set(tag, []);
    groups.get(tag).push(v);
  }
  const names = [...groups.keys()].filter((t) => t).sort((a, b) => a.localeCompare(b));
  const out = names.map((t) => [t, groups.get(t)]);
  if (groups.has("")) out.push(["Untagged", groups.get("")]);
  return out;
}

function renderList() {
  const ol = document.getElementById("views");
  ol.replaceChildren();
  // the workbench is pinned first, the saved views follow, newest first
  const views = state.views || [];
  const bench = views.filter((v) => v.kind === "bench");
  const saved = views.filter((v) => v.kind !== "bench").reverse();
  for (const v of bench) ol.appendChild(sidebarEntry(v));
  const box = document.getElementById("views-filter");
  box.hidden = !saved.length;
  const groups = groupViews(saved, box.value);
  for (const [name, members] of groups) {
    const head = document.createElement("li");
    head.className = "group";
    head.textContent = name;
    ol.appendChild(head);
    for (const v of members) ol.appendChild(sidebarEntry(v));
  }
  if (saved.length && !groups.length) {
    const none = document.createElement("li");
    none.className = "empty";
    none.textContent = "No saved view matches “" + box.value.trim() + "”";
    ol.appendChild(none);
  }
  markSelected();
}

document.getElementById("views-filter").addEventListener("input", renderList);

// One entry of the sidebar (emode#1703). A script's view shows its own data, a saved
// recipe opens on the workbench, and the entry's menu edits it on the workbench,
// renames it or removes it.
function sidebarEntry(v) {
  const li = document.createElement("li");
  li.dataset.id = v.id;
  const title = document.createElement("span");
  title.className = "title";
  title.textContent = v.title || v.id;
  li.appendChild(title);
  const when = document.createElement("span");
  when.className = "when";
  if (v.kind === "bench") {
    li.classList.add("pinned");
    when.textContent = "any data of this session";
    li.appendChild(when);
  } else {
    const time = new Date(v.created * 1000).toLocaleTimeString();
    when.textContent = v.session_only ? "this session, " + time : (v.script ? "script, " + time : "saved view");
    li.appendChild(when);
    if ((v.tags || []).length) {
      const tags = document.createElement("span");
      tags.className = "tags";
      tags.textContent = v.tags.join(", ");
      li.appendChild(tags);
    }
    const menu = document.createElement("button");
    menu.type = "button";
    menu.className = "entry-menu";
    menu.textContent = "⋯";
    menu.setAttribute("aria-label", "Actions for " + (v.title || v.id));
    menu.setAttribute("aria-expanded", "false");
    const actions = document.createElement("div");
    actions.className = "entry-actions";
    actions.hidden = true;
    const action = (text, run) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.addEventListener("click", (ev) => {
        if (ev && ev.stopPropagation) ev.stopPropagation();
        actions.hidden = true;
        menu.setAttribute("aria-expanded", "false");
        run();
      });
      actions.appendChild(b);
    };
    if (!v.session_only) action("Edit in workbench", () => editInWorkbench(v.id));
    action("Rename", () => openSaveSheet({ mode: "rename", id: v.id, name: v.title || "", tags: v.tags || [] }));
    action("Remove", () => removeSaved(v));
    menu.addEventListener("click", (ev) => {
      if (ev && ev.stopPropagation) ev.stopPropagation();
      actions.hidden = !actions.hidden;
      menu.setAttribute("aria-expanded", String(!actions.hidden));
    });
    li.append(menu, actions);
  }
  li.addEventListener("click", () => { closeDrawer(); openEntry(v.id); });
  return li;
}

function openEntry(id) {
  const target = "#/plots/" + id;
  if (location.hash === target) {
    // the same saved view again: open its recipe afresh
    state.openedSaved = null;
    route();
  } else {
    location.hash = target;
  }
}

function markSelected() {
  // the workbench loaded from a saved view marks that view (emode#1703)
  const origin = state.current === "bench" && state.reply && state.reply.bench && state.reply.bench.origin;
  const shown = origin ? origin.id : state.current;
  for (const li of document.querySelectorAll("#views li")) {
    li.classList.toggle("selected", li.dataset.id === shown);
  }
}

// ------------------------------------------------------------ one view

// The plot's size in device pixels. On a phone the stage is a column and the plot has no
// height until a reply sizes it (#185), so its height is then its width times the plot box's
// aspect: EMode samples a panel at these pixels, and a panel 0 tall came back one row (emode#1701).
function pixels() {
  const el = document.getElementById("figure");
  const r = window.devicePixelRatio || 1;
  const width = el.clientWidth || document.getElementById("stage").clientWidth || 0;
  const aspect = (state.reply && state.reply.aspect) || Math.SQRT1_2;
  const height = el.clientHeight > 0 ? el.clientHeight : width * aspect;
  return [Math.max(1, Math.round(width * r)), Math.max(1, Math.round(height * r))];
}

async function request(id, controls) {
  clearTimeout(state.busyTimer);
  state.busyTimer = setTimeout(() => setStatus("EMode is busy", true), 1000);
  try {
    // the plot's CSS width sets the All modes grid's columns
    const width = document.getElementById("stage").clientWidth || document.getElementById("figure").clientWidth;
    return await api("views/" + id + "/data", { controls: controls || {}, pixels: pixels(), width: width });
  } finally {
    clearTimeout(state.busyTimer);
    setStatus(state.ended ? "session ended" : "");
  }
}

async function show(id, controls) {
  if (!id) return;
  const first = state.current !== id;
  state.current = id;
  forgetPending(id);
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
    // a refused choice on the workbench keeps its menus, so another can be chosen
    if (reply.bench && (!state.reply || state.reply.id !== id)) benchRefused(reply);
    else if (reply.bench) buildControls({ ...state.reply, bench: reply.bench });
    if (!state.reply || state.reply.id !== id) return;
  } else {
    // a notice: the plot could not be brought up to date with the session (#182)
    showMessage(reply.notice || "");
    state.reply = reply;
    draw(reply);
    markSelected();
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
  // the legend sits above the box, in the top margin: its room goes there, not to the box (#188)
  const top = (m.t || 0) + legend;
  const bottom = m.b || 0;
  // an extreme aspect ratio (0.001 to 1000, #189) still draws: the box is at least 40 px tall,
  // and the figure at most 6000 px
  const tall = Math.min(Math.max(40, box * (reply.aspect || Math.SQRT1_2)), MAX_FIGURE_PX - top - bottom);
  return { width: width, height: Math.round(tall + top + bottom), margin: { ...m, t: top } };
}

const MAX_FIGURE_PX = 6000;

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
  // a plot taller than the stage must scroll: Plotly takes every touch on a plot with a
  // drag mode for a zoom box, so none would scroll it (#185). A click still opens a panel.
  const stage = document.getElementById("stage").clientHeight;
  if (stage > 0 && layout.height > stage) layout.dragmode = false;
  // a figure Plotly cannot draw still gets its controls, so another choice can be made (emode#1701)
  try {
    // Plotly's Reset axes goes back to the range the plot was first drawn with, and react keeps
    // that range when a later figure has another (the window display widens it, a plane or mode
    // switch changes it): a figure with other ranges is drawn anew so Home returns to its own (#201)
    const ranges = JSON.stringify([layout.xaxis && layout.xaxis.range, layout.yaxis && layout.yaxis.range]);
    const fresh = fig.dataset.ranges !== ranges;
    fig.dataset.ranges = ranges;
    const draws = fresh ? Plotly.newPlot : Plotly.react;
    const drawn = draws.call(Plotly, fig, reply.figure.data, layout, {
      responsive: true,
      displaylogo: false,
      toImageButtonOptions: { filename: "EM_plot_" + reply.id, scale: 3 },
    });
    if (drawn && typeof drawn.catch === "function") drawn.catch((err) => showMessage("The plot could not be drawn: " + err));
  } catch (err) {
    showMessage("The plot could not be drawn: " + err);
  }
  fig.removeAllListeners && fig.removeAllListeners("plotly_click");
  fig.on && fig.on("plotly_click", (ev) => {
    // a swipe that scrolled the plot ends in a click without a drag mode: it opens nothing
    if (swiped) return;
    // a schematic's block, port or connection opens its details (#159)
    const point = ev.points && ev.points[0];
    const meta = point && point.data && point.data.meta;
    if (typeof meta === "string" && meta.startsWith("component:")) openDetails(meta.slice("component:".length));
    // a panel of the All modes grid opens its mode in the single view
    else if (typeof meta === "string" && meta.startsWith("mode:")) renderNow({ mode: meta.slice("mode:".length) });
  });
  fig.removeAllListeners && fig.removeAllListeners("plotly_clickannotation");
  fig.on && fig.on("plotly_clickannotation", (ev) => {
    const target = ev.annotation && ev.annotation.hovertext;
    if (target && target.startsWith("go to ")) renderNow({ section: target.slice(6) });
  });
  buildControls(reply);
}

// A touch that moved more than a few pixels is a swipe, not a tap (#185).
let swiped = false;
let touchAt = null;
function watchSwipes(el) {
  el.addEventListener("touchstart", (ev) => {
    const t = ev.touches && ev.touches[0];
    touchAt = t ? [t.clientX, t.clientY] : null;
    swiped = false;
  }, { passive: true });
  el.addEventListener("touchmove", (ev) => {
    const t = ev.touches && ev.touches[0];
    if (t && touchAt && Math.hypot(t.clientX - touchAt[0], t.clientY - touchAt[1]) > 10) swiped = true;
  }, { passive: true });
}

// ------------------------------------------------------------ auto render (#183)
//
// With Auto render on (the default), a control change redraws at once. Off, the
// changes wait in `pending` and go together when Render is pressed. A click on
// the plot itself (a grid panel, a section arrow, a key) renders at once, with
// whatever waits. The choice is remembered per browser, where storage allows.

const AUTO_RENDER_KEY = "emode.autoRender";
const pending = { id: null, controls: {} };

function autoRender() {
  if (state.autoRender == null) {
    let stored = null;
    try {
      stored = localStorage.getItem(AUTO_RENDER_KEY);
    } catch (err) {
      stored = null;
    }
    state.autoRender = stored !== "off";
  }
  return state.autoRender;
}

function setAutoRender(on) {
  state.autoRender = !!on;
  try {
    localStorage.setItem(AUTO_RENDER_KEY, on ? "on" : "off");
  } catch (err) {
    // the choice is only a convenience
  }
  if (on && Object.keys(pending.controls).length) renderNow();
}

function pendingControls() {
  return pending.id === state.current ? pending.controls : {};
}

// Drop what waits when another plot is shown: it was meant for the one before.
function forgetPending(id) {
  if (pending.id !== id) {
    pending.id = id;
    pending.controls = {};
  }
  markRender();
}

function markRender() {
  const button = state.renderButton;
  if (!button) return;
  const waiting = Object.keys(pendingControls()).length > 0;
  button.disabled = !waiting;
  button.classList.toggle("pending", waiting);
}

// A control change: drawn now, or kept for Render.
function change(controls) {
  if (!state.current) return;
  if (autoRender()) return show(state.current, controls);
  forgetPending(state.current);
  Object.assign(pending.controls, controls);
  markRender();
}

// Draw now, with whatever waits.
function renderNow(extra) {
  if (!state.current) return;
  const controls = { ...pendingControls(), ...(extra || {}) };
  pending.controls = {};
  markRender();
  return show(state.current, controls);
}

function renderPanel() {
  const wrap = document.createElement("span");
  wrap.className = "render";
  const label = document.createElement("label");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = autoRender();
  box.addEventListener("change", () => setAutoRender(box.checked));
  label.append(box, " Auto render");
  const button = document.createElement("button");
  button.id = "render";
  button.type = "button";
  button.textContent = "Render";
  button.title = "Draw the plot with the controls changed since it was drawn";
  button.addEventListener("click", () => renderNow());
  state.renderButton = button;
  wrap.append(label, button);
  return wrap;
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
// max, a number box beside it for an exact value, and a dropdown of the menu's
// named entries (EME, the profiles, the reductions). The named entries are a
// <select>, not a text box's datalist: a datalist filters its suggestions by the
// text already in the box, so on a phone a box showing "pair" offered only "pair",
// and the keyboard's autocorrect could post a name that matched nothing.
// Dragging moves the number box only, and letting go posts the position once,
// through `onslide` with the number. The dropdown and the number box post their
// value through `ontext`. A menu with no span (a profile plot's z, which only names
// a profile) is the dropdown alone.
function zSlider(label, menu, value, ontext, onslide) {
  const wrap = document.createElement("span");
  wrap.className = "slider";
  const lo = Number(menu.min), hi = Number(menu.max);
  const ranged = hi > lo;
  const names = (menu.options || []).map(String);
  const current = value == null ? "" : String(value);
  const named = names.indexOf(current) >= 0;
  let position = null;
  if (names.length) {
    const s = document.createElement("select");
    s.setAttribute("aria-label", label);
    if (ranged) {
      // the first entry stands for a position along the range, and shows it
      position = document.createElement("option");
      position.value = named ? "" : current;
      position.textContent = named ? "position" : current;
      if (!named) position.selected = true;
      s.appendChild(position);
    }
    for (const v of names) {
      const o = document.createElement("option");
      o.value = o.textContent = v;
      if (v === current) o.selected = true;
      s.appendChild(o);
    }
    s.addEventListener("change", () => {
      if (s.value !== "" && !(position && s.value === position.value)) ontext(s.value);
    });
    const wrapSelect = document.createElement("label");
    wrapSelect.append(label + ":", s);
    wrap.append(wrapSelect);
  }
  if (!ranged) {
    if (!names.length) wrap.append(numberBox(label, current, ontext));
    return wrap;
  }
  const range = document.createElement("input");
  range.type = "range";
  const step = sliderStep(hi - lo);
  const digits = Math.max(0, -Math.floor(Math.log10(step)));
  range.min = String(lo);
  range.max = String(hi);
  range.step = String(step);
  const start = parseFloat(current);
  range.value = String(Number.isFinite(start) ? start : (lo + hi) / 2);
  range.setAttribute("aria-label", label + " (" + menu.unit + ")");
  const box = numberBox(names.length ? "" : label, named ? "" : current, ontext);
  const text = box.querySelector("input");
  const shown = () => Number(Number(range.value).toFixed(digits));
  range.addEventListener("input", () => {
    text.value = shown() + " " + menu.unit;
    if (position) position.textContent = text.value;
  });
  range.addEventListener("change", () => onslide(shown()));
  wrap.append(box, range);
  return wrap;
}

// A plain text box for a number, with no suggestions and no autocorrect.
function numberBox(label, value, onchange) {
  const wrap = document.createElement("label");
  if (label) wrap.append(label + ":");
  const input = document.createElement("input");
  input.type = "text";
  input.size = 9;
  input.value = value || "";
  input.setAttribute("inputmode", "decimal");
  input.setAttribute("autocorrect", "off");
  input.setAttribute("autocapitalize", "off");
  input.setAttribute("spellcheck", "false");
  input.addEventListener("change", () => onchange(input.value));
  wrap.append(input);
  return wrap;
}

// On a phone the per-kind controls fold into a Settings section, closed by default, so the
// plot is not pushed below the fold (#196). Its state is kept per browser, where storage allows.
const SETTINGS_KEY = "emode.settingsOpen";

function storedSettingsOpen() {
  try {
    return localStorage.getItem(SETTINGS_KEY) === "open";
  } catch (err) {
    return false;
  }
}

function settingsFold(box) {
  const details = document.createElement("details");
  details.className = "settings";
  details.open = storedSettingsOpen();
  const summary = document.createElement("summary");
  summary.textContent = "Settings";
  const body = document.createElement("div");
  details.append(summary, body);
  details.addEventListener("toggle", () => {
    try {
      localStorage.setItem(SETTINGS_KEY, details.open ? "open" : "closed");
    } catch (err) {
      // the choice is only a convenience
    }
  });
  box.append(details);
  return body;
}

function buildControls(reply) {
  const box = document.getElementById("controls");
  box.replaceChildren();
  if (reply.bench) buildBenchControls(box, reply.bench);
  const kind = reply.kind || "plot";
  buildLayerPanel(reply);
  const bar = narrowQuery.matches ? settingsFold(box) : box;
  if (kind === "smatrix") buildSMatrixControls(bar, reply);
  else if (kind === "material") buildMaterialControls(bar, reply);
  else if (kind === "farfield") buildFarFieldControls(bar, reply);
  else if (kind === "gds") buildGDSControls(bar, reply);
  else if (kind === "sweep") buildSweepControls(bar, reply);
  else buildPlotControls(bar, reply);
  if (kind === "schematic") bar.append(componentMenu(reply));
  if (kind !== "gds") bar.append(aspectControl(reply));
  if (kind !== "gds" && kind !== "schematic") bar.append(extentsPanel(reply));
  box.append(renderPanel());
  box.append(saveBar(reply));
  box.append(exportPanel(reply));
  markRender();
}

// crossing the breakpoint (a rotated phone) rebuilds the bar in the other layout
narrowQuery.addEventListener("change", () => {
  if (state.reply && state.current) buildControls(state.reply);
});

// ------------------------------------------------------------ saved views (emode#1703)
//
// The workbench's Save view button sits beside Render and Export, outside the phone's
// Settings fold, and opens a sheet: Save writes the workbench back to the saved view it
// came from, Save as keeps it under a new name. The bar says "modified from <name>" when
// the workbench's settings differ from that view. A script's view offers Edit in
// workbench, which loads its settings there.

function saveBar(reply) {
  const wrap = document.createElement("span");
  wrap.className = "save-view-bar";
  const bench = reply.bench;
  if (reply.id === "bench" && bench) {
    const origin = bench.origin;
    const mark = document.createElement("span");
    mark.className = "modified";
    mark.textContent = origin ? "modified from " + origin.name : "";
    mark.hidden = !(origin && bench.modified);
    const button = document.createElement("button");
    button.id = "save-view";
    button.type = "button";
    button.className = "primary";
    button.textContent = "Save view";
    button.title = "Keep these settings in the sidebar's saved views and in the simulation file";
    button.addEventListener("click", () => openSaveSheet({
      mode: "save", origin: origin, name: origin ? origin.name : (reply.title || ""),
      tags: origin ? origin.tags || [] : [],
    }));
    wrap.append(mark, button);
    return wrap;
  }
  const entry = (state.views || []).find((v) => v.id === reply.id);
  if (entry && entry.script) {
    const button = document.createElement("button");
    button.id = "edit-in-workbench";
    button.type = "button";
    button.textContent = "Edit in workbench";
    button.title = "Load this plot's settings into the workbench, to change and save them";
    button.addEventListener("click", () => editInWorkbench(reply.id));
    wrap.append(button);
  }
  return wrap;
}

// A button's one click handler, set again each time the sheet opens.
function onClick(id, run) {
  const el = document.getElementById(id);
  el._run = run;
  if (!el._bound) {
    el._bound = true;
    el.addEventListener("click", (ev) => el._run && el._run(ev));
  }
  return el;
}

let sheetOpener = null;

// Tags typed in the sheet, comma separated: trimmed, without blanks or repeats (GUI-M4d).
function sheetTags() {
  const seen = [];
  for (const t of document.getElementById("sheet-tags").value.split(",")) {
    const tag = t.trim();
    if (tag && !seen.includes(tag)) seen.push(tag);
  }
  return seen;
}

function openSaveSheet(how) {
  const sheet = document.getElementById("save-sheet");
  const name = document.getElementById("sheet-name");
  const rename = how.mode === "rename";
  document.getElementById("sheet-title").textContent = rename ? "Rename or tag the saved view" : "Save view";
  name.value = how.name || "";
  const tags = document.getElementById("sheet-tags");
  tags.value = (how.tags || []).join(", ");
  tags.hidden = false;
  onClick("sheet-save", () => postSaved({ action: "save", tags: sheetTags() })).hidden = rename || !how.origin;
  onClick("sheet-save-as", () => postSaved({ action: "save_as", name: name.value, tags: sheetTags() })).hidden = rename;
  onClick("sheet-rename", () => postSaved({ action: "rename", id: how.id, name: name.value, tags: sheetTags() })).hidden = !rename;
  onClick("sheet-cancel", closeSaveSheet);
  const save = document.getElementById("sheet-save");
  save.textContent = how.origin ? "Save to “" + how.origin.name + "”" : "Save";
  document.getElementById("sheet-error").hidden = true;
  if (sheet.hidden) sheetOpener = document.activeElement || null;
  sheet.hidden = false;
  if (name.focus) name.focus();
}

function closeSaveSheet() {
  const sheet = document.getElementById("save-sheet");
  if (sheet.hidden) return;
  sheet.hidden = true;
  if (sheetOpener && sheetOpener.focus) sheetOpener.focus();
  sheetOpener = null;
}

async function postSaved(body) {
  let reply;
  try {
    reply = await api("saved", body);
  } catch (err) {
    reply = { error: String(err) };
  }
  if (reply.error) {
    const error = document.getElementById("sheet-error");
    if (!document.getElementById("save-sheet").hidden) {
      error.textContent = reply.error;
      error.hidden = false;
    } else {
      showMessage(reply.error);
    }
    return;
  }
  closeSaveSheet();
  showMessage("");
  await loadList();
  // the bar's mark follows: the workbench now matches the view it was saved as
  if (state.current === "bench" && body.action !== "remove") show("bench", {});
}

function removeSaved(v) {
  const ok = typeof confirm !== "function" ||
    confirm("Remove the saved view “" + (v.title || v.id) + "”" + (v.session_only ? "?" : " from this session and its file?"));
  if (!ok) return;
  if (state.current === v.id) location.hash = "#/plots/bench";
  postSaved({ action: "remove", id: v.id });
}

// Load a script's view, or a saved recipe, into the workbench.
function editInWorkbench(id) {
  closeDrawer();
  state.openedSaved = null;
  pending.controls = {};
  const shown = show("bench", { saved: id });
  if (location.hash !== "#/plots/bench") location.hash = "#/plots/bench";
  return shown;
}

// A saved view's route (#/plots/<id>) opens its recipe on the workbench.
function openSaved(id) {
  state.openedSaved = id;
  pending.controls = {};
  return show("bench", { saved: id });
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !document.getElementById("save-sheet").hidden) {
    ev.preventDefault();
    closeSaveSheet();
  }
});

// ------------------------------------------------------------ the aspect ratio (#189)
//
// Width:height of the plot box, posted as `aspect_ratio`, the box's height over its width.
// sqrt(2) is every plot's default but the S-matrix's, which is 1:1. Custom takes any
// width:height from 0.001 to 1000: past that a figure degenerates.

const ASPECTS = [["√2", Math.SQRT2], ["golden", (1 + Math.sqrt(5)) / 2], ["silver", 1 + Math.SQRT2], ["1:1", 1]];
const ASPECT_MIN = 0.001;
const ASPECT_MAX = 1000;

function aspectControl(reply) {
  const c = reply.controls || {};
  const wrap = document.createElement("span");
  wrap.className = "aspect";
  const chosen = typeof c.aspect_ratio === "number" && c.aspect_ratio > 0 ? 1 / c.aspect_ratio : null;
  const named = chosen === null
    ? (reply.kind === "smatrix" ? "1:1" : "√2")
    : ((ASPECTS.find(([, r]) => Math.abs(r - chosen) < 1e-6 * r) || ["Custom"])[0]);
  const custom = numberBox("width:height", named === "Custom" ? String(Number(chosen.toPrecision(6))) : "", (text) => {
    const v = Number(String(text).trim());
    if (!(v >= ASPECT_MIN && v <= ASPECT_MAX)) {
      return showMessage("The aspect ratio is width:height, a number from " + ASPECT_MIN + " to " + ASPECT_MAX +
        ", and " + JSON.stringify(String(text)) + " is not one.");
    }
    showMessage("");
    change({ aspect_ratio: 1 / v });
  });
  custom.hidden = named !== "Custom";
  const menu = select("Aspect", ASPECTS.map(([n]) => n).concat(["Custom"]), named, (v) => {
    custom.hidden = v !== "Custom";
    const hit = ASPECTS.find(([n]) => n === v);
    if (hit) change({ aspect_ratio: 1 / hit[1] });
  });
  wrap.append(menu, custom);
  return wrap;
}

// ------------------------------------------------------------ the axis extents (#191)

// Each axis's ends: a number, or blank for automatic. A quantity that cannot be negative
// starts its automatic range at 0 in the client, which knows the data's kind.
function extentsPanel(reply) {
  const c = reply.controls || {};
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Axes";
  const list = document.createElement("div");
  for (const [name, key] of [["x min", "xmin"], ["x max", "xmax"], ["y min", "ymin"], ["y max", "ymax"]]) {
    list.append(numberBox(name, c[key] == null ? "" : String(c[key]), (v) => {
      const n = numberOrNull(v);
      if (n !== null && !Number.isFinite(n)) return showMessage(name + " is a number, or blank for automatic.");
      showMessage("");
      change({ [key]: n });
    }));
  }
  details.append(summary, list);
  return details;
}

// ------------------------------------------------------------ the workbench (emode#1701)
//
// The workbench's own menus come first: Field or Trend, then the data of that
// kind the session holds, from EMode's inventory. An entry the session cannot
// show yet is listed greyed with its reason ("run EME() first"). The chosen
// data's own controls follow, the bar the tab draws for its kind.

const CATEGORY_NAMES = { field: "Field", trend: "Trend" };

function optionSelect(label, options, selected, onchange) {
  const wrap = document.createElement("label");
  if (label) wrap.append(label + ":");
  const s = document.createElement("select");
  for (const opt of options) {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.text;
    if (opt.disabled) o.disabled = true;
    if (opt.title) o.title = opt.title;
    if (String(opt.value) === String(selected)) o.selected = true;
    s.appendChild(o);
  }
  s.addEventListener("change", () => onchange(s.value));
  wrap.appendChild(s);
  return wrap;
}

function datasetText(d) {
  const text = d.label + (d.summary ? ": " + d.summary : "");
  return d.available ? text : text + " (" + (d.reason || "not available") + ")";
}

// Choosing what the workbench shows draws it at once, even with Auto render off (#183):
// the changes kept for Render were the old data's, so they are dropped.
function benchChoose(controls) {
  if (!state.current) return;
  pending.controls = {};
  markRender();
  return show(state.current, controls);
}

function buildBenchControls(box, bench) {
  const row = document.createElement("span");
  row.className = "bench-menus";
  row.append(optionSelect("Plot",
    (bench.categories || []).map((c) => ({ value: c, text: CATEGORY_NAMES[c] || c })),
    bench.category, (v) => benchChoose({ category: v })));
  const data = (bench.datasets || []).map((d) => ({
    value: d.id, text: datasetText(d), disabled: !d.available, title: d.available ? "" : d.reason,
  }));
  if (!data.length) data.push({ value: "", text: "nothing yet", disabled: true });
  row.append(optionSelect("Data", data, bench.dataset, (v) => benchChoose({ dataset: v })));
  box.append(row);
}

// a refused choice with nothing shown yet: the workbench's menus alone
function benchRefused(reply) {
  const box = document.getElementById("controls");
  box.replaceChildren();
  buildBenchControls(box, reply.bench);
}

// The session changed (a solve, a label, a sweep): the workbench asks again what it can
// plot, once per two seconds at most and only while it is shown, so a script's loop of
// solves queues nothing behind the call lock.
function benchChanged() {
  if (state.current !== "bench" || state.benchTimer) return;
  state.benchTimer = setTimeout(() => {
    state.benchTimer = null;
    if (document.visibilityState === "visible" && state.current === "bench") show("bench", {});
  }, 2000);
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
  set.addEventListener("click", () => renderNow({
    wavelengths: [numberOrNull(lo.querySelector("input").value), numberOrNull(hi.querySelector("input").value)],
    step: numberOrNull(step.querySelector("input").value),
  }));
  const home = document.createElement("button");
  home.textContent = "Home";
  home.title = "The model's own wavelength range";
  home.addEventListener("click", () => renderNow({ wavelengths: null, step: null }));
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
  apply.addEventListener("click", () => renderNow(labelControls(entries)));
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

// Each row reads EMode's SessionInfo, except Holds, which the client draws from the plot
// inventory (GUI-M4d): `p` is the whole page, `e` its `emode` part.
const SESSION_ROWS = [
  ["EMode", (e) => e.version],
  ["Holds", (e, p) => p.holds],
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

function rows(table, spec, data, page) {
  for (const [label, get] of spec) {
    let value = null;
    try { value = data ? get(data, page || {}) : null; } catch (err) { value = null; }
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
  showSession(page);
}

function showSession(page) {
  const table = document.getElementById("session-table");
  table.replaceChildren();
  rows(table, SESSION_ROWS, page.emode, page);
  rows(table, CLIENT_ROWS, page.client, page);
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
  // the Slice menu: the planes, never the schematic, which is the View menu's (emode#1728). A
  // plane change clears the position (a z on x-y, a transverse position on a z panel), and a z
  // plane chosen from a profile's x-y plot names no profile, so EMode draws the device
  const planes = (m.planes || []).filter((p) => p !== "schematic");
  if (planes.length) {
    box.append(select("Slice", planes, sel.plane, (v) => change(v === "x-y"
      ? { plane: v, slice_position: null }
      : { plane: v, profile: null, slice_position: null })));
  }
  if (m.ports && m.ports.length) box.append(select("Port", m.ports, sel.port, (v) => change({ port: v })));
  if (m.sections && m.sections.length) {
    box.append(select("Section", m.sections, sel.section || "all", (v) => change({ section: v === "all" ? null : v })));
  }
  if (m.profiles && m.profiles.length) {
    // EME is the device, drawn on the x-y plane at its cut. No position goes with a choice:
    // the plot's own arguments may name no plane, and EMode's default z-x plane would read
    // a number as a y (#179).
    box.append(select("Profile", m.profiles, sel.profile, (v) => change(v === "EME"
      ? { profile: v, plane: "x-y", slice_position: null }
      : { profile: v, slice_position: null })));
  }
  if (m.modes && m.modes.length) box.append(select("Mode", m.modes, sel.mode, (v) => change({ mode: v })));
  if (m.slider) {
    // a z panel's transverse position or reduction (the x-y plane has none, its Profile menu
    // names EME and the profiles, #179)
    box.append(zSlider(m.slider.label.replace(":", ""), m.slider, m.slider.value,
      (v) => change({ slice_position: v }), (x) => change({ slice_position: x })));
  }
  if (m.position) {
    // the z of the x-y cut: marked on a z panel and drawn below it (emode#1625, #180), and the
    // device's own x-y plane's z, where a stale slice_position must not override it
    const xy = sel.plane === "x-y";
    const name = m.position.label.replace(":", "") + " (" + m.position.unit + ")";
    const move = (z) => change(xy ? { z_position: z, slice_position: null } : { z_position: z });
    const post = (v) => {
      const z = parseFloat(v);
      if (Number.isFinite(z)) move(z);
    };
    box.append(zSlider(xy ? name : "Cut " + name, m.position, parseFloat(m.position.value), post, move));
  }
  if (m.z_resolutions && m.z_resolutions.length) {
    box.append(textbox("Z-res (nm)", m.z_resolutions, sel.z_resolution || "Auto",
      (v) => change({ resolution: v === "Auto" || v === "" ? "auto" : Number(v) })));
  }
  if (m.settings && Object.keys(m.settings).length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Sampling";
    const list = document.createElement("div");
    const active = new Set(m.active_settings || []);
    for (const [name, arg] of SETTINGS) {
      // a setting EMode does not offer for this plot (Forward only and Blend slices outside an EME result) is not shown
      if (!(name in m.settings)) continue;
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
  // on the workbench the View menu switches its data, so the plot is shown again in place
  if (reply.id === id) return show(id, {});
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
  if (ev.key === "Escape") return renderNow({ section: null });
  const dir = { ArrowLeft: "◀", ArrowRight: "▶", ArrowUp: "▲", ArrowDown: "▼" }[ev.key];
  const hit = dir && arrows.find((a) => a.text === dir);
  if (hit) {
    ev.preventDefault();
    renderNow({ section: hit.hovertext.slice(6) });
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
  if (status.state === "signed_out" && status.two_factor_required) return switchToBrowserSignIn(status.error);
  showLoginError(status.error || "");
}

// EMode refused a password because the account has two-factor on (emode#470):
// say so above the form and start the browser sign-in, as if its button were pressed.
// EMode clears two_factor_required when the sign-in starts, so this runs once.
function switchToBrowserSignIn(message) {
  document.getElementById("login-password").value = "";
  showLoginError("");
  showAccountMessage(message || "This account uses two-factor authentication, so it signs in with your browser.");
  return startDeviceSignIn();
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
  showAccountMessage("");
  setText("login-submit", "Signing in…");
  return api("login/password", { email: email, password: input.value }).then(passwordAnswered).catch((err) => {
    setText("login-submit", "Log In");
    showLoginError(String(err));
  });
}

function passwordAnswered(status) {
  setText("login-submit", "Log In");
  if (status.state === "logged_in") document.getElementById("login-password").value = "";
  return showLogin(status);
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
  setText("error-reporting-explanation", "Sends EMode Photonix the error type, function, version and platform. Never your scripts, geometry or files.");
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
    if (!EMBED && !location.hash.startsWith("#/session") && !(onAccount() && account.state !== "logged_in")) {
      const target = "#/plots/" + data.id;
      // the same saved view plotted again (emode#1703): the hash does not change, so show it here
      if (location.hash === target) show(data.id);
      else location.hash = target;
    }
  } else if (event === "session-changed") {
    benchChanged();
  } else if (event === "saved-changed") {
    loadList().catch(() => {});
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
  const id = routeId();
  const entry = (state.views || []).find((v) => v.id === id);
  if (entry && entry.opens === "bench") {
    // a saved recipe opens on the workbench (emode#1703)
    if (state.openedSaved !== id) openSaved(id);
    else if (state.reply) fitFigure();
    return;
  }
  state.openedSaved = null;
  if (id && id !== state.current) show(id);
  else if (id && state.reply) fitFigure();
}

window.addEventListener("hashchange", route);
window.addEventListener("pagehide", () => {
  fetch("api/closed", { method: "POST", headers: HEADERS, body: "{}", keepalive: true }).catch(() => {});
});

document.addEventListener("DOMContentLoaded", async () => {
  if (EMBED) document.body.classList.add("embed");
  watchSwipes(document.getElementById("figure"));
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
