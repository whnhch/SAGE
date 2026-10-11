// SAGE static demo — a port of demo/demo (Streamlit) to plain HTML/JS.
// Recommendations and scores are the precomputed demo results from
// demo/demo/components/table_list.py and pivot_table.py.

const DATA_URL = "data/zomato.csv";

// ---------------------------------------------------------------------------
// Demo results
// table: { rowName, rows, group (column-level header, optional), cols, values[row][col], decimals }
// ---------------------------------------------------------------------------
const CUISINES = ["Italian", "French", "Japanese"];
const CITIES = ["Bengaluru", "New Delhi", "Mumbai"];

const PIVOT_TABLES_ITER1 = [
  {
    query: "AVG(Rating) GROUP BY Price Range, Cuisines",
    utility: 83,
    table: {
      rowName: "Cuisines", rows: CUISINES, group: "Price Range", cols: ["$", "$$", "$$$"],
      values: [[3.3, 3.5, 4.5], [3.6, 4.5, 3.4], [4.5, 3.2, 4.2]], decimals: 1,
    },
  },
  {
    query: "COUNT(ID) GROUP BY City, Cuisines",
    utility: 80,
    table: {
      rowName: "Cuisines", rows: CUISINES, group: "City", cols: CITIES,
      values: [[40, 20, 50], [30, 10, 40], [20, 40, 60]],
    },
  },
  {
    query: "MAX(Cost) GROUP BY Cuisines",
    utility: 57,
    table: { rowName: "Cuisines", rows: CUISINES, cols: ["Cost"], values: [[1300], [1200], [1000]] },
  },
  {
    query: "AVG(Rating) GROUP BY City",
    utility: 75,
    table: { rowName: "City", rows: CITIES, cols: ["Rating"], values: [[4.4], [4.5], [4.2]], decimals: 1 },
  },
  {
    query: "COUNT(ID) GROUP BY Cuisines, Has Online Delivery",
    utility: 70,
    table: {
      rowName: "Cuisines", rows: CUISINES, group: "Has Online Delivery", cols: ["Yes", "No"],
      values: [[40, 20], [30, 10], [20, 40]],
    },
  },
];

const PIVOT_TABLES_ITER2 = [
  {
    query: "AVG(Rating) GROUP BY City, Cuisines",
    utility: 85,
    table: {
      rowName: "Cuisines", rows: CUISINES, group: "City", cols: CITIES,
      values: [[3.7, 3.8, 4.1], [4.0, 4.7, 4.2], [3.9, 4.5, 4.3]], decimals: 1,
    },
  },
  {
    query: "AVG(Cost) BY City",
    utility: 85,
    table: { rowName: "City", rows: CITIES, cols: ["Cost"], values: [[800], [1200], [1400]] },
  },
  {
    query: "MIN(Rating) By City, Price Range",
    utility: 72,
    table: {
      rowName: "City", rows: CITIES, group: "Price Range", cols: ["$", "$$", "$$$"],
      values: [[3.3, 2.8, 2.6], [2.3, 1.0, 2.5], [3.2, 3.0, 2.8]], decimals: 1,
    },
  },
];

// Shown in "Liked Tables" once the dataset is loaded (not part of the recommendations).
const DEFAULT_LIKED_QUERY = "AVG(Rating) GROUP BY Price Range, City";

const ITER_RESULTS = {
  1: {
    tables: PIVOT_TABLES_ITER1,
    diversity: 32,
    insightfulness: { "Attribute Significance": 0.87, "Informativeness": 0.75, "Trend": 0.55, "Surprise": 0.60 },
    interpretability: { "Density": 1.00, "Semantic Validity": 0.90, "Conciseness": 0.95 },
  },
  2: {
    tables: PIVOT_TABLES_ITER2,
    diversity: 30,
    insightfulness: { "Attribute Significance": 1.00, "Informativeness": 0.70, "Trend": 0.00, "Surprise": 0.00 },
    interpretability: { "Density": 1.00, "Semantic Validity": 1.00, "Conciseness": 1.00 },
  },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  columns: [],
  rows: [],
  diversity: 20,
  numTables: 5,
  attributes: [],
  iteration: 0,
  recommend: false,
  selected: null,
  liked: new Set(),     // keys: `${iteration}|${query}`
  disliked: new Set(),
  loginTime: new Date(),
};

// First click shows iteration-1 results; every later click shows iteration-2 results.
const currentResults = () => ITER_RESULTS[state.iteration <= 1 ? 1 : 2];
const visibleTables = () => currentResults().tables.slice(0, state.numTables);
const likeKey = (query) => `${state.iteration}|${query}`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') inQuotes = false;
      else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      rows.push(row); row = [];
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => r.length > 1 || r[0] !== "");
}

const isNumeric = (v) => v !== "" && !isNaN(Number(v));

function formatValue(v, decimals) {
  return decimals ? v.toFixed(decimals) : v.toLocaleString("en-US");
}

const pct = (v) => `${Math.round(v * 100)}%`;

function pivotTableHTML(t) {
  const head = t.group
    ? `<tr><th class="rowname" rowspan="2">${esc(t.rowName)}</th><th class="group" colspan="${t.cols.length}">${esc(t.group)}</th></tr>
       <tr>${t.cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`
    : `<tr><th class="rowname">${esc(t.rowName)}</th>${t.cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  const body = t.rows
    .map((r, i) => `<tr><th>${esc(r)}</th>${t.values[i].map((v) => `<td class="num">${formatValue(v, t.decimals)}</td>`).join("")}</tr>`)
    .join("");
  return `<div class="table-scroll"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

function radarSVG(scores) {
  const labels = Object.keys(scores);
  const values = Object.values(scores);
  const size = 300, cx = size / 2, cy = size / 2, radius = 95;
  const angle = (i) => -Math.PI / 2 + (2 * Math.PI * i) / labels.length;
  const point = (i, r) => [cx + r * Math.cos(angle(i)), cy + r * Math.sin(angle(i))];
  const poly = (r) => labels.map((_, i) => point(i, r).map((n) => n.toFixed(1)).join(",")).join(" ");

  const rings = [0.2, 0.4, 0.6, 0.8, 1]
    .map((f) => `<polygon points="${poly(radius * f)}" fill="none" stroke="var(--radar-grid)" stroke-width="1"/>`)
    .join("");
  const spokes = labels
    .map((_, i) => { const [x, y] = point(i, radius); return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--radar-grid)"/>`; })
    .join("");
  const shape = labels.map((_, i) => point(i, radius * values[i]).map((n) => n.toFixed(1)).join(",")).join(" ");
  const text = labels
    .map((label, i) => {
      const [x, y] = point(i, radius + 18);
      const anchor = Math.abs(x - cx) < 4 ? "middle" : x > cx ? "start" : "end";
      return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" dominant-baseline="middle">${esc(label)}</text>`;
    })
    .join("");

  return `<svg class="radar" viewBox="-75 -5 ${size + 150} ${size + 10}" role="img" aria-label="Radar chart of ${esc(labels.join(", "))}">
    ${rings}${spokes}
    <polygon points="${shape}" fill="var(--radar-fill)" stroke="var(--radar-stroke)" stroke-width="2"/>
    ${text}
  </svg>`;
}

function metricRow(name, value, overall = false) {
  return `<div class="metric-row${overall ? " overall" : ""}">
    <span>${esc(name)}</span>
    <div class="bar"><span style="width:${(value * 100).toFixed(0)}%"></span></div>
    <span class="pct">${pct(value)}</span>
  </div>`;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderPreview() {
  const head = state.columns.map((c) => `<th>${esc(c)}</th>`).join("");
  const body = state.rows
    .slice(0, 3)
    .map((r) => `<tr>${r.map((v) => `<td${isNumeric(v) ? ' class="num"' : ""}>${esc(v)}</td>`).join("")}</tr>`)
    .join("");
  $("preview").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  $("dataset-chip").textContent =
    `zomato.csv · ${state.rows.length.toLocaleString("en-US")} rows × ${state.columns.length} columns`;
}

function renderAttributes() {
  const select = $("attr-select");
  const available = state.columns.filter((c) => !state.attributes.includes(c));
  select.innerHTML = `<option value="">Choose an attribute…</option>` +
    available.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  $("attr-chips").innerHTML = state.attributes
    .map((a, i) => `<span class="chip">${esc(a)}<button type="button" data-remove-attr="${i}" aria-label="Remove ${esc(a)}">×</button></span>`)
    .join("");
}

function renderParams() {
  $("diversity-input").value = state.diversity;
  $("num-input").value = state.numTables;
}

function renderList() {
  const el = $("rec-list");
  if (!state.recommend) {
    el.innerHTML = `<h2>Recommendations</h2><p class="subtle">Set parameters and press <b>Recommend</b>.</p>`;
    return;
  }
  const tables = visibleTables();
  const cards = tables
    .map((p, idx) => {
      const key = likeKey(p.query);
      const liked = state.liked.has(key);
      const disliked = state.disliked.has(key);
      const classes = ["rec-card", liked && "liked", disliked && "disliked", state.selected === idx && "selected"]
        .filter(Boolean).join(" ");
      return `<div class="${classes}">
        <p><b>Query:</b> ${esc(p.query)}</p>
        <p><b>Utility Score:</b> ${p.utility}%</p>
        ${pivotTableHTML(p.table)}
        <div class="rec-actions">
          <button type="button" data-view="${idx}">View Details</button>
          <button type="button" data-like="${idx}" class="${liked ? "on-like" : ""}" aria-pressed="${liked}" aria-label="Like">👍</button>
          <button type="button" data-dislike="${idx}" class="${disliked ? "on-dislike" : ""}" aria-pressed="${disliked}" aria-label="Dislike">👎</button>
        </div>
      </div>`;
    })
    .join("");
  el.innerHTML = `<h2>Recommendations (${tables.length})</h2>
    <p class="subtle">Diversity: ${currentResults().diversity}%</p>
    ${cards}`;
}

function renderDetail() {
  const el = $("detail");
  const header = `<h2>Pivot Table Visualization</h2>`;
  const p = state.selected === null ? null : visibleTables()[state.selected];
  if (!p) {
    el.innerHTML = header + (state.recommend
      ? `<p class="subtle">Click <b>View Details</b> on a recommendation.</p>`
      : "");
    return;
  }
  const { insightfulness: ins, interpretability: interp } = currentResults();
  const insOverall = ins["Attribute Significance"] * Math.max(ins["Informativeness"], ins["Trend"], ins["Surprise"]);
  const interpValues = Object.values(interp);
  const interpOverall = interpValues.reduce((a, b) => a + b, 0) / interpValues.length;

  el.innerHTML = `${header}
    <h3 class="detail-query">${esc(p.query)}</h3>
    ${pivotTableHTML(p.table)}
    <h3 class="detail-utility">Utility: ${p.utility}%</h3>
    <div class="metrics">
      <div class="metric-col">
        ${metricRow("Insightfulness", insOverall, true)}
        ${Object.entries(ins).map(([k, v]) => metricRow(k, v)).join("")}
        ${radarSVG(ins)}
      </div>
      <div class="metric-col">
        ${metricRow("Interpretability", interpOverall, true)}
        ${Object.entries(interp).map(([k, v]) => metricRow(k, v)).join("")}
        ${radarSVG(interp)}
      </div>
    </div>`;
}

function sortedQueries(set) {
  const seen = new Set();
  return [...set]
    .map((k) => { const i = k.indexOf("|"); return [Number(k.slice(0, i)), k.slice(i + 1)]; })
    .sort((a, b) => a[0] - b[0] || a[1].localeCompare(b[1]))
    .map(([, q]) => q)
    .filter((q) => !seen.has(q) && seen.add(q));
}

function renderPrefs() {
  const liked = [...(state.rows.length ? [DEFAULT_LIKED_QUERY] : []),
    ...sortedQueries(state.liked).filter((q) => q !== DEFAULT_LIKED_QUERY)];
  const disliked = sortedQueries(state.disliked);
  const t = state.loginTime;
  const stamp = `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")} ` +
    `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}:${String(t.getSeconds()).padStart(2, "0")}`;

  $("prefs").innerHTML = `<h2>User Preferences</h2>
    <div class="pref-meta">
      <p><b>User:</b> guest</p>
      <p><b>Session start:</b> ${stamp}</p>
    </div>
    <div class="pref-box">
      <h4>👍 Liked Tables</h4>
      ${liked.map((q) => `<div class="pref-item like">${esc(q)}</div>`).join("")}
    </div>
    <div class="pref-box">
      <h4>👎 Disliked Tables</h4>
      ${disliked.length
        ? disliked.map((q) => `<div class="pref-item dislike">${esc(q)}</div>`).join("")
        : `<p class="muted"><i>No disliked tables yet</i></p>`}
    </div>`;
}

function renderResults() {
  renderList();
  renderDetail();
  renderPrefs();
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
const LIMITS = { diversity: [10, 100], numTables: [1, 10] };

function setParam(name, value) {
  const [lo, hi] = LIMITS[name];
  const n = Number(value);
  state[name] = Math.min(hi, Math.max(lo, Number.isFinite(n) ? Math.round(n) : state[name]));
  renderParams();
  if (name === "numTables") {
    if (state.selected !== null && state.selected >= state.numTables) state.selected = null;
    renderResults();
  }
}

function toggle(set, other, key) {
  if (set.has(key)) set.delete(key);
  else { set.add(key); other.delete(key); }
}

function bindEvents() {
  document.querySelectorAll("[data-step]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.step;
      setParam(name, state[name] + Number(btn.dataset.delta));
    });
  });
  $("diversity-input").addEventListener("change", (e) => setParam("diversity", e.target.value));
  $("num-input").addEventListener("change", (e) => setParam("numTables", e.target.value));

  $("attr-select").addEventListener("change", (e) => {
    if (e.target.value) state.attributes.push(e.target.value);
    renderAttributes();
  });
  $("attr-chips").addEventListener("click", (e) => {
    const i = e.target.dataset.removeAttr;
    if (i === undefined) return;
    state.attributes.splice(Number(i), 1);
    renderAttributes();
  });

  $("recommend-btn").addEventListener("click", () => {
    state.iteration += 1;
    state.recommend = true;
    state.selected = null;
    renderResults();
  });

  $("rec-list").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const { view, like, dislike } = btn.dataset;
    if (view !== undefined) {
      state.selected = Number(view);
    } else if (like !== undefined) {
      toggle(state.liked, state.disliked, likeKey(visibleTables()[Number(like)].query));
    } else if (dislike !== undefined) {
      toggle(state.disliked, state.liked, likeKey(visibleTables()[Number(dislike)].query));
    } else return;
    renderResults();
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  bindEvents();
  renderParams();
  renderResults();
  $("recommend-btn").disabled = true;
  try {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const [columns, ...rows] = parseCSV(await res.text());
    state.columns = columns;
    state.rows = rows;
    renderPreview();
    renderAttributes();
    $("recommend-btn").disabled = false;
  } catch (err) {
    $("preview").innerHTML = `<p class="muted center">Could not load ${esc(DATA_URL)} (${esc(err.message)}).
      If you opened this file directly, serve the folder instead, e.g. <code>python3 -m http.server</code>.</p>`;
    $("dataset-chip").textContent = "Dataset unavailable";
  }
}

init();
