// SAGE static demo — a port of demo/demo (Streamlit) to plain HTML/JS.
// Recommendations are the precomputed example results from
// demo/demo/components/table_list.py and pivot_table.py.

// Attributes of the example restaurant dataset that the recommendations draw on.
const ATTRIBUTES = ["ID", "City", "Cuisines", "Price Range", "Cost", "Rating", "Has Online Delivery"];

// A few sample rows of the example dataset, shown in the preview.
const SAMPLE_COLUMNS = ["ID", "Name", "City", "Cuisines", "Price Range", "Cost", "Rating", "Has Online Delivery"];
const SAMPLE_ROWS = [
  [2437, "28 Capri Italy", "New Delhi", "Italian", "$$$", 1500, 3.9, "Yes"],
  [2217, "Tamura", "New Delhi", "Japanese", "$$$", 1500, 3.5, "No"],
  [18221572, "Onesta", "Bengaluru", "Pizza, Cafe, Italian", "$$", 600, 4.3, "Yes"],
  [18388642, "Grandmama's Cafe", "Mumbai", "Continental, Italian", "$$$", 1100, 3.8, "No"],
  [18025105, "Sinyora's", "New Delhi", "Italian", "$$", 500, 4.0, "Yes"],
  [18075122, "The Fusion Kitchen", "Mumbai", "North Indian, Italian, Chinese, Mexican", "$$$", 1000, 4.7, "Yes"],
  [305125, "Amalfi", "New Delhi", "Italian", "$$$", 1800, 3.9, "No"],
  [18305628, "Eat Street", "Bengaluru", "North Indian, Chinese, Italian, Street Food, Desserts", "$", 400, 4.3, "Yes"],
  [18245248, "Pasta Hub", "New Delhi", "Italian", "$", 200, 3.0, "No"],
  [18422898, "Hoot", "Bengaluru", "Continental, Italian, North Indian", "$$$", 1400, 3.9, "No"],
];
const PREVIEW_VISIBLE_ROWS = 3;

// ---------------------------------------------------------------------------
// Demo results
// table: { rowName, rows, group (column-level header, optional), cols, values[row][col], decimals }
//
// Per-table scores follow SAGE's utility (src/utility/utility.py):
//   utility = 0.5 * insightfulness + 0.5 * interpretability
//   insightfulness   = Attribute Significance * max(Informativeness, Trend, Surprise)
//   interpretability = mean(Density, Semantic Validity, Conciseness)
// Informativeness / Trend / Surprise / Conciseness are computed from each table's values with
// SAGE's formulas, calibrated so "COUNT(ID) GROUP BY City, Cuisines" keeps its original demo scores.
// Semantic Validity reflects how well the aggregate fits the measure (AVG > COUNT > MAX > MIN), and
// Attribute Significance (LLM-derived in SAGE) is set so each table reproduces its fixed utility.
// ---------------------------------------------------------------------------
function scores(significance, informativeness, trend, surprise, semanticValidity, conciseness) {
  return {
    insightfulness: {
      "Attribute Significance": significance,
      "Informativeness": informativeness,
      "Trend": trend,
      "Surprise": surprise,
    },
    interpretability: { "Density": 1.00, "Semantic Validity": semanticValidity, "Conciseness": conciseness },
  };
}

const CUISINES = ["Italian", "French", "Japanese"];
const CITIES = ["Bengaluru", "New Delhi", "Mumbai"];

const PIVOT_TABLES_ITER1 = [
  {
    query: "AVG(Rating) GROUP BY Price Range, Cuisines",
    scores: scores(0.68, 0.25, 0.51, 1.00, 1.00, 0.95),
    utility: 83,
    table: {
      rowName: "Cuisines", rows: CUISINES, group: "Price Range", cols: ["$", "$$", "$$$"],
      values: [[3.3, 3.5, 4.5], [3.6, 4.5, 3.4], [4.5, 3.2, 4.2]], decimals: 1,
    },
  },
  {
    query: "COUNT(ID) GROUP BY City, Cuisines",
    scores: scores(0.87, 0.75, 0.55, 0.60, 0.90, 0.95),
    utility: 80,
    table: {
      rowName: "Cuisines", rows: CUISINES, group: "City", cols: CITIES,
      values: [[40, 20, 50], [30, 10, 40], [20, 40, 60]],
    },
  },
  {
    query: "MAX(Cost) GROUP BY Cuisines",
    scores: scores(0.24, 0.32, 0.00, 0.90, 0.80, 0.98),
    utility: 57,
    table: { rowName: "Cuisines", rows: CUISINES, cols: ["Cost"], values: [[1300], [1200], [1000]] },
  },
  {
    query: "AVG(Rating) GROUP BY City",
    scores: scores(0.56, 0.09, 0.00, 0.90, 1.00, 0.98),
    utility: 75,
    table: { rowName: "City", rows: CITIES, cols: ["Rating"], values: [[4.4], [4.5], [4.2]], decimals: 1 },
  },
  {
    query: "COUNT(ID) GROUP BY Cuisines, Has Online Delivery",
    scores: scores(0.51, 0.87, 0.00, 0.45, 0.90, 0.97),
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
    scores: scores(0.72, 0.11, 0.55, 1.00, 1.00, 0.95),
    utility: 85,
    table: {
      rowName: "Cuisines", rows: CUISINES, group: "City", cols: CITIES,
      values: [[3.7, 3.8, 4.1], [4.0, 4.7, 4.2], [3.9, 4.5, 4.3]], decimals: 1,
    },
  },
  {
    query: "AVG(Cost) BY City",
    scores: scores(0.79, 0.66, 0.00, 0.90, 1.00, 0.98),
    utility: 85,
    table: { rowName: "City", rows: CITIES, cols: ["Cost"], values: [[800], [1200], [1400]] },
  },
  {
    query: "MIN(Rating) By City, Price Range",
    scores: scores(0.59, 0.37, 0.99, 1.00, 0.60, 0.95),
    utility: 72,
    table: {
      rowName: "City", rows: CITIES, group: "Price Range", cols: ["$", "$$", "$$$"],
      values: [[3.3, 2.8, 2.6], [2.3, 1.0, 2.5], [3.2, 3.0, 2.8]], decimals: 1,
    },
  },
];

const ITER_RESULTS = {
  1: { tables: PIVOT_TABLES_ITER1, diversity: 32 },
  2: { tables: PIVOT_TABLES_ITER2, diversity: 30 },
};

// Guide-mode explanations, keyed by the data-tip of each "?" badge.
const GUIDE_TIPS = {
  dataset: "A few rows of the example restaurant dataset. SAGE searches over pivot tables built from its columns: which to group by, which to aggregate, and how (AVG, COUNT, MAX, …).",
  diversity: "How different the recommended tables should be from one another. Higher values favor tables over different attributes and aggregations.",
  numTables: "How many pivot tables to recommend.",
  attributes: "Attributes you care about. SAGE favors tables that use them.",
  recommend: "Runs SAGE with these settings. In this demo the results are precomputed.",
  recs: "The recommended pivot tables, chosen to be both high-quality and varied.",
  achievedDiversity: "How varied the recommended set actually is.",
  query: "The pivot table's definition: an aggregate of one attribute, grouped by one or two others.",
  utility: "Overall quality of a table: the average of its insightfulness and interpretability.",
  actions: "View Details opens the table's score breakdown. 👍 / 👎 records your feedback.",
  detail: "Score breakdown for the selected table.",
  insightfulness: "How much the table reveals: Attribute Significance × the strongest of Informativeness, Trend and Surprise.",
  "Attribute Significance": "How meaningful the chosen attributes are for analysis. In SAGE this is judged with an LLM.",
  "Informativeness": "How much the values differ across rows and columns.",
  "Trend": "How strongly rows or columns move together.",
  "Surprise": "Whether a value stands out sharply from the rest.",
  interpretability: "How easy the table is to read: the average of Density, Semantic Validity and Conciseness.",
  "Density": "Share of cells that have a value. No empty cells means 100%.",
  "Semantic Validity": "Whether the aggregation makes sense for the measure, e.g. AVG of ratings rather than MIN.",
  "Conciseness": "Smaller tables are easier to take in, so they score higher.",
  prefs: "Your feedback during this session.",
  liked: "Tables you marked 👍. In SAGE, feedback guides the next round of recommendations.",
  disliked: "Tables you marked 👎. In SAGE, feedback guides the next round of recommendations.",
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  diversity: 20,
  numTables: 5,
  attributes: [],
  iteration: 0,
  recommend: false,
  selected: null,
  liked: new Set(),     // queries
  disliked: new Set(),
  loginTime: new Date(),
};

// First click shows iteration-1 results; every later click shows iteration-2 results.
const currentResults = () => ITER_RESULTS[state.iteration <= 1 ? 1 : 2];
const visibleTables = () => currentResults().tables.slice(0, state.numTables);

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

function formatValue(v, decimals) {
  return decimals ? v.toFixed(decimals) : v.toLocaleString("en-US");
}

const pct = (v) => `${Math.round(v * 100)}%`;

// "?" badge, visible only in guide mode.
const tip = (key) => `<button type="button" class="guide-dot" data-tip="${esc(key)}" aria-label="What is this?">?</button>`;

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

function metricRow(name, value, overall = false, tipKey = name) {
  return `<div class="metric-row${overall ? " overall" : ""}">
    <span>${esc(name)}${tip(tipKey)}</span>
    <div class="bar"><span style="width:${(value * 100).toFixed(0)}%"></span></div>
    <span class="pct">${pct(value)}</span>
  </div>`;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderPreview() {
  const head = SAMPLE_COLUMNS.map((c) => `<th>${esc(c)}</th>`).join("");
  const body = SAMPLE_ROWS
    .map((r) => `<tr>${r.map((v, i) => typeof v === "number"
      ? `<td class="num">${SAMPLE_COLUMNS[i] === "Rating" ? v.toFixed(1) : v}</td>`
      : `<td>${esc(v)}</td>`).join("")}</tr>`)
    .join("");
  $("preview").innerHTML = `<div class="preview-head"><h3>Example dataset${tip("dataset")}</h3>
      <span class="muted">${SAMPLE_ROWS.length} sample rows · scroll for more</span></div>
    <div class="table-scroll preview-scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;

  // Show the header plus the first few rows; the rest scroll.
  const scroller = $("preview").querySelector(".preview-scroll");
  const rows = scroller.querySelectorAll("tbody tr");
  const visible = [...rows].slice(0, PREVIEW_VISIBLE_ROWS).reduce((h, r) => h + r.offsetHeight, 0);
  scroller.style.maxHeight = `${scroller.querySelector("thead").offsetHeight + visible + 1}px`;
}

function renderAttributes() {
  const select = $("attr-select");
  const available = ATTRIBUTES.filter((c) => !state.attributes.includes(c));
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
    el.innerHTML = `<h2>Recommendations${tip("recs")}</h2><p class="subtle">Set parameters and press <b>Recommend</b>.</p>`;
    return;
  }
  const tables = visibleTables();
  const cards = tables
    .map((p, idx) => {
      const liked = state.liked.has(p.query);
      const disliked = state.disliked.has(p.query);
      const classes = ["rec-card", liked && "liked", disliked && "disliked", state.selected === idx && "selected"]
        .filter(Boolean).join(" ");
      return `<div class="${classes}">
        <p><b>Query:</b> ${esc(p.query)}${idx === 0 ? tip("query") : ""}</p>
        <p><b>Utility Score:</b> ${p.utility}%${idx === 0 ? tip("utility") : ""}</p>
        ${pivotTableHTML(p.table)}
        ${idx === 0 ? `<div class="actions-tip">${tip("actions")}</div>` : ""}
        <div class="rec-actions">
          <button type="button" data-view="${idx}">View Details</button>
          <button type="button" data-like="${idx}" class="${liked ? "on-like" : ""}" aria-pressed="${liked}" aria-label="Like">👍</button>
          <button type="button" data-dislike="${idx}" class="${disliked ? "on-dislike" : ""}" aria-pressed="${disliked}" aria-label="Dislike">👎</button>
        </div>
      </div>`;
    })
    .join("");
  el.innerHTML = `<h2>Recommendations (${tables.length})${tip("recs")}</h2>
    <p class="subtle">Diversity: ${currentResults().diversity}%${tip("achievedDiversity")}</p>
    ${cards}`;
}

function renderDetail() {
  const el = $("detail");
  const header = `<h2>Pivot Table Visualization${tip("detail")}</h2>`;
  const p = state.selected === null ? null : visibleTables()[state.selected];
  if (!p) {
    el.innerHTML = header + (state.recommend
      ? `<p class="subtle">Click <b>View Details</b> on a recommendation.</p>`
      : "");
    return;
  }
  const { insightfulness: ins, interpretability: interp } = p.scores;
  const insOverall = ins["Attribute Significance"] * Math.max(ins["Informativeness"], ins["Trend"], ins["Surprise"]);
  const interpValues = Object.values(interp);
  const interpOverall = interpValues.reduce((a, b) => a + b, 0) / interpValues.length;

  el.innerHTML = `${header}
    <h3 class="detail-query">${esc(p.query)}</h3>
    ${pivotTableHTML(p.table)}
    <h3 class="detail-utility">Utility: ${p.utility}%${tip("utility")}</h3>
    <div class="metrics">
      <div class="metric-col">
        ${metricRow("Insightfulness", insOverall, true, "insightfulness")}
        ${Object.entries(ins).map(([k, v]) => metricRow(k, v)).join("")}
        ${radarSVG(ins)}
      </div>
      <div class="metric-col">
        ${metricRow("Interpretability", interpOverall, true, "interpretability")}
        ${Object.entries(interp).map(([k, v]) => metricRow(k, v)).join("")}
        ${radarSVG(interp)}
      </div>
    </div>`;
}

function renderPrefs() {
  const liked = [...state.liked];
  const disliked = [...state.disliked];
  const t = state.loginTime;
  const stamp = `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")} ` +
    `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}:${String(t.getSeconds()).padStart(2, "0")}`;

  $("prefs").innerHTML = `<h2>User Preferences${tip("prefs")}</h2>
    <div class="pref-meta">
      <p><b>User:</b> guest</p>
      <p><b>Session start:</b> ${stamp}</p>
    </div>
    <div class="pref-box">
      <h4>👍 Liked Tables${tip("liked")}</h4>
      ${liked.length
        ? liked.map((q) => `<div class="pref-item like">${esc(q)}</div>`).join("")
        : `<p class="muted"><i>No liked tables yet</i></p>`}
    </div>
    <div class="pref-box">
      <h4>👎 Disliked Tables${tip("disliked")}</h4>
      ${disliked.length
        ? disliked.map((q) => `<div class="pref-item dislike">${esc(q)}</div>`).join("")
        : `<p class="muted"><i>No disliked tables yet</i></p>`}
    </div>`;
}

function renderResults() {
  renderList();
  renderDetail();
  renderPrefs();
  hideTip();
  updateCoach();
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
      toggle(state.liked, state.disliked, visibleTables()[Number(like)].query);
    } else if (dislike !== undefined) {
      toggle(state.disliked, state.liked, visibleTables()[Number(dislike)].query);
    } else return;
    renderResults();
  });
}

// ---------------------------------------------------------------------------
// Guide mode: "?" popups on every part of the page, plus a next-step coach
// ---------------------------------------------------------------------------
const STORAGE = { on: "sage-guide-on", hintSeen: "sage-guide-hint-seen" };
// intro: how many of the read-only intro steps (preview data, set parameters) have been passed with "Next".
const guide = { on: false, tipFor: null, tipPinned: false, coachHiddenFor: null, intro: 0 };

function storageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
}

// place: "right" puts the popup beside the target when there is room; otherwise below
// (or above when "above" is preferred or there's no room below).
function placeNear(el, target, gap = 10, place = "below") {
  const r = target.getBoundingClientRect();
  const w = el.offsetWidth, h = el.offsetHeight;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  let side;
  if (place === "right" && r.right + gap + w <= window.innerWidth - 8) side = "right";
  else if (place === "above" || place === "right") side = r.top - gap - h >= 0 ? "above" : "below";
  else side = r.bottom + gap + h <= window.innerHeight || r.top - gap - h < 0 ? "below" : "above";

  let left, top;
  if (side === "right") {
    left = r.right + gap;
    // top-aligned so a tall target (a whole panel) gets the popup beside its heading
    top = clamp(r.top, 8, window.innerHeight - h - 8);
    el.style.setProperty("--arrow-y", `${clamp(r.top + Math.min(r.height / 2, 32) - top, 12, h - 12)}px`);
  } else {
    left = clamp(r.left + r.width / 2 - 28, 8, window.innerWidth - w - 8);
    top = side === "below" ? r.bottom + gap : r.top - gap - h;
    el.style.setProperty("--arrow-x", `${clamp(r.left + r.width / 2 - left, 12, w - 12)}px`);
  }
  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
  ["below", "above", "right"].forEach((c) => el.classList.toggle(c, c === side));
}

function showTip(dot, pinned) {
  const pop = $("guide-pop");
  pop.textContent = GUIDE_TIPS[dot.dataset.tip] || "";
  pop.hidden = false;
  placeNear(pop, dot);
  guide.tipFor = dot;
  guide.tipPinned = pinned;
}

function hideTip() {
  $("guide-pop").hidden = true;
  guide.tipFor = null;
  guide.tipPinned = false;
}

// The next thing to do in the demo loop, based on the current state.
// target is outlined; anchor (defaults to target) is what the popup points at.
const COACH_TOTAL = 6;
function coachStep() {
  if (!state.recommend) {
    if (guide.intro < 1) {
      return { id: "preview", n: 1, target: $("preview"), next: true,
        text: "<b>Preview your data.</b> These are sample rows of the example dataset; scroll the table to see more." };
    }
    if (guide.intro < 2) {
      return { id: "params", n: 2, target: document.querySelector(".panel-params"), place: "right", next: true,
        text: "<b>Set the parameters:</b> diversity, number of tables and, optionally, preferred attributes." };
    }
    return { id: "recommend", n: 3, target: $("recommend-btn"),
      text: "Click <b>Recommend</b> to get pivot table recommendations." };
  }
  if (state.iteration === 1) {
    if (state.selected === null) {
      return { id: "view", n: 4, target: document.querySelector('[data-view="0"]'),
        text: "Click <b>View Details</b> to see how this table is scored." };
    }
    const hasFeedback = visibleTables().some((p) => state.liked.has(p.query) || state.disliked.has(p.query));
    if (!hasFeedback) {
      return { id: "feedback", n: 5, target: document.querySelector(`[data-like="${state.selected}"]`),
        text: "Like 👍 or dislike 👎 tables to tell SAGE what you prefer." };
    }
    return { id: "again", n: 6, target: $("recommend-btn"),
      text: "Click <b>Recommend</b> again. In SAGE, your feedback guides the next round." };
  }
  return { id: "done", target: document.querySelector(".switch"), label: "All set",
    text: "That's the full loop. Tap any <b>?</b> to learn what each part means." };
}

function isOnScreen(target) {
  const r = target.getBoundingClientRect();
  const scroller = target.closest(".panel-tall");
  const box = scroller ? scroller.getBoundingClientRect() : { top: 0, bottom: window.innerHeight };
  return r.bottom > Math.max(box.top, 0) && r.top < Math.min(box.bottom, window.innerHeight);
}

function updateCoach() {
  const coach = $("guide-coach");
  document.querySelectorAll(".guide-target").forEach((el) => el.classList.remove("guide-target"));
  const step = guide.on ? coachStep() : null;
  if (!step || !step.target || guide.coachHiddenFor === step.id) {
    coach.hidden = true;
    return;
  }
  const anchor = step.anchor || step.target;
  step.target.classList.add("guide-target");
  $("coach-step").textContent = step.label || `Step ${step.n} of ${COACH_TOTAL}`;
  $("coach-text").innerHTML = step.text;
  $("coach-next").hidden = !step.next;
  coach.hidden = !isOnScreen(anchor);
  if (!coach.hidden) placeNear(coach, anchor, 12, step.place);
}

function setGuide(on) {
  guide.on = on;
  guide.coachHiddenFor = null;
  document.body.classList.toggle("guide-on", on);
  $("guide-toggle").checked = on;
  storageSet(STORAGE.on, on ? "1" : "0");
  hideTip();
  updateCoach();
}

function dismissHint() {
  $("guide-hint").hidden = true;
  storageSet(STORAGE.hintSeen, "1");
}

function bindGuide() {
  $("guide-toggle").addEventListener("change", (e) => { dismissHint(); setGuide(e.target.checked); });
  $("guide-hint-on").addEventListener("click", () => { dismissHint(); setGuide(true); });
  $("guide-hint-close").addEventListener("click", dismissHint);
  $("coach-next").addEventListener("click", () => {
    guide.intro += 1;
    updateCoach();
  });
  $("coach-close").addEventListener("click", () => {
    guide.coachHiddenFor = coachStep().id;
    updateCoach();
  });

  const canHover = window.matchMedia("(hover: hover)").matches;
  document.addEventListener("click", (e) => {
    const dot = e.target.closest(".guide-dot");
    if (dot) {
      if (guide.tipFor === dot && guide.tipPinned) hideTip();
      else showTip(dot, true);
    } else if (!e.target.closest("#guide-pop")) {
      hideTip();
    }
  });
  if (canHover) {
    document.addEventListener("mouseover", (e) => {
      const dot = e.target.closest(".guide-dot");
      if (dot && !guide.tipPinned) showTip(dot, false);
    });
    document.addEventListener("mouseout", (e) => {
      const dot = e.target.closest(".guide-dot");
      if (dot && guide.tipFor === dot && !guide.tipPinned) hideTip();
    });
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });

  let frame = 0;
  const reposition = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      if (guide.tipFor) {
        if (guide.tipFor.isConnected && isOnScreen(guide.tipFor)) placeNear($("guide-pop"), guide.tipFor);
        else hideTip();
      }
      updateCoach();
    });
  };
  window.addEventListener("resize", reposition);
  document.addEventListener("scroll", reposition, true);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function init() {
  bindEvents();
  bindGuide();
  renderPreview();
  renderAttributes();
  renderParams();
  renderResults();
  setGuide(storageGet(STORAGE.on) === "1");
  if (!guide.on && storageGet(STORAGE.hintSeen) !== "1") $("guide-hint").hidden = false;
}

init();
