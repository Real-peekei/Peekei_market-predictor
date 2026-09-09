const API_BASE = "http://127.0.0.1:8001";

// Same duration-in-minutes mapping as backend/intervals.py, kept in sync
// so the dropdown and "All Timeframes" results always sort chronologically
// (1m before 5m before 1h before 1d ...) rather than alphabetically, even
// if the API ever returns them out of order.
const INTERVAL_MINUTES = {
  "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "45m": 45,
  "60m": 60, "1h": 60, "90m": 90, "2h": 120, "4h": 240, "8h": 480,
  "1d": 1440, "1wk": 1440 * 7, "1mo": 1440 * 30,
};
function sortIntervals(intervals) {
  return [...intervals].sort((a, b) => (INTERVAL_MINUTES[a] ?? 1e9) - (INTERVAL_MINUTES[b] ?? 1e9));
}

const el = (id) => document.getElementById(id);
const assetSelect = el("assetSelect");
const intervalSelect = el("intervalSelect");
const refreshBtn = el("refreshBtn");
const predictArea = el("predictArea");
const emptyState = el("emptyState");

let assetLabels = {}; // asset_key -> symbol, for display purposes

async function checkApi() {
  try {
    const res = await fetch(`${API_BASE}/`);
    if (!res.ok) throw new Error();
    el("apiDot").classList.add("live");
    el("apiStatusText").textContent = "API CONNECTED";
    return true;
  } catch {
    el("apiStatusText").textContent = "API OFFLINE — run: uvicorn server:app --port 8001";
    return false;
  }
}

async function loadAssets() {
  try {
    const res = await fetch(`${API_BASE}/api/assets`);
    if (!res.ok) throw new Error(`assets request failed (${res.status})`);
    const data = await res.json();
    const assets = data.assets || [];

    assetLabels = Object.fromEntries(assets.map((a) => [a.asset_key, a.symbol]));
    assetSelect.innerHTML = assets
      .map((a) => `<option value="${a.asset_key}">${a.symbol}</option>`)
      .join("");

    if (assets.length > 0) {
      await loadIntervals(assets[0].asset_key);
    }
  } catch (err) {
    console.error("Failed to load assets", err);
    el("apiStatusText").textContent = "No trained assets yet — run train_model.py or train_all.py";
  }
}

async function loadIntervals(assetKey) {
  try {
    const res = await fetch(`${API_BASE}/api/intervals?asset=${encodeURIComponent(assetKey)}`);
    const data = await res.json();
    const intervals = sortIntervals(data.intervals || []); // chronological, not alphabetical
    intervalSelect.innerHTML = intervals.map((i) => `<option value="${i}">${i.toUpperCase()}</option>`).join("");
  } catch (err) {
    console.error("Failed to load intervals", err);
  }
}

assetSelect.addEventListener("change", () => {
  loadIntervals(assetSelect.value);
  predictArea.classList.add("hidden");
  allArea.classList.add("hidden");
  emptyState.classList.remove("hidden");
});

function pct(x, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}

function renderMetrics(metrics) {
  const rows = [
    { label: "STRATEGY RETURN", val: pct(metrics.strategy_total_return), sign: metrics.strategy_total_return },
    { label: "BUY & HOLD RETURN", val: pct(metrics.buyhold_total_return), sign: metrics.buyhold_total_return },
    { label: "STRATEGY SHARPE", val: metrics.strategy_sharpe.toFixed(2), sign: metrics.strategy_sharpe },
    { label: "BUY & HOLD SHARPE", val: metrics.buyhold_sharpe.toFixed(2), sign: metrics.buyhold_sharpe },
    { label: "DIRECTION HIT RATE", val: pct(metrics.direction_hit_rate), sign: metrics.direction_hit_rate - 0.5 },
    { label: "MAX DRAWDOWN", val: pct(metrics.strategy_max_drawdown), sign: metrics.strategy_max_drawdown },
    { label: "BARS TESTED", val: metrics.n_bars_tested, sign: 0 },
  ];
  el("metricsGrid").innerHTML = rows.map((r) => `
    <div class="metric-cell">
      <span class="label">${r.label}</span>
      <span class="val ${r.sign > 0 ? "pos" : r.sign < 0 ? "neg" : ""}">${r.val}</span>
    </div>`).join("");
}

function renderEquityChart(dates, strategy, buyhold) {
  const svg = el("equityChart");
  const W = 600, H = 220, PAD = 10;
  const all = [...strategy, ...buyhold];
  const min = Math.min(...all), max = Math.max(...all);
  const range = max - min || 1;

  const toPoints = (series) => series.map((v, i) => {
    const x = PAD + (i / (series.length - 1)) * (W - 2 * PAD);
    const y = H - PAD - ((v - min) / range) * (H - 2 * PAD);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  svg.innerHTML = `
    <polyline points="${toPoints(buyhold)}" fill="none" stroke="#7A7A6E" stroke-width="1.5" />
    <polyline points="${toPoints(strategy)}" fill="none" stroke="#E8C165" stroke-width="2" />
  `;
}

async function runPrediction() {
  const asset = assetSelect.value;
  const interval = intervalSelect.value;
  if (!asset || !interval) return;

  refreshBtn.disabled = true;
  refreshBtn.textContent = "RUNNING…";

  try {
    const [predRes, backtestRes] = await Promise.all([
      fetch(`${API_BASE}/api/predict?asset=${encodeURIComponent(asset)}&interval=${interval}`),
      fetch(`${API_BASE}/api/backtest?asset=${encodeURIComponent(asset)}&interval=${interval}`),
    ]);

    if (!predRes.ok) {
      const err = await predRes.json().catch(() => ({}));
      throw new Error(err.detail || `Prediction request failed (${predRes.status})`);
    }
    const pred = await predRes.json();

    emptyState.classList.add("hidden");
    allArea.classList.add("hidden");
    predictArea.classList.remove("hidden");

    el("lastClose").textContent = pred.last_close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    el("asOf").textContent = pred.as_of;

    const badge = el("directionBadge");
    badge.textContent = pred.direction === "up" ? "▲ UP" : "▼ DOWN";
    badge.className = `direction-badge ${pred.direction}`;

    el("confFill").style.width = pct(pred.confidence, 0);
    el("confVal").textContent = pct(pred.confidence, 0);
    el("probUp").textContent = pct(pred.prob_up);
    el("probDown").textContent = pct(pred.prob_down);

    el("rangeLow").textContent = pred.expected_range.low_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    el("rangeHigh").textContent = pred.expected_range.high_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const span = pred.expected_range.high_return - pred.expected_range.low_return;
    const posInSpan = span > 0 ? (0 - pred.expected_range.low_return) / span : 0.5;
    el("rangeFill").style.left = `${Math.min(100, Math.max(0, posInSpan * 100))}%`;

    const trustHeadline = el("trustHeadline");
    trustHeadline.textContent = pred.trust.headline;
    trustHeadline.className = `trust-headline ${pred.trust.level}`;
    el("trustExplain").textContent = pred.trust.explanation || "";

    if (backtestRes.ok) {
      const bt = await backtestRes.json();
      renderEquityChart(bt.dates, bt.strategy_equity, bt.buyhold_equity);
      renderMetrics(bt.metrics);
    } else {
      el("metricsGrid").innerHTML = `<div class="metric-cell"><span class="label">BACKTEST</span><span class="val">not available</span></div>`;
    }
  } catch (err) {
    alert(`Request failed: ${err.message}`);
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = "RUN PREDICTION";
  }
}

refreshBtn.addEventListener("click", runPrediction);

// ---------------- all-timeframes mode ----------------
const allBtn = el("allBtn");
const allArea = el("allArea");
const allList = el("allList");

async function runAllTimeframes() {
  const asset = assetSelect.value;
  if (!asset) return;

  allBtn.disabled = true;
  allBtn.textContent = "RUNNING…";

  try {
    const res = await fetch(`${API_BASE}/api/predict-all?asset=${encodeURIComponent(asset)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();

    predictArea.classList.add("hidden");
    emptyState.classList.add("hidden");
    allArea.classList.remove("hidden");

    if (data.predictions.length === 0) {
      allList.innerHTML = `<p class="trust-explain">No trained timeframes yet — train at least one first.</p>`;
      return;
    }

    // sort chronologically client-side too, defensively - the API already
    // does this, but this guarantees correct display order regardless
    const sorted = [...data.predictions].sort(
      (a, b) => (INTERVAL_MINUTES[a.interval] ?? 1e9) - (INTERVAL_MINUTES[b.interval] ?? 1e9)
    );

    allList.innerHTML = sorted.map((p) => `
      <div class="all-row">
        <span class="interval-tag">${p.interval.toUpperCase()}</span>
        <span class="dir-tag ${p.direction}">${p.direction === "up" ? "▲ UP" : "▼ DOWN"} ${pct(p.confidence, 0)}</span>
        <span class="range-txt">${p.expected_range.low_price.toLocaleString(undefined, {minimumFractionDigits:2})} – ${p.expected_range.high_price.toLocaleString(undefined, {minimumFractionDigits:2})}</span>
        <span class="trust-tag ${p.trust.level}">${p.trust.level.replace("_", " ")}</span>
      </div>`).join("");
  } catch (err) {
    alert(`Request failed: ${err.message}`);
  } finally {
    allBtn.disabled = false;
    allBtn.textContent = "ALL TIMEFRAMES";
  }
}

allBtn.addEventListener("click", runAllTimeframes);

(async function init() {
  const ok = await checkApi();
  if (ok) await loadAssets();
})();
