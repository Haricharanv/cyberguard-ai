// frontend/app.js - CyberGuard AI Frontend Client

const API_BASE = "";

// Elements
const logInput = document.getElementById("log-input");
const btnAnalyze = document.getElementById("btn-analyze");
const btnClean = document.getElementById("btn-sample-clean");
const btnAttack = document.getElementById("btn-simulate-attack");
const metricTotal = document.getElementById("metric-total");
const metricAnomalies = document.getElementById("metric-anomalies");
const metricRate = document.getElementById("metric-rate");
const metricCritical = document.getElementById("metric-critical");
const metricLatency = document.getElementById("metric-latency");
const timerLabel = document.getElementById("analysis-timer");
const resultsContainer = document.getElementById("results-container");
const resultsPlaceholder = document.getElementById("results-table-placeholder");
const resultsContent = document.getElementById("results-content");
const eventsTbody = document.getElementById("events-tbody");
const heatmapContainer = document.getElementById("heatmap-container");
const sessionBadgeContainer = document.getElementById("session-badge-container");

// Standard Clean HDFS Log lines for demo
const CLEAN_HDFS_SAMPLE = [
  "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
  "081109 203519 145 INFO dfs.DataNode$BlockReceiver: Received block blk_-1608999687 of size 91178 from /10.250.19.102",
  "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687 terminating",
  "081109 203519 146 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687 of size 91178 from /10.250.19.102",
  "081109 203520 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.250.19.102:54107 dest: /10.250.19.102:50010",
  "081109 203521 150 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1608999687 terminating",
  "081109 203522 152 INFO dfs.DataNode$BlockReceiver: Verification succeeded for blk_-1608999687",
  "081109 203523 155 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.19.102:50010 is added to blk_-1608999687 size 91178"
].join("\n");

// Load Stats on init
async function refreshStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) return;
    const data = await res.json();
    metricTotal.textContent = data.total_analyzed.toLocaleString();
    metricAnomalies.textContent = data.anomalies_detected.toLocaleString();
    metricRate.textContent = `Rate: ${data.anomaly_rate_percent}%`;
    metricCritical.textContent = data.critical_alerts.toLocaleString();
    metricLatency.innerHTML = `${data.avg_latency_ms} <span class="unit">ms</span>`;
  } catch (e) {
    console.warn("API not reachable yet", e);
  }
}

// Preset Handlers
btnClean.addEventListener("click", () => {
  logInput.value = CLEAN_HDFS_SAMPLE;
});

btnAttack.addEventListener("click", async () => {
  btnAttack.disabled = true;
  btnAttack.textContent = "Loading Attack Vector...";
  try {
    const res = await fetch(`${API_BASE}/api/simulate-attack`);
    const data = await res.json();
    logInput.value = data.logs.join("\n");
  } catch (err) {
    alert("Backend starting up... please wait a few seconds.");
  } finally {
    btnAttack.disabled = false;
    btnAttack.textContent = "🚨 Simulate Attack";
  }
});

// Run Deep Learning Inference
btnAnalyze.addEventListener("click", async () => {
  const text = logInput.value.trim();
  if (!text) {
    alert("Please enter or load log lines to analyze.");
    return;
  }

  const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  btnAnalyze.disabled = true;
  btnAnalyze.innerHTML = `<span>⏳ Analyzing with Bi-LSTM...</span>`;
  timerLabel.textContent = "Processing...";

  try {
    const tStart = performance.now();
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_logs: lines, session_id: "demo-session" })
    });

    if (!res.ok) throw new Error("Inference failed.");
    const data = await res.json();
    const elapsed = Math.round(performance.now() - tStart);

    timerLabel.textContent = `Completed in ${data.latency_ms} ms (Total roundtrip: ${elapsed} ms)`;
    renderResults(data);
    refreshStats();
  } catch (err) {
    alert("Error executing model inference. Verify backend server is running.");
    console.error(err);
  } finally {
    btnAnalyze.disabled = false;
    btnAnalyze.innerHTML = `<span>⚡ Run Deep Learning Analysis</span>`;
  }
});

// Render Results Table & Heatmap
function renderResults(data) {
  resultsPlaceholder.style.display = "none";
  resultsContent.style.display = "block";
  eventsTbody.innerHTML = "";

  // Session Badge
  sessionBadgeContainer.innerHTML = data.anomaly_detected
    ? `<span class="tag-sev sev-${data.highest_severity}">SESSION THREAT DETECTED • ${data.highest_severity}</span>`
    : `<span class="tag-sev sev-NORMAL">SESSION CLEAN • NORMAL</span>`;

  data.events.forEach((evt, idx) => {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    tr.innerHTML = `
      <td>${evt.line_no}</td>
      <td style="color: #cbd5e1; word-break: break-all;">${escapeHtml(evt.raw_text)}</td>
      <td><span class="tag-gray">T_${evt.template_id}</span></td>
      <td><span class="tag-sev sev-${evt.severity}">${evt.severity}</span></td>
      <td style="font-family: var(--font-mono);">${(evt.anomaly_score * 100).toFixed(1)}%</td>
      <td><button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;">Inspect XAI</button></td>
    `;

    // Row click renders its token attention weights
    tr.addEventListener("click", () => {
      document.querySelectorAll(".events-table tbody tr").forEach(r => r.style.background = "");
      tr.style.background = "rgba(59, 130, 246, 0.15)";
      renderHeatmap(evt);
    });

    eventsTbody.appendChild(tr);
  });

  // Default heatmap view: inspect the first anomalous event, or the last event
  const targetEvt = data.events.find(e => e.is_anomaly) || data.events[data.events.length - 1];
  if (targetEvt) {
    renderHeatmap(targetEvt);
  }
}

// Render Attention Weights Heatmap Bar
function renderHeatmap(evt) {
  heatmapContainer.innerHTML = "";
  const weights = evt.attention_attribution || [];

  if (weights.length === 0) {
    heatmapContainer.innerHTML = `<span style="color: var(--text-muted); font-size: 12px;">No attention context available.</span>`;
    return;
  }

  weights.forEach((w, i) => {
    const cell = document.createElement("div");
    cell.className = "heat-cell";

    // Dynamic color gradient based on attention weight (Blue -> Purple -> Red)
    const intensity = Math.min(1.0, w * 3.5); // scale for visibility
    const r = Math.round(239 * intensity);
    const g = Math.round(68 * (1 - intensity));
    const b = Math.round(246 * (1 - intensity * 0.5));
    const alpha = Math.max(0.2, intensity);

    cell.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    cell.style.color = intensity > 0.4 ? "#fff" : "#cbd5e1";
    cell.title = `Token Step t-${weights.length - 1 - i}: Attention Weight ${(w * 100).toFixed(2)}%`;

    cell.innerHTML = `
      <span class="heat-label">t-${weights.length - 1 - i}</span>
      <strong>${(w * 100).toFixed(0)}%</strong>
    `;

    heatmapContainer.appendChild(cell);
  });
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Initial setup
logInput.value = CLEAN_HDFS_SAMPLE;
refreshStats();
