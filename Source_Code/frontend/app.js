// frontend/app.js - CyberGuard AI SOC Dashboard Client

const API_BASE = "";

// Elements
const logInput = document.getElementById("log-input");
const btnAnalyze = document.getElementById("btn-analyze");
const btnAnalyzeBatch = document.getElementById("btn-analyze-batch");
const btnClean = document.getElementById("btn-sample-clean");
const btnAttack = document.getElementById("btn-simulate-attack");
const attackSelect = document.getElementById("attack-scenario");
const fileInput = document.getElementById("log-file-input");
const btnStreamHdfs = document.getElementById("btn-stream-hdfs");
const streamStatus = document.getElementById("stream-status");

const badgeTau = document.getElementById("badge-tau");
const badgeDevice = document.getElementById("badge-device");
const metricTotal = document.getElementById("metric-total");
const metricAnomalies = document.getElementById("metric-anomalies");
const metricRate = document.getElementById("metric-rate");
const metricCritical = document.getElementById("metric-critical");
const metricLatency = document.getElementById("metric-latency");
const timerLabel = document.getElementById("analysis-timer");

const resultsPlaceholder = document.getElementById("results-table-placeholder");
const resultsContent = document.getElementById("results-content");
const eventsTbody = document.getElementById("events-tbody");
const heatmapContainer = document.getElementById("heatmap-container");
const xaiLineLabel = document.getElementById("xai-inspected-line");
const sessionBadgeContainer = document.getElementById("session-badge-container");
const triageBanner = document.getElementById("triage-banner");

const sessionsDrawer = document.getElementById("sessions-drawer");
const batchSessionCount = document.getElementById("batch-session-count");
const batchAnomalySummary = document.getElementById("batch-anomaly-summary");
const sessionPills = document.getElementById("session-pills");

const kdeChartContainer = document.getElementById("kde-chart-container");
const nllChartContainer = document.getElementById("nll-chart-container");
const kdeReadout = document.getElementById("kde-readout");
const benchmarkTbody = document.getElementById("benchmark-tbody");

let cachedKdeCurve = null;
let activeStreamTimer = null;

// Authentic Normal HDFS Block Write + Serve + Scan Sequence
const CLEAN_HDFS_SAMPLE = [
  "081109 203518 143 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000590_0/part-00590 blk_-1608999687",
  "081109 203519 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.251.19.102:54106 dest: /10.251.19.102:50010",
  "081109 203520 148 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687 of size 67108864 from /10.251.19.102",
  "081109 203521 150 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687 terminating",
  "081109 203522 152 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.19.102:50010 is added to blk_-1608999687 size 67108864",
  "081109 203523 155 INFO dfs.DataNode$DataXceiver: 10.251.19.102:50010 Served block blk_-1608999687 to /10.251.42.84",
  "081109 203524 158 INFO dfs.DataBlockScanner: Verification succeeded for blk_-1608999687"
].join("\n");

// Tab Switching
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// Load Health, KDE Curve, Benchmark, and Stats
async function initDashboardMeta() {
  try {
    const [healthRes, kdeRes, benchRes] = await Promise.all([
      fetch(`${API_BASE}/api/health`),
      fetch(`${API_BASE}/api/kde-curve`),
      fetch(`${API_BASE}/api/benchmark`)
    ]);
    if (healthRes.ok) {
      const h = await healthRes.json();
      badgeTau.textContent = `KDE τ: ${h.kde_threshold ?? "--"} (α=0.02)`;
      const osTag = h.opensearch_enabled ? " • OpenSearch ON" : "";
      badgeDevice.textContent = `PyTorch • ${String(h.device || "cpu").toUpperCase()}${osTag}`;
    }
    if (kdeRes.ok) {
      cachedKdeCurve = await kdeRes.json();
      renderKdeChart(cachedKdeCurve, null);
    }
    if (benchRes.ok) {
      const b = await benchRes.json();
      renderBenchmarkTable(b.rows || []);
    }
  } catch (e) {
    console.warn("Metadata load delayed", e);
  }
}

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
    console.warn("Stats endpoint unreachable", e);
  }
}

function renderBenchmarkTable(rows) {
  if (!rows.length) {
    benchmarkTbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text-muted);">No benchmark CSV found. Run python run_pipeline.py.</td></tr>`;
    return;
  }
  benchmarkTbody.innerHTML = rows.map(r => {
    const isOurs = String(r["Architecture"]).includes("Attention-BiLSTM") && String(r["Threshold Strategy"]).includes("Adaptive KDE");
    const fprPct = (Number(r["FPR"]) * 100).toFixed(2) + "%";
    const stratBadge = String(r["Threshold Strategy"]).includes("KDE")
      ? `<span class="tag-green">${escapeHtml(String(r["Threshold Strategy"]))}</span>`
      : `<span class="tag-gray">${escapeHtml(String(r["Threshold Strategy"]))}</span>`;
    return `
      <tr class="${isOurs ? "highlight-row" : ""}">
        <td><strong>${escapeHtml(String(r["Architecture"]))}</strong></td>
        <td>${stratBadge}</td>
        <td>${Number(r["Precision"]).toFixed(4)}</td>
        <td>${Number(r["Recall"]).toFixed(4)}</td>
        <td><strong>${Number(r["F1-Score"]).toFixed(4)}</strong></td>
        <td>${fprPct}</td>
        <td>${r["ROC-AUC"] !== undefined ? Number(r["ROC-AUC"]).toFixed(4) : "--"}</td>
      </tr>
    `;
  }).join("");
}

// Inline SVG Gaussian KDE Density Chart + Session Marker
function renderKdeChart(kdeData, currentSession) {
  if (!kdeData || !kdeData.x || !kdeData.x.length) return;
  const W = 540, H = 148, padL = 36, padR = 14, padT = 14, padB = 24;
  const xs = kdeData.x;
  const pdfs = kdeData.pdf;
  const tau = kdeData.threshold;
  const sessionNll = currentSession ? currentSession.session_max_nll : null;

  const xMin = xs[0];
  const xMax = Math.max(xs[xs.length - 1], sessionNll ? sessionNll * 1.12 : 0);
  const yMax = Math.max(...pdfs) * 1.15 || 1.0;

  const mapX = v => padL + ((v - xMin) / Math.max(1e-6, xMax - xMin)) * (W - padL - padR);
  const mapY = p => (H - padB) - (p / yMax) * (H - padT - padB);

  let normalPath = `M ${mapX(xs[0]).toFixed(1)} ${(H - padB).toFixed(1)} `;
  let tailPath = "";
  let linePts = "";
  let tailStarted = false;

  xs.forEach((xv, i) => {
    const px = mapX(xv).toFixed(1);
    const py = mapY(pdfs[i]).toFixed(1);
    linePts += `${i === 0 ? "M" : "L"} ${px} ${py} `;
    if (xv <= tau) {
      normalPath += `L ${px} ${py} `;
    } else {
      if (!tailStarted) {
        tailStarted = true;
        tailPath = `M ${px} ${(H - padB).toFixed(1)} L ${px} ${py} `;
      } else {
        tailPath += `L ${px} ${py} `;
      }
    }
  });
  const tauX = mapX(tau).toFixed(1);
  normalPath += `L ${tauX} ${(H - padB).toFixed(1)} Z`;
  if (tailStarted) {
    tailPath += `L ${mapX(xs[xs.length - 1]).toFixed(1)} ${(H - padB).toFixed(1)} Z`;
  }

  let markerHtml = "";
  if (sessionNll !== null && sessionNll !== undefined) {
    const sx = mapX(Math.min(xMax, Math.max(xMin, sessionNll))).toFixed(1);
    const isAnom = currentSession.anomaly_detected;
    const color = isAnom ? "#ef4444" : "#10b981";
    kdeReadout.innerHTML = `Session Max NLL: <strong style="color:${color}">${sessionNll.toFixed(2)}</strong> vs τ=${tau.toFixed(2)}`;
    markerHtml = `
      <line x1="${sx}" y1="${padT}" x2="${sx}" y2="${H - padB}" stroke="${color}" stroke-width="2" />
      <circle cx="${sx}" cy="${padT + 10}" r="5" fill="${color}" stroke="#fff" stroke-width="1.5" />
      <text x="${Math.min(W - 90, Math.max(padL + 4, Number(sx) - 24))}" y="${padT + 25}" fill="${color}" font-size="10" font-family="Fira Code">${isAnom ? "THREAT" : "NORMAL"} (${sessionNll.toFixed(2)})</text>
    `;
  } else {
    kdeReadout.textContent = `Threshold τ: ${tau.toFixed(2)}`;
  }

  kdeChartContainer.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="145">
      <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#334155" stroke-width="1"/>
      <path d="${normalPath}" fill="rgba(59, 130, 246, 0.22)" />
      ${tailPath ? `<path d="${tailPath}" fill="rgba(239, 68, 68, 0.30)" />` : ""}
      <path d="${linePts}" fill="none" stroke="#38bdf8" stroke-width="2" />
      <line x1="${tauX}" y1="${padT}" x2="${tauX}" y2="${H - padB}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 3"/>
      <text x="${Number(tauX) + 4}" y="${H - padB - 6}" fill="#fbbf24" font-size="10" font-family="Fira Code">τ=${tau.toFixed(2)}</text>
      ${markerHtml}
      <text x="${padL}" y="${H - 6}" fill="#94a3b8" font-size="10" font-family="Fira Code">${xMin.toFixed(1)}</text>
      <text x="${W - 44}" y="${H - 6}" fill="#94a3b8" font-size="10" font-family="Fira Code">${xMax.toFixed(1)} NLL</text>
    </svg>
  `;
}

// Inline SVG Per-Event NLL Spike Chart
function renderNllTimelineChart(session) {
  if (!session || !session.events || !session.events.length) return;
  const W = 540, H = 148, padL = 34, padR = 14, padT = 16, padB = 24;
  const events = session.events;
  const tau = session.kde_threshold;
  const maxVal = Math.max(tau * 1.35, ...events.map(e => e.event_nll), 2.0);

  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const barSlot = plotW / events.length;
  const barW = Math.min(34, Math.max(6, barSlot * 0.68));
  const tauY = (H - padB) - (tau / maxVal) * plotH;

  const bars = events.map((ev, i) => {
    const bh = Math.max(2, (ev.event_nll / maxVal) * plotH);
    const bx = padL + i * barSlot + (barSlot - barW) / 2;
    const by = (H - padB) - bh;
    const fill = ev.is_anomaly ? "#ef4444" : "#3b82f6";
    return `
      <rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${barW.toFixed(1)}" height="${bh.toFixed(1)}" rx="3" fill="${fill}" opacity="0.88">
        <title>Line ${ev.line_no}: NLL ${ev.event_nll.toFixed(2)} (${ev.severity})</title>
      </rect>
      ${events.length <= 20 ? `<text x="${(bx + barW / 2).toFixed(1)}" y="${H - 7}" text-anchor="middle" fill="#94a3b8" font-size="9.5" font-family="Fira Code">#${ev.line_no}</text>` : ""}
    `;
  }).join("");

  nllChartContainer.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="145">
      <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#334155" stroke-width="1"/>
      ${bars}
      <line x1="${padL}" y1="${tauY.toFixed(1)}" x2="${W - padR}" y2="${tauY.toFixed(1)}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 3"/>
      <text x="${padL + 4}" y="${Math.max(11, tauY - 4).toFixed(1)}" fill="#fbbf24" font-size="10" font-family="Fira Code">Threshold τ=${tau.toFixed(2)}</text>
      <text x="4" y="${padT + 4}" fill="#94a3b8" font-size="9.5" font-family="Fira Code">${maxVal.toFixed(1)}</text>
    </svg>
  `;
}

// Presets & File Upload
btnClean.addEventListener("click", () => {
  stopStreaming();
  logInput.value = CLEAN_HDFS_SAMPLE;
});

btnAttack.addEventListener("click", async () => {
  stopStreaming();
  btnAttack.disabled = true;
  const orig = btnAttack.textContent;
  btnAttack.textContent = "Loading...";
  try {
    const scenario = attackSelect.value;
    const res = await fetch(`${API_BASE}/api/simulate-attack?scenario=${encodeURIComponent(scenario)}`);
    const data = await res.json();
    logInput.value = data.logs.join("\n");
    timerLabel.textContent = `Loaded scenario: ${data.attack_type}`;
    await runSingleAnalysis();
  } catch (err) {
    alert("Backend unavailable. Check server status.");
  } finally {
    btnAttack.disabled = false;
    btnAttack.textContent = orig;
  }
});

btnAnalyze.addEventListener("click", () => {
  stopStreaming();
  runSingleAnalysis();
});

btnAnalyzeBatch.addEventListener("click", () => {
  stopStreaming();
  runBatchAnalysis();
});

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  stopStreaming();
  const formData = new FormData();
  formData.append("file", file);
  timerLabel.textContent = `Uploading & partitioning ${file.name}...`;
  try {
    const tStart = performance.now();
    const res = await fetch(`${API_BASE}/api/analyze-file`, { method: "POST", body: formData });
    if (!res.ok) throw new Error("Upload analysis failed");
    const batchData = await res.json();
    const elapsed = Math.round(performance.now() - tStart);
    timerLabel.textContent = `Parsed ${batchData.total_lines} lines into ${batchData.total_sessions} Block sessions in ${batchData.latency_ms} ms (Roundtrip: ${elapsed} ms)`;
    renderMultiSessionResults(batchData);
    refreshStats();
  } catch (err) {
    alert("Failed to analyze uploaded log file.");
    console.error(err);
  } finally {
    fileInput.value = "";
  }
});

// Live Stream Replay of Real HDFS_sample.log
function stopStreaming() {
  if (activeStreamTimer) {
    clearInterval(activeStreamTimer);
    activeStreamTimer = null;
    btnStreamHdfs.textContent = "📡 Stream HDFS Sample";
    streamStatus.textContent = "";
  }
}

btnStreamHdfs.addEventListener("click", async () => {
  if (activeStreamTimer) {
    stopStreaming();
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/api/sample-hdfs?limit=36`);
    const data = await res.json();
    const allLines = data.logs || [];
    if (!allLines.length) return;

    let idx = 0;
    const chunkSize = 6;
    logInput.value = "";
    btnStreamHdfs.textContent = "⏹ Stop Stream";

    const pushChunk = async () => {
      if (idx >= allLines.length) {
        stopStreaming();
        streamStatus.textContent = "Stream complete.";
        return;
      }
      const chunk = allLines.slice(idx, idx + chunkSize);
      idx += chunkSize;
      logInput.value = (logInput.value ? logInput.value + "\n" : "") + chunk.join("\n");
      streamStatus.textContent = `Streaming Loghub HDFS (${Math.min(idx, allLines.length)}/${allLines.length} lines)...`;
      await runBatchAnalysis(true);
    };

    await pushChunk();
    activeStreamTimer = setInterval(pushChunk, 1800);
  } catch (e) {
    alert("Could not load HDFS sample stream.");
  }
});

async function runSingleAnalysis() {
  const lines = logInput.value.split("\n").map(l => l.trim()).filter(Boolean);
  if (!lines.length) {
    alert("Please enter or load log lines to analyze.");
    return;
  }
  btnAnalyze.disabled = true;
  timerLabel.textContent = "Running Drain3 + Attention-BiLSTM + KDE...";
  try {
    const tStart = performance.now();
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_logs: lines, session_id: "live-session" })
    });
    if (!res.ok) throw new Error("Inference failed.");
    const data = await res.json();
    const elapsed = Math.round(performance.now() - tStart);
    timerLabel.textContent = `Completed in ${data.latency_ms} ms (Roundtrip: ${elapsed} ms)`;
    sessionsDrawer.style.display = "none";
    renderSessionResults(data);
    refreshStats();
  } catch (err) {
    alert("Error executing inference. Verify backend is running.");
    console.error(err);
  } finally {
    btnAnalyze.disabled = false;
  }
}

async function runBatchAnalysis(isStream = false) {
  const lines = logInput.value.split("\n").map(l => l.trim()).filter(Boolean);
  if (!lines.length) return;
  try {
    const tStart = performance.now();
    const res = await fetch(`${API_BASE}/api/analyze-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_logs: lines, session_id: "batch-stream", auto_group: true })
    });
    if (!res.ok) throw new Error("Batch analysis failed.");
    const batchData = await res.json();
    const elapsed = Math.round(performance.now() - tStart);
    if (!isStream) {
      timerLabel.textContent = `Grouped ${batchData.total_lines} lines into ${batchData.total_sessions} sessions in ${batchData.latency_ms} ms (Roundtrip: ${elapsed} ms)`;
    }
    renderMultiSessionResults(batchData);
    refreshStats();
  } catch (err) {
    console.error(err);
  }
}

function renderMultiSessionResults(batchData) {
  const sessions = batchData.sessions || [];
  if (!sessions.length) return;

  if (sessions.length === 1) {
    sessionsDrawer.style.display = "none";
    renderSessionResults(sessions[0]);
    return;
  }

  sessionsDrawer.style.display = "block";
  batchSessionCount.textContent = sessions.length;
  batchAnomalySummary.textContent = `${batchData.anomalous_sessions} Anomalous / ${sessions.length} Sessions`;
  sessionPills.innerHTML = "";

  sessions.slice(0, 60).forEach((sess, idx) => {
    const pill = document.createElement("button");
    pill.className = `session-pill ${sess.anomaly_detected ? "anom" : ""} ${idx === 0 ? "active" : ""}`;
    pill.textContent = `${sess.session_id} (${sess.total_events} ev | NLL ${sess.session_max_nll.toFixed(2)})`;
    pill.addEventListener("click", () => {
      document.querySelectorAll(".session-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      renderSessionResults(sess);
    });
    sessionPills.appendChild(pill);
  });

  renderSessionResults(sessions[0]);
}

function renderSessionResults(data) {
  resultsPlaceholder.style.display = "none";
  resultsContent.style.display = "block";
  eventsTbody.innerHTML = "";

  sessionBadgeContainer.innerHTML = data.anomaly_detected
    ? `<span class="tag-sev sev-${data.highest_severity}">THREAT DETECTED • ${data.highest_severity} (Max NLL ${data.session_max_nll.toFixed(2)})</span>`
    : `<span class="tag-sev sev-NORMAL">CLEAN SESSION • NORMAL (Max NLL ${data.session_max_nll.toFixed(2)})</span>`;

  triageBanner.className = `triage-banner ${data.anomaly_detected ? "threat" : ""}`;
  triageBanner.textContent = data.triage_summary || "";

  if (cachedKdeCurve) {
    renderKdeChart(cachedKdeCurve, data);
  }
  renderNllTimelineChart(data);

  data.events.forEach(evt => {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    const tmplBadge = evt.is_oov
      ? `<span class="tag-oov">&lt;UNK&gt; OOV</span>`
      : `<span class="tag-gray">T_${evt.template_id}</span>`;
    tr.innerHTML = `
      <td>${evt.line_no}</td>
      <td style="color: #cbd5e1; word-break: break-all;">${escapeHtml(evt.raw_text)}</td>
      <td>${tmplBadge}<div style="font-size:10.5px;color:var(--text-muted);margin-top:3px;">${escapeHtml(evt.template_text.slice(0, 72))}</div></td>
      <td style="color:${evt.is_anomaly ? "#fca5a5" : "#93c5fd"};font-weight:600;">${evt.event_nll.toFixed(2)}</td>
      <td><span class="tag-sev sev-${evt.severity}">${evt.severity}</span></td>
      <td>${(evt.anomaly_score * 100).toFixed(0)}%</td>
      <td><button class="btn btn-secondary btn-xs">Inspect</button></td>
    `;

    tr.addEventListener("click", () => {
      document.querySelectorAll(".events-table tbody tr").forEach(r => r.style.background = "");
      tr.style.background = "rgba(59, 130, 246, 0.14)";
      renderHeatmap(evt);
    });
    eventsTbody.appendChild(tr);
  });

  const targetEvt = data.events.find(e => e.is_anomaly) || data.events[data.events.length - 1];
  if (targetEvt) renderHeatmap(targetEvt);
}

// Human-Readable Pad-Masked Attention Attribution Heatmap
function renderHeatmap(evt) {
  heatmapContainer.innerHTML = "";
  xaiLineLabel.textContent = `Inspecting Line #${evt.line_no} (NLL ${evt.event_nll.toFixed(2)})`;
  const ctxTokens = evt.context_tokens || [];

  if (!ctxTokens.length) {
    heatmapContainer.innerHTML = `<span style="color: var(--text-muted); font-size: 12px;">Line #${evt.line_no} is the session entry point (no preceding context tokens). Select Line #2+ to inspect attention attributions.</span>`;
    return;
  }

  ctxTokens.forEach(tok => {
    const cell = document.createElement("div");
    cell.className = "heat-cell";
    const w = tok.weight || 0;
    const intensity = Math.min(1.0, w * 2.6);
    const r = Math.round(30 + 209 * intensity);
    const g = Math.round(41 + 40 * (1 - intensity));
    const b = Math.round(130 + 116 * (1 - intensity));
    const alpha = Math.max(0.22, 0.25 + intensity * 0.65);

    cell.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    cell.title = `${tok.offset} (Line ${tok.line_no}): ${tok.raw_preview}`;

    cell.innerHTML = `
      <div class="heat-top">
        <span>${tok.offset} • Line ${tok.line_no}</span>
        <span class="heat-pct">${(w * 100).toFixed(1)}%</span>
      </div>
      <div class="heat-tmpl">${escapeHtml(tok.template_text)}</div>
    `;
    heatmapContainer.appendChild(cell);
  });
}

function escapeHtml(str) {
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Telemetry Reset Handler
const btnReset = document.getElementById("btn-reset-telemetry");
if (btnReset) {
  btnReset.addEventListener("click", async () => {
    try {
      await fetch(`${API_BASE}/api/stats/reset`, { method: "POST" });
      await refreshStats();
    } catch (e) {
      console.warn("Failed to reset stats", e);
    }
  });
}

// Initialize Dashboard
logInput.value = CLEAN_HDFS_SAMPLE;
(async () => {
  await initDashboardMeta();
  await refreshStats();
  await runSingleAnalysis();
})();
