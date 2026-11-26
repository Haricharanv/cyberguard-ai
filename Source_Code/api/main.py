"""
api/main.py
FastAPI Inference Service for Real-Time Cybersecurity Log Anomaly Detection.

Supports:
  - Frozen Drain3 online-trained template matching (zero hardcoded keyword rules)
  - Positional Pad-Masked Attention-BiLSTM next-event probability estimation
  - Adaptive Gaussian KDE thresholding on session-level max NLL
  - Multi-session Block-ID auto-grouping & raw .log file upload analysis
  - Token-level XAI triage attributions & live KDE density curve endpoints
  - Optional OpenSearch alert sink
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.parser import LogParser, UNK_TOKEN_ID, group_logs_by_session
from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer
from src.data_loader import generate_benchmark_log_corpus, download_hdfs_sample
from src.train import train_model
from src.integrations import OpenSearchAlertSink


# ---------------------------------------------------------
# Application State
# ---------------------------------------------------------
class AppState:
    model: Optional[AttentionBiLSTM] = None
    kde_scorer: Optional[AdaptiveKDEScorer] = None
    parser: Optional[LogParser] = None
    opensearch_sink: Optional[OpenSearchAlertSink] = None
    vocab_size: int = 50
    window_size: int = 10
    device: str = "cpu"
    recent_logs: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "total_analyzed": 0,
        "anomalies_detected": 0,
        "critical_alerts": 0,
        "avg_latency_ms": 0.0,
    }


state = AppState()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")


def init_cyberguard_engine() -> None:
    """Initializes or loads Drain3 parser, AttentionBiLSTM weights, and KDE scorer."""
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join(BASE_DIR, "models", "checkpoints")
    checkpoint_path = os.path.join(ckpt_dir, "attention_bilstm_best.pt")
    kde_path = os.path.join(ckpt_dir, "attention_bilstm_kde.pkl")
    parser_path = os.path.join(ckpt_dir, "drain3_parser.pkl")

    if (
        os.path.exists(checkpoint_path)
        and os.path.exists(kde_path)
        and os.path.exists(parser_path)
    ):
        print(f"[OK] Loading pre-trained engine artifacts from {ckpt_dir}...")
        state.parser = LogParser.load(parser_path)
        ckpt = torch.load(checkpoint_path, map_location=state.device, weights_only=True)
        state.vocab_size = ckpt["vocab_size"]
        state.window_size = ckpt.get("window_size", 10)
        state.model = AttentionBiLSTM(
            vocab_size=state.vocab_size,
            embed_dim=ckpt.get("embed_dim", 64),
            hidden_dim=ckpt.get("hidden_dim", 128),
            num_layers=ckpt.get("num_layers", 2),
            num_heads=ckpt.get("num_heads", 4),
        )
        state.model.load_state_dict(ckpt["model_state"])
        state.model.to(state.device)
        state.model.eval()
        state.kde_scorer = AdaptiveKDEScorer.load(kde_path)
    else:
        print("[*] Missing artifacts. Bootstrapping Drain3 + AttentionBiLSTM + KDE...")
        raw_dir = os.path.join(BASE_DIR, "data", "raw")
        proc_dir = os.path.join(BASE_DIR, "data", "processed")
        hdfs_sample = os.path.join(raw_dir, "HDFS_sample.log")
        try:
            hdfs_sample = download_hdfs_sample(raw_dir)
        except Exception:
            pass
        train_norm, val_norm, _, parser = generate_benchmark_log_corpus(
            num_normal_sessions=600,
            num_anomaly_sessions=120,
            output_dir=proc_dir,
            hdfs_sample_path=hdfs_sample,
            parser_save_path=parser_path,
        )
        state.parser = parser
        state.vocab_size = max(parser.vocab_size, 32)
        model, kde, _ = train_model(
            model_type="attention_bilstm",
            train_sequences=train_norm,
            val_sequences=val_norm,
            vocab_size=state.vocab_size,
            epochs=16,
            save_dir=ckpt_dir,
        )
        state.model = model
        state.kde_scorer = kde
        state.model.eval()

    state.opensearch_sink = OpenSearchAlertSink()
    print(f"[OK] CyberGuard Engine Ready! (Vocab={state.vocab_size}, KDE tau={state.kde_scorer.threshold:.4f})")


def _ensure_engine_loaded() -> None:
    if state.model is None or state.kde_scorer is None or state.parser is None:
        init_cyberguard_engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_engine_loaded()
    yield


app = FastAPI(
    title="CyberGuard AI — Log Anomaly Detection Platform",
    description="Real-time Drain3 + Positional Bi-LSTM + Self-Attention + Adaptive KDE Platform",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def serve_dashboard():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "CyberGuard AI API is operational. Frontend index not found."}


# ---------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------
class LogAnalyzeRequest(BaseModel):
    raw_logs: List[str]
    session_id: Optional[str] = "session-001"
    auto_group: Optional[bool] = False


class ContextTokenAttribution(BaseModel):
    offset: str
    line_no: int
    template_id: int
    template_text: str
    raw_preview: str
    weight: float


class LogEventResult(BaseModel):
    line_no: int
    raw_text: str
    template_id: int
    template_text: str
    is_oov: bool
    event_nll: float
    is_anomaly: bool = False
    severity: str = "NORMAL"
    anomaly_score: float = 0.0
    attention_attribution: List[float]
    context_tokens: List[ContextTokenAttribution] = []


class AnalysisResponse(BaseModel):
    session_id: str
    total_events: int
    anomaly_detected: bool
    highest_severity: str
    session_max_nll: float
    session_mean_nll: float
    kde_threshold: float
    severity_score: float
    triage_summary: str
    events: List[LogEventResult]
    latency_ms: float


class MultiSessionAnalysisResponse(BaseModel):
    total_lines: int
    total_sessions: int
    anomalous_sessions: int
    kde_threshold: float
    latency_ms: float
    sessions: List[AnalysisResponse]


# ---------------------------------------------------------
# Core Inference Helpers
# ---------------------------------------------------------
def _classify_severity(sev_score: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "NORMAL"
    if sev_score >= 0.70:
        return "CRITICAL"
    if sev_score >= 0.35:
        return "HIGH"
    return "MEDIUM"


def _analyze_session_lines(
    indexed_lines: List[tuple],
    session_id: str,
    update_stats: bool = True,
) -> AnalysisResponse:
    """
    Runs Drain3 frozen matching -> causal AttentionBiLSTM inference -> KDE session max-NLL scoring
    for a single logical session [(line_no, raw_line), ...].
    """
    _ensure_engine_loaded()
    t0 = time.perf_counter()

    parsed_items = []
    for orig_line_no, raw_line in indexed_lines:
        parsed = state.parser.match_line(raw_line)
        parsed_items.append(
            {
                "line_no": orig_line_no,
                "raw_text": raw_line,
                "token_id": parsed["token_id"],
                "template_text": parsed["template"],
                "is_oov": parsed["token_id"] == UNK_TOKEN_ID,
            }
        )

    template_seq = [item["token_id"] for item in parsed_items]
    per_event_nlls: List[float] = []
    per_event_attns: List[List[float]] = []
    per_event_context_tokens: List[List[ContextTokenAttribution]] = []

    state.model.eval()
    with torch.no_grad():
        if len(template_seq) >= 2:
            batch_inputs = []
            batch_meta = []
            for idx in range(1, len(template_seq)):
                start_idx = max(0, idx - state.window_size)
                context_indices = list(range(start_idx, idx))
                context_ids = [template_seq[j] for j in context_indices]
                pad_len = state.window_size - len(context_ids)
                padded_input = ([0] * pad_len) + context_ids
                batch_inputs.append([min(x, state.vocab_size - 1) for x in padded_input])
                batch_meta.append((idx, context_indices, pad_len))

            x_batch = torch.tensor(batch_inputs, dtype=torch.long, device=state.device)
            logits_batch, attn_batch = state.model(x_batch)
            probs_batch = torch.softmax(logits_batch, dim=-1)

            # Entry point (idx=0) has no prior transitions
            per_event_nlls.append(0.0)
            per_event_attns.append([0.0] * state.window_size)
            per_event_context_tokens.append([])

            for row_i, (idx, context_indices, pad_len) in enumerate(batch_meta):
                target_idx = min(template_seq[idx], state.vocab_size - 1)
                observed_prob = max(probs_batch[row_i, target_idx].item(), 1e-12)
                per_event_nlls.append(-float(np.log(observed_prob)))

                raw_attn = attn_batch[row_i, -1, :].cpu().numpy()
                for p_i in range(pad_len):
                    raw_attn[p_i] = 0.0
                attn_sum = float(np.sum(raw_attn))
                norm_attn = (raw_attn / attn_sum).tolist() if attn_sum > 1e-9 else [0.0] * state.window_size
                per_event_attns.append([round(float(a), 4) for a in norm_attn])

                ctx_list: List[ContextTokenAttribution] = []
                for pos, src_idx in enumerate(context_indices):
                    w = float(norm_attn[pad_len + pos])
                    src_item = parsed_items[src_idx]
                    steps_back = idx - src_idx
                    ctx_list.append(
                        ContextTokenAttribution(
                            offset=f"t-{steps_back}",
                            line_no=src_item["line_no"],
                            template_id=src_item["token_id"],
                            template_text=src_item["template_text"],
                            raw_preview=src_item["raw_text"][:100],
                            weight=round(w, 4),
                        )
                    )
                per_event_context_tokens.append(ctx_list)
        else:
            per_event_nlls.append(0.0)
            per_event_attns.append([0.0] * state.window_size)
            per_event_context_tokens.append([])

    # Session-level scoring over causal transitions (idx >= 1)
    transition_nlls = per_event_nlls[1:] if len(per_event_nlls) > 1 else per_event_nlls
    session_max_nll = float(np.max(transition_nlls)) if transition_nlls else 0.0
    session_mean_nll = float(np.mean(transition_nlls)) if transition_nlls else 0.0
    session_anomaly, severity_score, _ = state.kde_scorer.score(session_max_nll)
    session_sev = _classify_severity(severity_score, session_anomaly)

    events: List[LogEventResult] = []
    worst_idx = 0
    worst_nll = -1.0

    for idx, item in enumerate(parsed_items):
        ev_nll = per_event_nlls[idx]
        if idx >= 1 and ev_nll > worst_nll:
            worst_nll = ev_nll
            worst_idx = idx
        evt_anom, evt_sev_score, _ = state.kde_scorer.score(ev_nll)
        evt_sev = _classify_severity(evt_sev_score, evt_anom)

        events.append(
            LogEventResult(
                line_no=item["line_no"],
                raw_text=item["raw_text"],
                template_id=item["token_id"],
                template_text=item["template_text"],
                is_oov=item["is_oov"],
                event_nll=round(ev_nll, 4),
                is_anomaly=evt_anom,
                severity=evt_sev,
                anomaly_score=round(evt_sev_score, 4),
                attention_attribution=per_event_attns[idx],
                context_tokens=per_event_context_tokens[idx],
            )
        )

    # Human-readable triage diagnostic summary
    tau = float(state.kde_scorer.threshold)
    if len(parsed_items) < 2:
        triage_summary = "Single-event session observed; insufficient transitions for sequence scoring."
    elif not session_anomaly:
        triage_summary = (
            f"Session follows normal operational sequence (Max NLL {session_max_nll:.2f} <= KDE threshold {tau:.2f})."
        )
    else:
        worst_ev = events[worst_idx]
        reason = "Unseen zero-day log template (<UNK>)" if worst_ev.is_oov else "Unexpected sequence transition"
        top_ctx = sorted(worst_ev.context_tokens, key=lambda c: c.weight, reverse=True)[:2]
        ctx_desc = ", ".join(
            f"Line {c.line_no} ({c.weight * 100:.0f}%)" for c in top_ctx
        ) if top_ctx else "preceding context"
        triage_summary = (
            f"{session_sev} alert at Line {worst_ev.line_no}: {reason} "
            f"(NLL {worst_ev.event_nll:.2f} > tau {tau:.2f}). "
            f"Attention focused on {ctx_desc}."
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    response = AnalysisResponse(
        session_id=session_id,
        total_events=len(parsed_items),
        anomaly_detected=session_anomaly,
        highest_severity=session_sev,
        session_max_nll=round(session_max_nll, 4),
        session_mean_nll=round(session_mean_nll, 4),
        kde_threshold=round(tau, 4),
        severity_score=round(severity_score, 4),
        triage_summary=triage_summary,
        events=events,
        latency_ms=round(latency_ms, 2),
    )

    if update_stats:
        state.stats["total_analyzed"] += len(parsed_items)
        if session_anomaly:
            state.stats["anomalies_detected"] += 1
            state.recent_logs.append(
                {
                    "session_id": session_id,
                    "session_max_nll": round(session_max_nll, 4),
                    "session_mean_nll": round(session_mean_nll, 4),
                    "severity": session_sev,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )
            if len(state.recent_logs) > 50:
                state.recent_logs.pop(0)
            if state.opensearch_sink is not None:
                state.opensearch_sink.publish_session_alert(response.model_dump())
        if session_sev == "CRITICAL":
            state.stats["critical_alerts"] += 1
        prev_avg = state.stats["avg_latency_ms"]
        state.stats["avg_latency_ms"] = round(
            latency_ms if prev_avg <= 0.0 else (prev_avg * 0.85 + latency_ms * 0.15), 2
        )

    return response


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/api/health")
async def health_check():
    _ensure_engine_loaded()
    return {
        "status": "healthy",
        "engine": "Drain3 + Positional Pad-Masked Attention-BiLSTM + Adaptive KDE",
        "device": state.device,
        "model_loaded": state.model is not None,
        "kde_fitted": state.kde_scorer is not None,
        "kde_threshold": round(float(state.kde_scorer.threshold), 4) if state.kde_scorer else None,
        "vocab_size": state.vocab_size,
        "window_size": state.window_size,
        "opensearch_enabled": bool(state.opensearch_sink and state.opensearch_sink.enabled),
    }


@app.get("/api/stats")
async def get_dashboard_stats():
    anomaly_rate = (state.stats["anomalies_detected"] / max(1, state.stats["total_analyzed"])) * 100
    return {
        **state.stats,
        "anomaly_rate_percent": round(anomaly_rate, 2),
        "recent_alerts": state.recent_logs[-10:],
    }


@app.post("/api/stats/reset")
async def reset_dashboard_stats():
    state.stats = {
        "total_analyzed": 0,
        "anomalies_detected": 0,
        "critical_alerts": 0,
        "avg_latency_ms": 0.0,
    }
    state.recent_logs.clear()
    return {"status": "reset", "stats": state.stats}


@app.get("/api/kde-curve")
async def get_kde_distribution_curve():
    """Returns sampled Gaussian KDE probability density points and threshold tau for live visualization."""
    _ensure_engine_loaded()
    scorer = state.kde_scorer
    losses = scorer.fitted_losses
    lo = float(max(0.0, losses.min() - 0.5))
    hi = float(max(scorer.threshold * 2.2, losses.max() + 1.5))
    xs = np.linspace(lo, hi, 160)
    pdfs = scorer.kde.evaluate(xs)
    return {
        "x": [round(float(v), 4) for v in xs],
        "pdf": [round(float(p), 6) for p in pdfs],
        "threshold": round(float(scorer.threshold), 4),
        "alpha": scorer.alpha,
        "normal_stats": {
            "count": int(losses.size),
            "min": round(float(losses.min()), 4),
            "mean": round(float(losses.mean()), 4),
            "max": round(float(losses.max()), 4),
            "std": round(float(losses.std()), 4),
        },
    }


@app.get("/api/benchmark")
async def get_benchmark_results():
    """Returns live empirical benchmark results from data/processed/benchmark_results.csv."""
    csv_path = os.path.join(BASE_DIR, "data", "processed", "benchmark_results.csv")
    if not os.path.exists(csv_path):
        return {"available": False, "rows": []}
    df = pd.read_csv(csv_path)
    return {"available": True, "rows": df.to_dict(orient="records")}


@app.get("/api/sample-hdfs")
async def get_sample_hdfs_logs(limit: int = Query(default=60, ge=5, le=500)):
    """Returns real Loghub HDFS log lines from data/raw/HDFS_sample.log (auto-downloads if absent)."""
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    sample_path = os.path.join(raw_dir, "HDFS_sample.log")
    if not os.path.exists(sample_path):
        try:
            sample_path = download_hdfs_sample(raw_dir)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"HDFS_sample.log not found: {exc}")
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return {"source": "Loghub HDFS_2k.log", "total_available": len(lines), "logs": lines[:limit]}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_logs(payload: LogAnalyzeRequest):
    raw_lines = [ln.strip() for ln in payload.raw_logs if ln.strip()]
    if not raw_lines:
        raise HTTPException(status_code=400, detail="Empty log lines received.")

    indexed = [(idx + 1, line) for idx, line in enumerate(raw_lines)]
    return _analyze_session_lines(indexed, payload.session_id or "session-001", update_stats=True)


@app.post("/api/analyze-batch", response_model=MultiSessionAnalysisResponse)
async def analyze_logs_batch(payload: LogAnalyzeRequest):
    raw_lines = [ln.strip() for ln in payload.raw_logs if ln.strip()]
    if not raw_lines:
        raise HTTPException(status_code=400, detail="Empty log lines received.")

    _ensure_engine_loaded()
    t0 = time.perf_counter()
    grouped = group_logs_by_session(raw_lines, default_session_id=payload.session_id or "session-001")

    session_results: List[AnalysisResponse] = []
    for sess_id, indexed_lines in grouped.items():
        if len(indexed_lines) >= 2:
            session_results.append(_analyze_session_lines(indexed_lines, sess_id, update_stats=True))
        else:
            # Single-line block fragments still parse cleanly and show template info
            session_results.append(_analyze_session_lines(indexed_lines, sess_id, update_stats=False))

    # Surface anomalous and highest-NLL sessions first for immediate SOC triage
    session_results.sort(key=lambda s: (not s.anomaly_detected, -s.session_max_nll))
    anomalous_count = sum(1 for s in session_results if s.anomaly_detected)
    total_latency = (time.perf_counter() - t0) * 1000.0

    return MultiSessionAnalysisResponse(
        total_lines=len(raw_lines),
        total_sessions=len(session_results),
        anomalous_sessions=anomalous_count,
        kde_threshold=round(float(state.kde_scorer.threshold), 4),
        latency_ms=round(total_latency, 2),
        sessions=session_results,
    )


@app.post("/api/analyze-file", response_model=MultiSessionAnalysisResponse)
async def analyze_log_file(file: UploadFile = File(...)):
    """Ingests an uploaded .log/.txt file, partitions by Block ID, and ranks anomalous sessions."""
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Uploaded file contains no log lines.")
    if len(lines) > 5000:
        lines = lines[:5000]
    return await analyze_logs_batch(LogAnalyzeRequest(raw_logs=lines, session_id=file.filename or "upload-batch"))


@app.get("/api/simulate-attack")
async def simulate_attack_sequence(scenario: str = Query(default="zero_day")):
    """
    Returns realistic multi-stage security/fault attack scenarios:
      - zero_day: DataNode CRC32 Invalidation + Unauthorized Peer Injection (<UNK> templates)
      - order_violation: Normal INFO operations executed in illegal protocol order (no ERROR keywords!)
      - long_range: Allocation followed by noisy interleaved scans then unrecovered deletion
    """
    scenarios = {
        "zero_day": {
            "attack_type": "DataNode CRC Invalidation & Unauthorized Peer Injection",
            "logs": [
                "081109 203518 143 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000102_0/part-00102 blk_-1608999687",
                "081109 203519 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.251.19.102:54106 dest: /10.251.19.102:50010",
                "081109 203520 148 WARN dfs.DataNode$DataXceiver: Authentication failed for token credential key id 49102",
                "081109 203521 150 ERROR dfs.DataNode$DataXceiver: Unexpected block header CRC32 checksum mismatch in blk_-1608999687",
                "081109 203522 152 FATAL dfs.DataNode$BlockReceiver: Exception in writeTo: Connection abruptly reset by unauthorized peer /192.168.1.105:44321",
                "081109 203523 155 INFO dfs.DataNode$DataDirectory: Rollback attempted on storage directory /var/data/hdfs/current",
            ],
        },
        "order_violation": {
            "attack_type": "Stealthy Protocol State Inversion (100% Normal INFO Tokens)",
            "logs": [
                "081109 204110 212 INFO dfs.FSDataset: Deleting block blk_-9928172635 file /mnt/hadoop/dfs/data/current/subdir12/blk_-9928172635",
                "081109 204111 214 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.43.115:50010 is added to blk_-9928172635 size 67108864",
                "081109 204112 215 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-9928172635 terminating",
                "081109 204113 218 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000221_0/part-00221 blk_-9928172635",
                "081109 204114 220 INFO dfs.DataNode$DataXceiver: Receiving block blk_-9928172635 src: /10.251.43.115:41200 dest: /10.251.43.115:50010",
            ],
        },
        "long_range": {
            "attack_type": "Long-Range Lifecycle Violation (Incomplete Write + Silent Purge)",
            "logs": [
                "081109 205010 301 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000501_0/part-00501 blk_5519283746",
                "081109 205011 305 INFO dfs.DataNode$DataXceiver: Receiving block blk_5519283746 src: /10.251.73.220:39201 dest: /10.251.73.220:50010",
                "081109 205012 310 INFO dfs.DataBlockScanner: Verification succeeded for blk_5519283746",
                "081109 205013 312 INFO dfs.LeaseManager: Renewing client lease for blk_5519283746 holder DFSClient_412",
                "081109 205014 314 INFO dfs.DataNode$BPServiceActor: Heartbeat report acknowledged for blk_5519283746",
                "081109 205015 318 INFO dfs.BlockPoolSliceScanner: Checksum verified cleanly for blk_5519283746",
                "081109 205016 321 INFO dfs.DataNodeMetrics: Recorded block I/O telemetry latency 4 ms for blk_5519283746",
                "081109 205017 325 INFO dfs.FSDataset: Deleting block blk_5519283746 file /mnt/hadoop/dfs/data/current/subdir4/blk_5519283746",
            ],
        },
    }
    chosen = scenarios.get(scenario, scenarios["zero_day"])
    return {"scenario": scenario, **chosen}
