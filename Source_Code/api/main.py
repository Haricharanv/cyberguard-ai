"""
api/main.py
FastAPI Inference Service for Real-Time Cybersecurity Log Anomaly Detection.

Supports live Drain3 parsing, Attention-BiLSTM inference, Adaptive KDE
thresholding on session-level max NLL, and token-level attention heatmaps
for SOC triage.
"""

import os
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.parser import LogParser, extract_hdfs_block_id
from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer
from src.data_loader import generate_benchmark_log_corpus
from src.train import train_model

# ---------------------------------------------------------
# App Initialization & State
# ---------------------------------------------------------
app = FastAPI(
    title="CyberGuard AI — Log Anomaly Detection Platform",
    description="Real-time Bi-LSTM + Self-Attention Log Analysis & Explainable AI Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def serve_dashboard():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "CyberGuard AI API is operational. Frontend index not found."}

class AppState:
    model: Optional[AttentionBiLSTM] = None
    kde_scorer: Optional[AdaptiveKDEScorer] = None
    parser: Optional[LogParser] = None
    vocab_size: int = 100
    window_size: int = 10
    device: str = "cpu"
    recent_logs: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "total_analyzed": 0,
        "anomalies_detected": 0,
        "critical_alerts": 0,
        "avg_latency_ms": 1.2
    }

state = AppState()


# ---------------------------------------------------------
# Models & Schemas
# ---------------------------------------------------------
class LogAnalyzeRequest(BaseModel):
    raw_logs: List[str]
    session_id: Optional[str] = "session-001"

class LogEventResult(BaseModel):
    line_no: int
    raw_text: str
    template_id: int
    template_text: str
    event_nll: float
    is_anomaly: bool = False
    severity: str = "NORMAL"
    anomaly_score: float = 0.0
    attention_attribution: List[float]

class AnalysisResponse(BaseModel):
    session_id: str
    total_events: int
    anomaly_detected: bool
    highest_severity: str
    session_max_nll: float
    session_mean_nll: float
    kde_threshold: float
    severity_score: float
    events: List[LogEventResult]
    latency_ms: float


# ---------------------------------------------------------
# Startup Hook: Auto-train lightweight weights if not found
# ---------------------------------------------------------
def init_cyberguard_engine():
    """Initializes or loads model checkpoints and Drain3 parser."""
    if state.parser is None:
        state.parser = LogParser()
    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(base_dir, "models", "checkpoints", "attention_bilstm_best.pt")
    kde_path = os.path.join(base_dir, "models", "checkpoints", "attention_bilstm_kde.pkl")

    if os.path.exists(checkpoint_path) and os.path.exists(kde_path):
        print(f"[OK] Loading pre-trained model checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=state.device)
        state.vocab_size = ckpt["vocab_size"]
        state.window_size = ckpt.get("window_size", 10)
        state.model = AttentionBiLSTM(
            vocab_size=state.vocab_size,
            embed_dim=ckpt.get("embed_dim", 64),
            hidden_dim=ckpt.get("hidden_dim", 128),
            num_layers=2,
            num_heads=4
        )
        state.model.load_state_dict(ckpt["model_state"])
        state.model.to(state.device)
        state.model.eval()
        state.kde_scorer = AdaptiveKDEScorer.load(kde_path)
    else:
        print("[*] No saved weights found. Auto-training baseline weights on benchmark corpus...")
        train_norm, val_norm, _ = generate_benchmark_log_corpus(num_normal_sessions=400, num_anomaly_sessions=50)
        model, kde, _ = train_model(
            model_type="attention_bilstm",
            train_sequences=train_norm,
            val_sequences=val_norm,
            epochs=10,
            save_dir=os.path.join(base_dir, "models", "checkpoints")
        )
        state.model = model
        state.kde_scorer = kde
        state.model.eval()

    print("[OK] CyberGuard Engine Ready!")


# Eager initialization for immediate readiness
init_cyberguard_engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_cyberguard_engine()
    yield


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "Attention-BiLSTM + Adaptive KDE (Session NLL)",
        "device": state.device,
        "model_loaded": state.model is not None,
        "kde_fitted": state.kde_scorer is not None,
        "kde_threshold": state.kde_scorer.threshold if state.kde_scorer else None,
        "vocab_size": state.vocab_size,
        "window_size": state.window_size
    }


@app.get("/api/stats")
async def get_dashboard_stats():
    anomaly_rate = (
        (state.stats["anomalies_detected"] / max(1, state.stats["total_analyzed"])) * 100
    )
    return {
        **state.stats,
        "anomaly_rate_percent": round(anomaly_rate, 2),
        "recent_alerts": state.recent_logs[-10:]
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_logs(payload: LogAnalyzeRequest):
    if not payload.raw_logs:
        raise HTTPException(status_code=400, detail="Empty log lines received.")

    t0 = time.time()
    events = []
    template_seq = []

    # 1. Parse raw lines using Drain3
    mined_templates = []
    for idx, line in enumerate(payload.raw_logs):
        parsed = state.parser.parse_line(line)
        t_id = parsed["cluster_id"]
        template_seq.append(t_id)
        mined_templates.append(parsed.get("template", f"Template_{t_id}"))

    # 2. Sequential Inference: compute per-event NLLs and attention weights
    # Strictly follow next-event prediction with ZERO target leakage:
    # To predict the likelihood of event template_seq[idx], the input window
    # is constructed strictly from the preceding events template_seq[:idx].
    # The event being scored is NEVER present in the input sequence.
    per_event_nlls = []
    per_event_attns = []

    state.model.eval()
    with torch.no_grad():
        for idx in range(len(template_seq)):
            context = template_seq[:idx]

            # Build sliding window from preceding context (padded with 0 if shorter than window_size)
            if len(context) < state.window_size:
                padded_input = [0] * (state.window_size - len(context)) + context
            else:
                padded_input = context[-state.window_size:]

            # Safe input tensor clamped to vocab size
            inp_clamped = [min(x, state.vocab_size - 1) for x in padded_input]
            x_tensor = torch.tensor([inp_clamped], dtype=torch.long, device=state.device)

            logits, attn_weights = state.model(x_tensor)
            probs = torch.softmax(logits, dim=-1)

            # Compute NLL for this event (target is strictly the current event being evaluated)
            target_idx = min(template_seq[idx], state.vocab_size - 1)
            observed_prob = max(probs[0, target_idx].item(), 1e-12)
            event_nll = -np.log(observed_prob)
            per_event_nlls.append(event_nll)

            # Attention weights across the preceding context window (triage attribution)
            raw_attn = attn_weights[0, -1, :].cpu().numpy()
            norm_attn = (raw_attn / (np.sum(raw_attn) + 1e-7)).tolist()
            per_event_attns.append(norm_attn)

    # 3. Session-level anomaly decision: max-event NLL vs KDE threshold
    #    Max NLL captures sparse anomalous transitions that would be diluted by averaging.
    session_max_nll = float(np.max(per_event_nlls)) if per_event_nlls else 0.0
    session_mean_nll = float(np.mean(per_event_nlls)) if per_event_nlls else 0.0
    session_anomaly, severity, _ = state.kde_scorer.score(session_max_nll)

    # 4. Build event-level results
    for idx, line in enumerate(payload.raw_logs):
        evt_anom, evt_sev_score, _ = state.kde_scorer.score(per_event_nlls[idx])
        if evt_sev_score >= 0.75:
            evt_sev = "CRITICAL"
        elif evt_sev_score >= 0.40:
            evt_sev = "HIGH"
        elif evt_sev_score >= 0.15:
            evt_sev = "MEDIUM"
        else:
            evt_sev = "NORMAL"

        evt = LogEventResult(
            line_no=idx + 1,
            raw_text=line,
            template_id=template_seq[idx],
            template_text=mined_templates[idx],
            event_nll=round(per_event_nlls[idx], 4),
            is_anomaly=evt_anom,
            severity=evt_sev,
            anomaly_score=round(evt_sev_score, 4),
            attention_attribution=[round(a, 4) for a in per_event_attns[idx]]
        )
        events.append(evt)

    latency_ms = (time.time() - t0) * 1000.0

    # Update global stats
    state.stats["total_analyzed"] += len(payload.raw_logs)
    if session_anomaly:
        state.stats["anomalies_detected"] += 1
    if severity >= 0.75:
        state.stats["critical_alerts"] += 1
    state.stats["avg_latency_ms"] = round(
        (state.stats["avg_latency_ms"] * 0.9) + (latency_ms * 0.1), 2
    )

    # Classify session severity
    if severity >= 0.75:
        session_sev = "CRITICAL"
    elif severity >= 0.40:
        session_sev = "HIGH"
    elif severity >= 0.15:
        session_sev = "MEDIUM"
    else:
        session_sev = "NORMAL"

    # Store recent anomalous sessions
    if session_anomaly:
        state.recent_logs.append({
            "session_id": payload.session_id,
            "session_max_nll": round(session_max_nll, 4),
            "session_mean_nll": round(session_mean_nll, 4),
            "severity": session_sev,
            "timestamp": time.strftime("%H:%M:%S")
        })
        if len(state.recent_logs) > 50:
            state.recent_logs.pop(0)

    return AnalysisResponse(
        session_id=payload.session_id or "session-001",
        total_events=len(payload.raw_logs),
        anomaly_detected=session_anomaly,
        highest_severity=session_sev,
        session_max_nll=round(session_max_nll, 4),
        session_mean_nll=round(session_mean_nll, 4),
        kde_threshold=round(state.kde_scorer.threshold, 4),
        severity_score=round(severity, 4),
        events=events,
        latency_ms=round(latency_ms, 2)
    )


@app.get("/api/simulate-attack")
async def simulate_attack_sequence():
    """
    Simulates a live security attack on an HDFS data node (Privilege escalation + out-of-order write).
    """
    sample_attack_logs = [
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687 terminating",
        "081109 203520 148 WARN dfs.DataNode$DataXceiver: Authentication failed for token credential key id 49102",
        "081109 203521 150 ERROR dfs.DataNode$DataXceiver: Unexpected block header CRC32 checksum mismatch in blk_-1608999687",
        "081109 203522 152 FATAL dfs.DataNode$BlockReceiver: Exception in writeTo: Connection abruptly reset by unauthorized peer /192.168.1.105:44321",
        "081109 203523 155 INFO dfs.DataNode$DataDirectory: Rollback attempted on storage directory /var/data/hdfs/current"
    ]
    return {
        "attack_type": "DataNode CRC Invalidation & Unauthorized Peer Injection",
        "logs": sample_attack_logs
    }
