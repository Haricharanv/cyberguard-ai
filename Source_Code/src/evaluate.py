"""
src/evaluate.py
Evaluation and Benchmark Comparison Engine.
Compares DeepLog Baseline (Top-K & Adaptive KDE) vs Attention-BiLSTM (Top-K & Adaptive KDE).
Uses batched per-session tensor evaluation for sub-10ms CPU inference latency.
"""

import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer


def _build_session_windows(sequence: List[int], window_size: int) -> Tuple[List[List[int]], List[int]]:
    """Builds all causal sliding windows and target IDs for a session in one batch."""
    windows = []
    targets = []
    for idx in range(1, len(sequence)):
        ctx = sequence[max(0, idx - window_size) : idx]
        if len(ctx) < window_size:
            ctx = [0] * (window_size - len(ctx)) + ctx
        windows.append(ctx)
        targets.append(sequence[idx])
    return windows, targets


def evaluate_session_topk(
    model: torch.nn.Module,
    sequence: List[int],
    window_size: int = 10,
    top_k: int = 5,
    device: str = "cpu",
) -> Tuple[bool, float]:
    """
    Session-level evaluation using DeepLog's Top-k rule:
    Flags a session if ANY actual next event is NOT in the top-k predicted candidates.
    """
    if len(sequence) < 2:
        return False, 0.0

    windows, targets = _build_session_windows(sequence, window_size)
    model.eval()
    with torch.no_grad():
        inp_t = torch.tensor(windows, dtype=torch.long, device=device)
        if isinstance(model, AttentionBiLSTM):
            logits, _ = model(inp_t)
        else:
            logits = model(inp_t)

        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.size(-1))
        topk_indices = torch.topk(probs, k=k, dim=-1).indices  # (T, k)

        flagged = False
        max_nll = 0.0
        for row_idx, tgt in enumerate(targets):
            tgt_clamped = min(tgt, probs.size(-1) - 1)
            p_obs = max(probs[row_idx, tgt_clamped].item(), 1e-12)
            max_nll = max(max_nll, -float(np.log(p_obs)))
            if tgt not in topk_indices[row_idx].tolist():
                flagged = True

    return flagged, max_nll


def compute_session_nll(
    model: torch.nn.Module,
    sequence: List[int],
    window_size: int = 10,
    device: str = "cpu",
) -> Tuple[float, float, List[float], List[List[float]]]:
    """
    Computes batched per-transition NLLs and attention attributions across a session.
    Returns (session_max_nll, session_mean_nll, per_event_nlls, all_attributions).
    """
    if len(sequence) < 2:
        return 0.0, 0.0, [], []

    windows, targets = _build_session_windows(sequence, window_size)
    model.eval()
    with torch.no_grad():
        inp_t = torch.tensor(windows, dtype=torch.long, device=device)
        if isinstance(model, AttentionBiLSTM):
            logits, attn_weights = model(inp_t)
            all_attributions = attn_weights[:, -1, :].cpu().numpy().tolist()
        else:
            logits = model(inp_t)
            all_attributions = []

        probs = torch.softmax(logits, dim=-1)
        tgt_clamped = torch.tensor(
            [min(t, probs.size(-1) - 1) for t in targets], dtype=torch.long, device=device
        )
        obs_probs = probs[torch.arange(len(targets), device=device), tgt_clamped].clamp(min=1e-12)
        per_event_nlls = (-torch.log(obs_probs)).cpu().numpy().tolist()

    session_max_nll = float(np.max(per_event_nlls)) if per_event_nlls else 0.0
    session_mean_nll = float(np.mean(per_event_nlls)) if per_event_nlls else 0.0
    return session_max_nll, session_mean_nll, per_event_nlls, all_attributions


def evaluate_session_kde(
    model: torch.nn.Module,
    kde_scorer: AdaptiveKDEScorer,
    sequence: List[int],
    window_size: int = 10,
    device: str = "cpu",
) -> Tuple[bool, float, float, List[List[float]]]:
    """Evaluates session max per-event NLL against Adaptive KDE threshold tau."""
    session_max_nll, _, _, all_attributions = compute_session_nll(
        model, sequence, window_size, device
    )
    is_anomaly, severity, _ = kde_scorer.score(session_max_nll)
    return is_anomaly, severity, session_max_nll, all_attributions


def collect_session_nlls(
    model: torch.nn.Module,
    sequences: List[List[int]],
    window_size: int = 10,
    device: str = "cpu",
) -> np.ndarray:
    """Computes session-level max NLLs for normal validation sequences."""
    nlls = []
    for seq in sequences:
        max_nll, _, _, _ = compute_session_nll(model, seq, window_size, device)
        nlls.append(max_nll)
    return np.array(nlls, dtype=np.float64)


def _compute_metrics_row(
    name: str,
    strategy: str,
    y_true: np.ndarray,
    preds: List[int],
    scores: List[float],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        roc_auc = roc_auc_score(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        roc_auc, pr_auc = 0.0, 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    return {
        "Architecture": name,
        "Threshold Strategy": strategy,
        "Precision": round(float(p), 4),
        "Recall": round(float(r), 4),
        "F1-Score": round(float(f1), 4),
        "FPR": round(float(fpr), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "PR-AUC": round(float(pr_auc), 4),
        "Latency (ms)": round(avg_lat, 2),
    }


def run_benchmark_suite(
    models_dict: Dict[str, Tuple[torch.nn.Module, AdaptiveKDEScorer]],
    test_df: pd.DataFrame,
    window_size: int = 10,
    top_k: int = 5,
    device: str = "cpu",
) -> pd.DataFrame:
    """Runs comparative benchmark comparing Static Top-K vs Adaptive KDE."""
    y_true = test_df["is_anomaly"].values
    results = []

    print(f"\n================ BENCHMARK EVALUATION (N={len(test_df):,} sessions) ================")

    for name, (model, kde_scorer) in models_dict.items():
        model = model.to(device)
        model.eval()

        # ------ Method A: Top-k Evaluation ------
        preds_topk, scores_topk, lats_topk = [], [], []
        for seq in test_df["sequence"]:
            t0 = time.perf_counter()
            pred, max_nll = evaluate_session_topk(
                model, seq, window_size=window_size, top_k=top_k, device=device
            )
            lats_topk.append((time.perf_counter() - t0) * 1000.0)
            preds_topk.append(1 if pred else 0)
            scores_topk.append(max_nll)

        results.append(
            _compute_metrics_row(name, f"Static Top-{top_k}", y_true, preds_topk, scores_topk, lats_topk)
        )

        # ------ Method B: Adaptive KDE (Session Max-NLL) ------
        if kde_scorer is not None:
            preds_kde, scores_kde, lats_kde = [], [], []
            for seq in test_df["sequence"]:
                t0 = time.perf_counter()
                flag, sev, max_nll, _ = evaluate_session_kde(
                    model, kde_scorer, seq, window_size=window_size, device=device
                )
                lats_kde.append((time.perf_counter() - t0) * 1000.0)
                preds_kde.append(1 if flag else 0)
                scores_kde.append(max_nll)

            results.append(
                _compute_metrics_row(name, "Adaptive KDE (Session NLL)", y_true, preds_kde, scores_kde, lats_kde)
            )

            if "anomaly_category" in test_df.columns:
                tmp_df = test_df.copy()
                tmp_df["pred"] = preds_kde
                print(f"\n  [{name} + Adaptive KDE] Recall by Anomaly Class:")
                for cat, grp in tmp_df[tmp_df["is_anomaly"] == 1].groupby("anomaly_category"):
                    cat_rec = grp["pred"].mean()
                    print(f"    - {cat:20s}: {cat_rec:.2%} ({int(grp['pred'].sum())}/{len(grp)})")

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    print("===================================================================================\n")
    return results_df
