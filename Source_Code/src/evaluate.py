"""
src/evaluate.py
Evaluation and Benchmark Comparison Engine.

Compares DeepLog Baseline (Top-k) vs Attention-BiLSTM (Top-k and Adaptive KDE).
Calculates Precision, Recall, F1-Score, False Positive Rate (FPR).

Methodology:
  - Top-k evaluation: follows DeepLog's original rule — flag a session if
    ANY transition's true next event is not in the top-k predictions.
  - KDE evaluation: computes the session-level max negative log-likelihood
    (NLL) across all transitions, then compares this aggregate against the
    KDE-derived threshold tau. This avoids the multi-testing inflation that
    occurs when checking each event independently, while preserving sensitivity
    to sparse anomaly transitions.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.models.deeplog import DeepLog
from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer


def _build_sliding_window(sequence: List[int], idx: int, window_size: int) -> List[int]:
    """Constructs zero-padded sliding window ending at position idx."""
    if idx < window_size:
        return [0] * (window_size - (idx + 1)) + sequence[: idx + 1]
    return sequence[idx - window_size + 1 : idx + 1]


def evaluate_session_topk(
    model: torch.nn.Module,
    sequence: List[int],
    window_size: int = 10,
    top_k: int = 9,
    device: str = "cpu"
) -> bool:
    """
    Session-level evaluation using DeepLog's Top-k rule:
    If ANY actual next event in the sequence is NOT in the top-k predicted
    candidates, the entire session is flagged as an ANOMALY (True).
    """
    if len(sequence) < 2:
        return False

    model.eval()
    with torch.no_grad():
        for i in range(len(sequence) - 1):
            inp = _build_sliding_window(sequence, i, window_size)
            inp_t = torch.tensor([inp], dtype=torch.long, device=device)
            target = sequence[i + 1]

            if isinstance(model, AttentionBiLSTM):
                logits, _ = model(inp_t)
            else:
                logits = model(inp_t)

            probs = torch.softmax(logits, dim=-1)
            top_candidates = torch.topk(
                probs, k=min(top_k, logits.size(-1)), dim=-1
            ).indices[0].tolist()

            if target not in top_candidates:
                return True  # Anomaly detected

    return False


def compute_session_nll(
    model: torch.nn.Module,
    sequence: List[int],
    window_size: int = 10,
    device: str = "cpu"
) -> Tuple[float, List[float], List[List[float]]]:
    """
    Computes per-transition negative log-likelihoods (NLL) across all
    transitions in a session, returning both the session max-event NLL
    (the primary anomaly decision statistic) and mean NLL (for sequence health).
    Also collects attention attributions (when the model is AttentionBiLSTM).

    Returns:
        session_max_nll:   the max of -log(P(true_next)) across all steps.
        session_mean_nll:  the mean of -log(P(true_next)) across all steps.
        per_event_nlls:    list of NLL values, one per transition.
        all_attributions:  list of attention weight vectors (empty for DeepLog).
    """
    if len(sequence) < 2:
        return 0.0, 0.0, [], []

    model.eval()
    per_event_nlls = []
    all_attributions = []

    with torch.no_grad():
        for i in range(len(sequence) - 1):
            inp = _build_sliding_window(sequence, i, window_size)
            inp_t = torch.tensor([inp], dtype=torch.long, device=device)
            target = sequence[i + 1]

            if isinstance(model, AttentionBiLSTM):
                logits, attn_weights = model(inp_t)
                attn = attn_weights[0, -1, :].cpu().numpy().tolist()
                all_attributions.append(attn)
            else:
                logits = model(inp_t)

            probs = torch.softmax(logits, dim=-1)
            tgt_idx = min(target, probs.size(-1) - 1)
            observed_prob = max(probs[0, tgt_idx].item(), 1e-12)

            # Negative log-likelihood for this transition
            nll = -np.log(observed_prob)
            per_event_nlls.append(nll)

    session_max_nll = float(np.max(per_event_nlls)) if per_event_nlls else 0.0
    session_mean_nll = float(np.mean(per_event_nlls)) if per_event_nlls else 0.0
    return session_max_nll, session_mean_nll, per_event_nlls, all_attributions


def evaluate_session_kde(
    model: torch.nn.Module,
    kde_scorer: AdaptiveKDEScorer,
    sequence: List[int],
    window_size: int = 10,
    device: str = "cpu"
) -> Tuple[bool, float, List[List[float]]]:
    """
    Session-level evaluation using Adaptive KDE threshold.

    Uses the **max per-event NLL** as the session-level statistic.
    This captures sparse anomalous transitions that would be diluted
    by averaging across many normal transitions in long sessions.

    Steps:
      1. Compute NLL for every transition in the session.
      2. Take the session max NLL (worst-case transition).
      3. Score the max NLL against the KDE-derived threshold tau.

    Returns:
        is_anomaly:      True if the session max NLL exceeds tau.
        severity:        normalized severity score [0, 1].
        all_attributions: attention weight vectors per transition (if BiLSTM).
    """
    session_max_nll, _, _, all_attributions = compute_session_nll(
        model, sequence, window_size, device
    )

    is_anomaly, severity, _ = kde_scorer.score(session_max_nll)
    return is_anomaly, severity, all_attributions


def collect_session_nlls(
    model: torch.nn.Module,
    sequences: List[List[int]],
    window_size: int = 10,
    device: str = "cpu"
) -> np.ndarray:
    """
    Computes session-level max NLLs for a list of sequences.
    Used to fit the KDE scorer on normal validation sessions.

    Returns max per-event NLL per session (not mean), since anomalies
    manifest as individual event spikes within sessions.
    """
    nlls = []
    for seq in sequences:
        max_nll, _, _, _ = compute_session_nll(model, seq, window_size, device)
        nlls.append(max_nll)
    return np.array(nlls, dtype=np.float64)


def run_benchmark_suite(
    models_dict: Dict[str, Tuple[torch.nn.Module, AdaptiveKDEScorer]],
    test_df: pd.DataFrame,
    window_size: int = 10,
    top_k: int = 9,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Runs full comparative benchmark across test sessions.
    Evaluates:
      1. Each model with static Top-k thresholding.
      2. Each model with Adaptive KDE (session-level max NLL) thresholding.
    """
    y_true = test_df["is_anomaly"].values
    results = []

    print(f"\n================ BENCHMARK EVALUATION (N={len(test_df):,} sessions) ================")

    for name, (model, kde_scorer) in models_dict.items():
        model = model.to(device)
        model.eval()

        # ------ Method A: Top-k Evaluation ------
        preds_topk = []
        for seq in test_df["sequence"]:
            pred = evaluate_session_topk(
                model, seq, window_size=window_size, top_k=top_k, device=device
            )
            preds_topk.append(1 if pred else 0)

        p_k = precision_score(y_true, preds_topk, zero_division=0)
        r_k = recall_score(y_true, preds_topk, zero_division=0)
        f1_k = f1_score(y_true, preds_topk, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(
            y_true, preds_topk
        ).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
        fpr_k = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results.append({
            "Architecture": name,
            "Threshold Strategy": f"Static Top-{top_k}",
            "Precision": round(p_k, 4),
            "Recall": round(r_k, 4),
            "F1-Score": round(f1_k, 4),
            "FPR": round(fpr_k, 4)
        })

        # ------ Method B: Adaptive KDE (Session Max-NLL) ------
        if kde_scorer is not None:
            preds_kde = []
            scores_kde = []
            for seq in test_df["sequence"]:
                flag, sev, _ = evaluate_session_kde(
                    model, kde_scorer, seq, window_size=window_size, device=device
                )
                preds_kde.append(1 if flag else 0)
                scores_kde.append(sev)

            p_kde = precision_score(y_true, preds_kde, zero_division=0)
            r_kde = recall_score(y_true, preds_kde, zero_division=0)
            f1_kde = f1_score(y_true, preds_kde, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(
                y_true, preds_kde
            ).ravel() if len(np.unique(y_true)) > 1 else (0, 0, 0, 0)
            fpr_kde = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results.append({
                "Architecture": name,
                "Threshold Strategy": "Adaptive KDE (Session NLL)",
                "Precision": round(p_kde, 4),
                "Recall": round(r_kde, 4),
                "F1-Score": round(f1_kde, 4),
                "FPR": round(fpr_kde, 4)
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("===================================================================================\n")
    return results_df
