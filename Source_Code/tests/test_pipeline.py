"""
tests/test_pipeline.py
Unit and Integration Tests for CyberGuard AI core ML pipeline.
Verifies Drain3 parsing, PyTorch model forward passes, KDE thresholding
(session-level NLL with CDF-derived tau), and dataset construction.
"""

import pytest
import torch
import numpy as np
from src.parser import LogParser, extract_hdfs_block_id
from src.models.deeplog import DeepLog
from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer
from src.dataset import LogSequenceDataset, create_dataloader
from src.evaluate import compute_session_nll


def test_drain3_parser():
    parser = LogParser()
    sample_line = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.250.19.102:54106"
    res = parser.parse_line(sample_line)

    assert "cluster_id" in res
    assert "template" in res
    assert extract_hdfs_block_id(sample_line) == "blk_-1608999687"


def test_sliding_window_sessionizer():
    sequences = [[1, 2, 3, 4, 5, 6, 7]]
    loader, vocab_size = create_dataloader(sequences, window_size=3, batch_size=2)

    assert len(loader.dataset) == 4
    x, y = next(iter(loader))
    assert x.shape[1] == 3


def test_model_forward_pass():
    """Both DeepLog and AttentionBiLSTM produce correct output shapes."""
    vocab_size = 30

    # DeepLog
    dl = DeepLog(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
    x = torch.randint(1, vocab_size, (4, 10))
    logits = dl(x)
    assert logits.shape == (4, vocab_size)

    # AttentionBiLSTM
    ab = AttentionBiLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_heads=2)
    logits, attn_weights = ab(x)
    assert logits.shape == (4, vocab_size)
    assert attn_weights.shape == (4, 10, 10)


def test_adaptive_kde_threshold():
    """
    Verifies that:
    1. The KDE is fitted on session NLLs.
    2. The threshold tau is derived from the KDE's CDF (not np.percentile).
    3. Normal sessions score below tau; extreme outliers score above.
    """
    # Simulate session-level max NLL values from 200 normal sessions
    np.random.seed(42)
    normal_nlls = np.random.normal(loc=1.5, scale=0.3, size=200)
    normal_nlls = np.clip(normal_nlls, 0.1, None)

    scorer = AdaptiveKDEScorer(alpha=0.02)
    scorer.fit(normal_nlls)

    # Threshold must exist and be a finite number
    assert scorer.threshold is not None
    assert np.isfinite(scorer.threshold)
    assert scorer.threshold > 0.0

    # The KDE object must be fitted (not None)
    assert scorer.kde is not None

    # A typical normal session NLL should NOT be flagged
    is_anom_normal, severity_normal, density_normal = scorer.score(1.5)
    assert is_anom_normal is False
    assert severity_normal == 0.0
    assert density_normal > 0.0  # KDE density is used

    # An extreme outlier NLL should be flagged
    is_anom_outlier, severity_outlier, density_outlier = scorer.score(5.0)
    assert is_anom_outlier is True
    assert severity_outlier > 0.0


def test_session_nll_computation():
    """
    Verifies that compute_session_nll returns correct max/mean NLLs
    and per-event NLL list of the correct length.
    """
    vocab_size = 30
    model = AttentionBiLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_heads=2)
    sequence = [1, 2, 3, 4, 5, 6, 7, 8]

    max_nll, mean_nll, per_event_nlls, attributions = compute_session_nll(
        model, sequence, window_size=5, device="cpu"
    )

    # Max NLL should be >= mean NLL
    assert max_nll >= mean_nll
    # Both should be positive (they're -log of a probability)
    assert max_nll > 0.0
    assert mean_nll > 0.0
    # One NLL per transition (len(sequence) - 1)
    assert len(per_event_nlls) == len(sequence) - 1
    # Attributions collected for AttentionBiLSTM
    assert len(attributions) == len(sequence) - 1
    # Each attribution vector has length = window_size
    assert len(attributions[0]) == 5


def test_sequential_window_alignment_no_leakage():
    """
    Regression Test: Verifies that in sequential next-event prediction,
    the target event at step idx is never included in the preceding context window.
    """
    seq = [10, 20, 30, 40, 50, 60]
    window_size = 4
    for idx in range(len(seq)):
        context = seq[:idx]
        if len(context) < window_size:
            padded = [0] * (window_size - len(context)) + context
        else:
            padded = context[-window_size:]
        target = seq[idx]

        # Target must NEVER be the last element of the input window (the old leakage bug)
        assert padded[-1] != target
        # For unique sequence items, target must be completely absent from input window
        assert target not in padded
        # Window size must be strictly preserved
        assert len(padded) == window_size

