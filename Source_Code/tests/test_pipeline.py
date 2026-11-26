"""
tests/test_pipeline.py
Unit and Integration Tests for CyberGuard AI core ML pipeline.
Verifies Drain3 parsing & frozen read-only OOV matching, causal window alignment,
pad-masked AttentionBiLSTM zero-attention on PAD tokens, and Adaptive KDE CDF thresholding.
"""

import pytest
import torch
import numpy as np
from src.parser import LogParser, extract_hdfs_block_id, group_logs_by_session, UNK_TOKEN_ID
from src.models.deeplog import DeepLog
from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer
from src.dataset import create_dataloader
from src.evaluate import compute_session_nll


def test_drain3_parser_and_frozen_oov_match():
    parser = LogParser()
    train_line = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    res = parser.parse_line(train_line, update_tree=True)

    assert "cluster_id" in res
    assert "template" in res
    assert res["token_id"] >= 2
    assert extract_hdfs_block_id(train_line) == "blk_-1608999687"

    # Read-only inference match on a new timestamp, block ID, and IP should match same cluster
    same_shape_line = "081109 211044 999 INFO dfs.DataNode$DataXceiver: Receiving block blk_999111222 src: /10.251.1.5:33100 dest: /10.251.1.6:50010"
    matched = parser.match_line(same_shape_line)
    assert matched["matched"] is True
    assert matched["token_id"] == res["token_id"]

    # Unseen zero-day log format must NOT mutate tree and must map to UNK_TOKEN_ID
    vocab_before = parser.vocab_size
    oov_line = "081109 211045 111 FATAL dfs.DataNode$BlockReceiver: Connection abruptly reset by unauthorized peer /192.168.1.50:44321"
    unseen = parser.match_line(oov_line)
    assert unseen["matched"] is False
    assert unseen["token_id"] == UNK_TOKEN_ID
    assert parser.vocab_size == vocab_before


def test_group_logs_by_session():
    lines = [
        "081109 203518 143 INFO dfs.DataNode: Receiving block blk_111 src: /10.0.0.1:500",
        "081109 203519 144 INFO dfs.DataNode: Receiving block blk_222 src: /10.0.0.2:500",
        "081109 203520 145 INFO dfs.DataNode: Received block blk_111 size 67108864",
    ]
    grouped = group_logs_by_session(lines)
    assert len(grouped) == 2
    assert len(grouped["blk_111"]) == 2
    assert len(grouped["blk_222"]) == 1


def test_sliding_window_causal_alignment():
    # Sequence of length 7 produces 6 causal transitions (t = 1..6)
    sequences = [[2, 3, 4, 5, 6, 7, 8]]
    loader, vocab_size = create_dataloader(sequences, window_size=3, batch_size=2)

    assert len(loader.dataset) == 6
    x, y = next(iter(loader))
    assert x.shape[1] == 3
    # First transition predicts seq[1]=3 from [0, 0, 2]
    assert loader.dataset[0][0].tolist() == [0, 0, 2]
    assert int(loader.dataset[0][1]) == 3


def test_model_forward_pass_and_pad_masking():
    """Both models return expected shapes, and AttentionBiLSTM gives strictly 0.0 weight to PAD tokens."""
    vocab_size = 30

    dl = DeepLog(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
    x = torch.randint(2, vocab_size, (4, 10))
    logits = dl(x)
    assert logits.shape == (4, vocab_size)

    ab = AttentionBiLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_heads=2)
    ab.eval()
    logits, attn_weights = ab(x)
    assert logits.shape == (4, vocab_size)
    assert attn_weights.shape == (4, 10, 10)

    # Left-padded sequence: first 6 slots are PAD (0), last 4 are valid tokens
    x_padded = torch.tensor([[0, 0, 0, 0, 0, 0, 2, 3, 4, 5]], dtype=torch.long)
    _, attn_padded = ab(x_padded)
    last_row_attn = attn_padded[0, -1, :]
    assert torch.allclose(last_row_attn[:6], torch.zeros(6), atol=1e-6)
    assert torch.isclose(last_row_attn[6:].sum(), torch.tensor(1.0), atol=1e-5)


def test_adaptive_kde_threshold():
    np.random.seed(42)
    normal_nlls = np.clip(np.random.normal(loc=1.5, scale=0.3, size=200), 0.1, None)

    scorer = AdaptiveKDEScorer(alpha=0.02)
    scorer.fit(normal_nlls)

    assert scorer.threshold is not None
    assert np.isfinite(scorer.threshold)
    assert scorer.kde is not None

    is_anom_normal, severity_normal, density_normal = scorer.score(1.5)
    assert is_anom_normal is False
    assert severity_normal == 0.0
    assert density_normal > 0.0

    is_anom_outlier, severity_outlier, _ = scorer.score(5.0)
    assert is_anom_outlier is True
    assert severity_outlier > 0.0


def test_session_nll_computation():
    vocab_size = 30
    model = AttentionBiLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_heads=2)
    sequence = [2, 3, 4, 5, 6, 7, 8, 9]

    max_nll, mean_nll, per_event_nlls, attributions = compute_session_nll(
        model, sequence, window_size=5, device="cpu"
    )

    assert max_nll >= mean_nll > 0.0
    assert len(per_event_nlls) == len(sequence) - 1
    assert len(attributions) == len(sequence) - 1
    assert len(attributions[0]) == 5
