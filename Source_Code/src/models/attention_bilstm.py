"""
src/models/attention_bilstm.py
Novel Architecture: Positional Bidirectional LSTM with Pad-Masked Multi-Head Self-Attention.
Masks PAD tokens (id=0) across both BiLSTM hidden representations and attention weights
so zero-padding never pollutes sequence context or XAI triage heatmaps.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple

# Disable MKL-DNN on CPU to avoid Windows "could not create a primitive" RuntimeError in _VF.lstm
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding so Self-Attention encodes exact step recency."""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionBiLSTM(nn.Module):
    """
    Positional Embedding -> Bidirectional LSTM -> Pad-Masked Multi-Head Self-Attention
    -> Residual LayerNorm -> Context Fusion -> Next-Event Logits + Clean Attention Map.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        bi_dim = hidden_dim * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=bi_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(bi_dim)
        self.dropout = nn.Dropout(dropout)

        # Fuse last-step representation with valid-token mean-pooled attended representation
        self.fusion = nn.Sequential(
            nn.Linear(bi_dim * 2, bi_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(bi_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len) token IDs (0 = PAD)
        Returns:
            logits:       (batch_size, vocab_size)
            attn_weights: (batch_size, seq_len, seq_len) with PAD keys strictly masked to 0
        """
        pad_mask = x == 0  # (B, S), True where PAD
        valid_mask = ~pad_mask  # (B, S)

        # Guard against all-padded rows to avoid NaN inside softmax
        safe_pad_mask = pad_mask.clone()
        all_pad = safe_pad_mask.all(dim=1)
        if all_pad.any():
            safe_pad_mask[all_pad, -1] = False

        emb = self.pos_encoding(self.embedding(x))
        emb = emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        emb = self.embed_dropout(emb)

        lstm_out, _ = self.bilstm(emb)
        lstm_out = lstm_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        attn_out, attn_weights = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            key_padding_mask=safe_pad_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # Explicitly zero-out PAD positions in attention matrix and renormalize across valid keys
        attn_weights = attn_weights.masked_fill(pad_mask.unsqueeze(1), 0.0)
        attn_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn_weights = attn_weights / attn_sum

        normed = self.layer_norm(self.dropout(attn_out) + lstm_out)
        normed = normed.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        last_step = normed[:, -1, :]

        # Masked mean pooling over valid (non-PAD) tokens captures long-range global context
        valid_float = valid_mask.unsqueeze(-1).float()
        valid_count = valid_float.sum(dim=1).clamp(min=1.0)
        pooled = (normed * valid_float).sum(dim=1) / valid_count

        fused = self.fusion(torch.cat([last_step, pooled], dim=-1))
        logits = self.fc(fused)

        return logits, attn_weights

    def get_token_attributions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts normalized 1D attention attribution per input token (PAD tokens are strictly 0.0).
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            token_importance = attn_weights[:, -1, :]
        return probs, token_importance
