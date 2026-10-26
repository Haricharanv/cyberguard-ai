"""
src/models/attention_bilstm.py
Novel Architecture: Bidirectional LSTM with Multi-Head Self-Attention.
Extracts contextual bi-directional representations and outputs attention weights for
Explainable AI (XAI) token-level heatmap attribution.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

# Disable MKL-DNN on CPU to avoid Windows "could not create a primitive" RuntimeError in _VF.lstm
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False


class AttentionBiLSTM(nn.Module):
    """
    Bidirectional LSTM + Multi-Head Self-Attention + Residual Norm + Softmax Head.
    Exposes attention weights for XAI visualization in cybersecurity dashboards.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super(AttentionBiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Bidirectional LSTM -> hidden state size is hidden_dim * 2
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        bi_dim = hidden_dim * 2

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=bi_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Residual Normalization
        self.layer_norm = nn.LayerNorm(bi_dim)
        self.dropout = nn.Dropout(dropout)

        # Output projection head
        self.fc = nn.Linear(bi_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len)
        Returns:
            logits: (batch_size, vocab_size)
            attn_weights: (batch_size, seq_len, seq_len)
        """
        # (B, S) -> (B, S, embed_dim)
        emb = self.embed_dropout(self.embedding(x))

        # (B, S, bi_dim)
        lstm_out, _ = self.bilstm(emb)

        # Self-Attention: query=key=value=lstm_out
        attn_out, attn_weights = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            need_weights=True,
            average_attn_weights=True  # (B, S, S)
        )

        # Residual connection + LayerNorm
        normed = self.layer_norm(self.dropout(attn_out) + lstm_out)

        # Context representation from last timestep
        last_step = normed[:, -1, :]

        # Next-event prediction logits
        logits = self.fc(last_step)

        return logits, attn_weights

    def get_token_attributions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 1D importance score per token in the input sequence.
        Returns:
            probs: (batch_size, vocab_size)
            importance: (batch_size, seq_len) - normalized attention contribution to the decision
        """
        self.eval()
        with torch.no_grad():
            logits, attn_weights = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            # The last token's attention distribution over previous tokens: attn_weights[:, -1, :]
            token_importance = attn_weights[:, -1, :]
        return probs, token_importance
