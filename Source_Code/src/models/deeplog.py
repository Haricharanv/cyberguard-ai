"""
src/models/deeplog.py
Baseline DeepLog Model Implementation (Du et al., ACM CCS 2017).
A unidirectional LSTM trained on normal log sequences for next-event prediction.
"""

import torch
import torch.nn as nn
from typing import Tuple, List

# Disable MKL-DNN on CPU to avoid Windows "could not create a primitive" RuntimeError in _VF.lstm
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False


class DeepLog(nn.Module):
    """
    Standard DeepLog architecture:
    Embedding Layer -> Unidirectional LSTM (2 layers) -> Fully Connected -> Softmax
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(DeepLog, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len)
        returns logits: (batch_size, vocab_size)
        """
        # (B, S) -> (B, S, embed_dim)
        embedded = self.embedding(x)
        
        # (B, S, hidden_dim)
        lstm_out, _ = self.lstm(embedded)
        
        # Take the output of the last timestep
        last_step = lstm_out[:, -1, :]
        
        # (B, vocab_size)
        logits = self.fc(last_step)
        return logits

    def predict_top_k(self, x: torch.Tensor, k: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns top-k predicted template IDs and their probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(k, self.vocab_size), dim=-1)
        return top_indices, top_probs
