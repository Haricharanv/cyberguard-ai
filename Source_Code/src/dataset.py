"""
src/dataset.py
PyTorch Dataset and DataLoader utilities for log sequences.
Transforms template ID sequences into causal sliding-window training pairs
(x_seq, next_event_target) aligned 1:1 with sequential evaluation & live inference.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


class LogSequenceDataset(Dataset):
    """
    Causal sliding-window dataset covering ALL transitions t = 1 .. len(seq)-1.
    For each position i >= 1, left-pads preceding context seq[:i] with PAD=0
    up to window_size, predicting target seq[i].
    This ensures early session prefixes (idx < window_size) are trained identically
    to how evaluate.py and api/main.py score sequences at inference time.
    """

    def __init__(self, sequences: List[List[int]], window_size: int = 10, vocab_size: Optional[int] = None):
        self.window_size = window_size
        self.inputs = []
        self.targets = []

        for seq in sequences:
            if len(seq) < 2:
                continue
            for idx in range(1, len(seq)):
                context = seq[max(0, idx - window_size) : idx]
                if len(context) < window_size:
                    padded = [0] * (window_size - len(context)) + context
                else:
                    padded = context
                self.inputs.append(padded)
                self.targets.append(seq[idx])

        if len(self.inputs) == 0:
            self.inputs = torch.empty((0, window_size), dtype=torch.long)
            self.targets = torch.empty((0,), dtype=torch.long)
        else:
            self.inputs = torch.tensor(self.inputs, dtype=torch.long)
            self.targets = torch.tensor(self.targets, dtype=torch.long)

        if vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            max_in = int(self.inputs.max().item()) if len(self.inputs) > 0 else 0
            max_tgt = int(self.targets.max().item()) if len(self.targets) > 0 else 0
            self.vocab_size = max(max_in, max_tgt, 1) + 1

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_dataloader(
    sequences: List[List[int]],
    window_size: int = 10,
    batch_size: int = 128,
    shuffle: bool = True,
    vocab_size: Optional[int] = None,
) -> Tuple[DataLoader, int]:
    """Constructs a PyTorch DataLoader and returns (loader, vocab_size)."""
    dataset = LogSequenceDataset(sequences, window_size=window_size, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader, dataset.vocab_size
