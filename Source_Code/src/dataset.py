"""
src/dataset.py
PyTorch Dataset and DataLoader utilities for log sequences.
Transforms template ID sequences into sliding window training pairs (x_seq, next_event_target).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import numpy as np


class LogSequenceDataset(Dataset):
    """
    Sliding window dataset for sequence modeling of log events.
    Given sequence [e1, e2, e3, e4, e5] with window_size=3:
      Sample 1: Input [e1, e2, e3] -> Target: e4
      Sample 2: Input [e2, e3, e4] -> Target: e5
    """

    def __init__(self, sequences: List[List[int]], window_size: int = 10, vocab_size: int = None):
        self.window_size = window_size
        self.inputs = []
        self.targets = []

        for seq in sequences:
            # Pad short sequences with 0 (index 0 is reserved for PAD)
            if len(seq) < window_size + 1:
                padded = [0] * (window_size + 1 - len(seq)) + seq
                self.inputs.append(padded[:window_size])
                self.targets.append(padded[window_size])
            else:
                for i in range(len(seq) - window_size):
                    self.inputs.append(seq[i : i + window_size])
                    self.targets.append(seq[i + window_size])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        # Max index in dataset if vocab_size not specified
        if vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            max_in = self.inputs.max().item() if len(self.inputs) > 0 else 0
            max_tgt = self.targets.max().item() if len(self.targets) > 0 else 0
            self.vocab_size = max(max_in, max_tgt) + 1

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def create_dataloader(
    sequences: List[List[int]], 
    window_size: int = 10, 
    batch_size: int = 128, 
    shuffle: bool = True,
    vocab_size: int = None
) -> Tuple[DataLoader, int]:
    """
    Factory helper to construct PyTorch DataLoader and return vocab size.
    """
    dataset = LogSequenceDataset(sequences, window_size=window_size, vocab_size=vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader, dataset.vocab_size
