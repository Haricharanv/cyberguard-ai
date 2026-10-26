"""
src/train.py
Model Training Engine for Log Anomaly Detection.
Supports both Baseline DeepLog (LSTM) and Novel Attention-BiLSTM architectures.

After training, the best checkpoint is reloaded and the Adaptive KDE Scorer is
fitted on session-level max negative log-likelihood (NLL) values from normal
validation sequences. This ensures the saved model and the saved KDE threshold
are strictly synchronized and derived from the model's continuous CDF.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, Any, Tuple, List

from src.models.deeplog import DeepLog
from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer
from src.dataset import create_dataloader
from src.evaluate import collect_session_nlls


def train_model(
    model_type: str,
    train_sequences: List[List[int]],
    val_sequences: List[List[int]],
    window_size: int = 10,
    batch_size: int = 128,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    epochs: int = 25,
    lr: float = 0.001,
    patience: int = 5,
    device: str = None,
    save_dir: str = "models/checkpoints"
) -> Tuple[nn.Module, AdaptiveKDEScorer, Dict[str, Any]]:
    """
    Trains DeepLog or Attention-BiLSTM model on normal log sequences,
    restores the best checkpoint, then fits an Adaptive KDE Scorer on
    session-level max-event NLLs from the normal validation set.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on device: {device.upper()} | Model: {model_type}")

    os.makedirs(save_dir, exist_ok=True)

    # Build DataLoaders
    train_loader, vocab_size_tr = create_dataloader(
        train_sequences, window_size=window_size, batch_size=batch_size, shuffle=True
    )
    val_loader, vocab_size_val = create_dataloader(
        val_sequences, window_size=window_size, batch_size=batch_size, shuffle=False,
        vocab_size=max(vocab_size_tr, 50)
    )
    vocab_size = max(vocab_size_tr, vocab_size_val)
    print(f"[*] Vocabulary size: {vocab_size:,} templates | Training samples: {len(train_loader.dataset):,}")

    # Instantiate Model
    if model_type.lower() == "deeplog":
        model = DeepLog(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    elif model_type.lower() in ["attention_bilstm", "attention", "bilstm_attn"]:
        model = AttentionBiLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'deeplog' or 'attention_bilstm'.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if isinstance(model, AttentionBiLSTM):
                logits, _ = model(x_batch)
            else:
                logits = model(x_batch)

            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                if isinstance(model, AttentionBiLSTM):
                    val_logits, _ = model(x_val)
                else:
                    val_logits = model(x_val)

                v_loss = criterion(val_logits, y_val)
                total_val_loss += v_loss.item()

                preds = torch.argmax(val_logits, dim=-1)
                correct += (preds == y_val).sum().item()
                total_samples += y_val.size(0)

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_acc = correct / max(1, total_samples)

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_checkpoint_path = os.path.join(save_dir, f"{model_type}_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "window_size": window_size,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "model_type": model_type
            }, best_checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break

    # ---------------------------------------------------------------
    # CRITICAL: Restore best checkpoint before computing validation
    # NLLs and fitting the KDE. This guarantees that the model in memory,
    # the checkpoint on disk, and the KDE threshold are strictly
    # synchronized to the exact same optimal validation state.
    # ---------------------------------------------------------------
    if os.path.exists(best_checkpoint_path):
        print(f"[*] Reloading best model weights from {best_checkpoint_path} for KDE fitting...")
        best_ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        model.eval()

    # ---------------------------------------------------------------
    # Fit Adaptive KDE Scorer on SESSION-LEVEL max NLLs from the
    # normal validation set. The max per-event NLL is the decision
    # statistic, capturing sparse anomalous transitions without
    # diluting them across long normal sessions.
    # ---------------------------------------------------------------
    print("[*] Computing session-level max NLLs on normal validation sequences...")
    session_nlls = collect_session_nlls(
        model, val_sequences, window_size=window_size, device=device
    )
    print(f"    Collected {len(session_nlls)} session max-NLL values.")
    print(f"    Max-NLL distribution: min={session_nlls.min():.4f}, "
          f"mean={session_nlls.mean():.4f}, max={session_nlls.max():.4f}, "
          f"std={session_nlls.std():.4f}")

    print("[*] Fitting Adaptive Gaussian KDE on session max-NLL distribution...")
    kde_scorer = AdaptiveKDEScorer(alpha=0.02)
    kde_scorer.fit(session_nlls)
    kde_save_path = os.path.join(save_dir, f"{model_type}_kde.pkl")
    kde_scorer.save(kde_save_path)
    print(f"[OK] KDE fitted. Dynamic NLL anomaly threshold tau: {kde_scorer.threshold:.4f}")
    print(f"     (Sessions with max-event NLL > {kde_scorer.threshold:.4f} are flagged as anomalous)")

    return model, kde_scorer, history
