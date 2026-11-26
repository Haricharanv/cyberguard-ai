"""
src/train.py
Model Training Engine for Log Anomaly Detection.
Supports Baseline DeepLog (LSTM) and Novel Positional Pad-Masked Attention-BiLSTM.
Synchronizes best validation checkpoint with Adaptive Gaussian KDE calibration.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Tuple, List, Optional

from src.models.deeplog import DeepLog
from src.models.attention_bilstm import AttentionBiLSTM
from src.threshold import AdaptiveKDEScorer
from src.dataset import create_dataloader
from src.evaluate import collect_session_nlls


def train_model(
    model_type: str,
    train_sequences: List[List[int]],
    val_sequences: List[List[int]],
    vocab_size: Optional[int] = None,
    window_size: int = 10,
    batch_size: int = 128,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    epochs: int = 20,
    lr: float = 0.0015,
    patience: int = 6,
    alpha: float = 0.02,
    device: Optional[str] = None,
    save_dir: str = "models/checkpoints",
) -> Tuple[nn.Module, AdaptiveKDEScorer, Dict[str, Any]]:
    """
    Trains DeepLog or Attention-BiLSTM on normal log sequences, reloads the best
    checkpoint, and fits an Adaptive KDE Scorer on normal validation session max-NLLs.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on device: {device.upper()} | Model: {model_type}")

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(42)

    train_loader, vocab_size_tr = create_dataloader(
        train_sequences, window_size=window_size, batch_size=batch_size, shuffle=True, vocab_size=vocab_size
    )
    inferred_vocab = max(vocab_size_tr, vocab_size or 50)
    val_loader, _ = create_dataloader(
        val_sequences, window_size=window_size, batch_size=batch_size, shuffle=False, vocab_size=inferred_vocab
    )
    final_vocab_size = inferred_vocab
    print(f"[*] Vocabulary size: {final_vocab_size:,} tokens | Training transitions: {len(train_loader.dataset):,}")

    if model_type.lower() == "deeplog":
        model = DeepLog(
            vocab_size=final_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif model_type.lower() in ["attention_bilstm", "attention", "bilstm_attn"]:
        model = AttentionBiLSTM(
            vocab_size=final_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'deeplog' or 'attention_bilstm'.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-4)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_checkpoint_path = os.path.join(save_dir, f"{model_type}_best.pt")
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
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
        scheduler.step()

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

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": final_vocab_size,
                    "window_size": window_size,
                    "embed_dim": embed_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "model_type": model_type,
                },
                best_checkpoint_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break

    if os.path.exists(best_checkpoint_path):
        print(f"[*] Reloading best model weights from {best_checkpoint_path} for KDE fitting...")
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state"])
        model.eval()

    print("[*] Computing session-level max NLLs on normal validation sequences...")
    session_nlls = collect_session_nlls(model, val_sequences, window_size=window_size, device=device)
    print(
        f"    Collected {len(session_nlls)} session max-NLL values | "
        f"min={session_nlls.min():.4f}, mean={session_nlls.mean():.4f}, "
        f"max={session_nlls.max():.4f}, std={session_nlls.std():.4f}"
    )

    print("[*] Fitting Adaptive Gaussian KDE on session max-NLL distribution...")
    kde_scorer = AdaptiveKDEScorer(alpha=alpha)
    kde_scorer.fit(session_nlls)
    kde_save_path = os.path.join(save_dir, f"{model_type}_kde.pkl")
    kde_scorer.save(kde_save_path)
    print(f"[OK] KDE fitted (alpha={alpha}). Dynamic NLL anomaly threshold tau: {kde_scorer.threshold:.4f}")

    return model, kde_scorer, history
