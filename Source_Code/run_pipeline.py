"""
run_pipeline.py
End-to-End Orchestrator for CyberGuard AI Platform.
Executes dataset preparation, model training (DeepLog & Attention-BiLSTM),
adaptive KDE calibration, and formal benchmark comparative evaluation.
"""

import os
import sys
import torch
import pandas as pd

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import generate_benchmark_log_corpus, download_hdfs_sample
from src.train import train_model
from src.evaluate import run_benchmark_suite
from src.threshold import AdaptiveKDEScorer


def main():
    print("=================================================================")
    print("[*] CYBERGUARD AI -- FULL PIPELINE ORCHESTRATOR")
    print("    AI-Based Cybersecurity Log Anomaly Detection Platform")
    print("=================================================================\n")

    # Step 1: Data Preparation
    print("[STEP 1/4] Preparing Benchmark Datasets...")
    try:
        download_hdfs_sample("data/raw")
    except Exception as e:
        print(f"[!] Loghub sample download skipped or offline: {e}")

    train_normal, val_normal, test_df = generate_benchmark_log_corpus(
        num_normal_sessions=800,
        num_anomaly_sessions=150,
        output_dir="data/processed"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Data ready. Computation device: {device.upper()}\n")

    # Step 2: Train DeepLog Baseline
    print("[STEP 2/4] Training DeepLog Baseline (Unidirectional LSTM)...")
    deeplog_model, deeplog_kde, _ = train_model(
        model_type="deeplog",
        train_sequences=train_normal,
        val_sequences=val_normal,
        epochs=8,
        batch_size=128,
        device=device,
        save_dir="models/checkpoints"
    )
    print("[OK] DeepLog baseline training completed.\n")

    # Step 3: Train Attention-BiLSTM (Novel Proposed Model)
    print("[STEP 3/4] Training Attention-BiLSTM with Multi-Head Self-Attention...")
    attn_model, attn_kde, _ = train_model(
        model_type="attention_bilstm",
        train_sequences=train_normal,
        val_sequences=val_normal,
        epochs=10,
        batch_size=128,
        device=device,
        save_dir="models/checkpoints"
    )
    print("[OK] Attention-BiLSTM model and Adaptive KDE threshold calibrated.\n")

    # Step 4: Run Comparative Benchmark
    print("[STEP 4/4] Running Formal Comparative Benchmark Suite...")
    models_to_evaluate = {
        "DeepLog (Baseline)": (deeplog_model, deeplog_kde),
        "Attention-BiLSTM (Ours)": (attn_model, attn_kde)
    }

    results_df = run_benchmark_suite(
        models_dict=models_to_evaluate,
        test_df=test_df,
        window_size=10,
        top_k=9,
        device=device
    )

    # Save benchmark results
    results_path = "data/processed/benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[OK] Benchmark results saved to: {results_path}")

    print("\n=================================================================")
    print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
    print("   To launch the interactive SOC dashboard, run:")
    print("   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
