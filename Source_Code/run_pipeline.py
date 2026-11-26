"""
run_pipeline.py
End-to-End Orchestrator for CyberGuard AI Platform.
Executes real HDFS Drain3 template extraction, corpus generation, model training
(DeepLog & Attention-BiLSTM), Adaptive KDE calibration, and benchmark comparison.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import generate_benchmark_log_corpus, download_hdfs_sample
from src.train import train_model
from src.evaluate import run_benchmark_suite


def main():
    print("=================================================================")
    print("[*] CYBERGUARD AI -- FULL PIPELINE ORCHESTRATOR")
    print("    Drain3 + Positional Pad-Masked Bi-LSTM + Attention + KDE")
    print("=================================================================\n")

    # Step 1: Data Preparation & Drain3 Mining
    print("[STEP 1/4] Mining HDFS Templates & Preparing Benchmark Datasets...")
    try:
        hdfs_path = download_hdfs_sample("data/raw")
    except Exception as e:
        print(f"[!] Loghub sample download skipped or offline: {e}")
        hdfs_path = "data/raw/HDFS_sample.log"

    train_normal, val_normal, test_df, parser = generate_benchmark_log_corpus(
        num_normal_sessions=1200,
        num_anomaly_sessions=240,
        output_dir="data/processed",
        hdfs_sample_path=hdfs_path,
        parser_save_path="models/checkpoints/drain3_parser.pkl",
    )

    vocab_size = max(parser.vocab_size, 36)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Data ready. Drain3 vocab size: {vocab_size} | Computation device: {device.upper()}\n")

    # Step 2: Train DeepLog Baseline
    print("[STEP 2/4] Training DeepLog Baseline (Unidirectional LSTM)...")
    deeplog_model, deeplog_kde, _ = train_model(
        model_type="deeplog",
        train_sequences=train_normal,
        val_sequences=val_normal,
        vocab_size=vocab_size,
        epochs=18,
        batch_size=128,
        alpha=0.02,
        device=device,
        save_dir="models/checkpoints",
    )
    print("[OK] DeepLog baseline training completed.\n")

    # Step 3: Train Positional Pad-Masked Attention-BiLSTM (Proposed Model)
    print("[STEP 3/4] Training Attention-BiLSTM with Multi-Head Self-Attention...")
    attn_model, attn_kde, _ = train_model(
        model_type="attention_bilstm",
        train_sequences=train_normal,
        val_sequences=val_normal,
        vocab_size=vocab_size,
        epochs=24,
        batch_size=128,
        alpha=0.02,
        device=device,
        save_dir="models/checkpoints",
    )
    print("[OK] Attention-BiLSTM model and Adaptive KDE threshold calibrated.\n")

    # Step 4: Run Comparative Benchmark
    print("[STEP 4/4] Running Formal Comparative Benchmark Suite...")
    models_to_evaluate = {
        "DeepLog (Baseline)": (deeplog_model, deeplog_kde),
        "Attention-BiLSTM (Ours)": (attn_model, attn_kde),
    }

    results_df = run_benchmark_suite(
        models_dict=models_to_evaluate,
        test_df=test_df,
        window_size=10,
        top_k=9,
        device=device,
    )

    results_path = "data/processed/benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[OK] Benchmark results saved to: {results_path}")

    print("\n=================================================================")
    print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
    print("   To launch the interactive SOC dashboard, run:")
    print("   run.bat  (or .venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000)")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
