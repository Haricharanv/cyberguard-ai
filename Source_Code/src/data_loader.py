"""
src/data_loader.py
Dataset Downloader and Benchmark Generator.
Downloads sample logs from Loghub or generates realistic HDFS/BGL benchmark sequences for testing and training.
"""

import os
import random
import requests
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


HDFS_SAMPLE_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"


def download_hdfs_sample(target_dir: str = "data/raw") -> str:
    """
    Downloads the official Loghub HDFS 2,000 log line benchmark sample.
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "HDFS_sample.log")
    
    if os.path.exists(target_path):
        print(f"[OK] Found existing sample log at: {target_path}")
        return target_path

    print(f"[*] Downloading HDFS sample from {HDFS_SAMPLE_URL}...")
    resp = requests.get(HDFS_SAMPLE_URL, timeout=30)
    resp.raise_for_status()
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"[OK] Downloaded HDFS sample to: {target_path}")
    return target_path


def generate_benchmark_log_corpus(
    num_normal_sessions: int = 1200,
    num_anomaly_sessions: int = 200,
    output_dir: str = "data/processed"
) -> Tuple[List[List[int]], List[List[int]], pd.DataFrame]:
    """
    Generates realistic HDFS session sequences following typical block lifecycle patterns:
      Normal pattern:
        Allocating block (1) -> Serving block to datanode (2) -> Receiving block replica (3)
        -> PacketResponder terminated (4) -> Verification succeeded (5) -> Block closed (6)
      Anomalous patterns (Cybersecurity deviations & system faults):
        - Sudden connection reset / handshake abort
        - Unauthorized block write without allocation
        - Packet verification CRC mismatch
        - Out-of-order replication sequence injection
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    # Standard normal event templates (18 distinct normal events)
    # Block Write Workflow:  1 -> 2 -> 3 -> 4 -> 5 -> 6
    # Block Read Workflow:   7 -> 8 -> 9 -> 10
    # Replication Workflow:  11 -> 12 -> 13 -> 14
    # Compaction Workflow:   15 -> 16 -> 17
    # Heartbeat:             18

    # Rare / Anomaly event templates (never seen during normal training)
    # 30: Connection reset by peer
    # 31: Verification failed: checksum mismatch
    # 32: Unauthorized write attempt rejected
    # 33: Heartbeat expired, node dead
    # 34: Corrupted block detected
    # 35: Permission escalation denied
    # 36: Unexpected block state transition

    train_normal = []
    val_normal = []
    test_sessions = []

    WORKFLOWS = {
        "write":     [1, 2, 3, 4, 5, 6],
        "read":      [7, 8, 9, 10],
        "replicate": [11, 12, 13, 14],
        "compact":   [15, 16, 17],
    }

    ANOMALY_EVENTS = [30, 31, 32, 33, 34, 35, 36]

    def make_normal_seq():
        n_workflows = random.randint(2, 4)
        seq = []
        for _ in range(n_workflows):
            wf = WORKFLOWS[random.choice(list(WORKFLOWS.keys()))][:]
            if random.random() < 0.3:
                wf.insert(random.randint(0, len(wf)), 18)
            if len(wf) > 3 and random.random() < 0.15:
                wf.pop(random.randint(1, len(wf) - 2))
            seq.extend(wf)
        return seq

    def make_anomalous_seq():
        normal = make_normal_seq()
        anomaly_type = random.choice(["injection", "replacement", "burst", "abrupt_cut"])
        idx = random.randint(2, max(2, len(normal) - 2))
        if anomaly_type == "injection":
            normal.insert(idx, random.choice(ANOMALY_EVENTS))
        elif anomaly_type == "replacement":
            normal[idx] = random.choice(ANOMALY_EVENTS)
        elif anomaly_type == "burst":
            for j in range(random.randint(2, 3)):
                normal.insert(idx + j, random.choice(ANOMALY_EVENTS))
        elif anomaly_type == "abrupt_cut":
            normal = normal[:idx] + [random.choice(ANOMALY_EVENTS), 33]
        return normal

    # Generate training normal sessions (80%) and validation normal (20%)
    for i in range(num_normal_sessions):
        seq = make_normal_seq()
        if i < int(num_normal_sessions * 0.75):
            train_normal.append(seq)
        else:
            val_normal.append(seq)

    # Generate test set (mixture of normal and anomalies)
    test_records = []
    # 400 normal test sessions
    for i in range(400):
        seq = make_normal_seq()
        test_records.append({"block_id": f"blk_norm_{i}", "sequence": seq, "is_anomaly": 0})

    # Anomaly test sessions
    for i in range(num_anomaly_sessions):
        seq = make_anomalous_seq()
        test_records.append({"block_id": f"blk_anom_{i}", "sequence": seq, "is_anomaly": 1})

    random.shuffle(test_records)
    test_df = pd.DataFrame(test_records)

    test_csv_path = os.path.join(output_dir, "test_sessions.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"[OK] Benchmark dataset ready: {len(train_normal)} train, {len(val_normal)} val, {len(test_df)} test (Anomaly ratio: {test_df['is_anomaly'].mean():.1%})")

    return train_normal, val_normal, test_df
