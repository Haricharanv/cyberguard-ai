"""
src/session.py
Session Builder and Windowing Engine.
Groups parsed log events by session identifiers (e.g., HDFS block_id or time-based sliding windows)
and attaches ground-truth anomaly labels.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from src.parser import LogParser, extract_hdfs_block_id


def build_hdfs_sessions(log_file_path: str, max_lines: int = None) -> Dict[str, List[int]]:
    """
    Parses an HDFS log file and groups template IDs by Block ID.
    Returns: {block_id: [template_id_1, template_id_2, ...]}
    """
    parser = LogParser()
    sessions = defaultdict(list)

    print(f"[*] Parsing HDFS log: {log_file_path}")
    count = 0
    for event in parser.parse_file(log_file_path):
        block_id = extract_hdfs_block_id(event["raw"])
        if block_id != "UNKNOWN":
            sessions[block_id].append(event["cluster_id"])
        
        count += 1
        if max_lines and count >= max_lines:
            break
        if count % 100000 == 0:
            print(f"    Processed {count:,} lines...")

    print(f"[OK] Extracted {len(sessions):,} unique session blocks from {count:,} lines.")
    return dict(sessions)


def load_hdfs_anomaly_labels(label_file_path: str) -> Dict[str, int]:
    """
    Loads anomaly labels CSV for HDFS: BlockId,Label (Normal=0, Anomaly=1)
    """
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(f"Label file not found: {label_file_path}")

    df = pd.read_csv(label_file_path)
    label_map = {}
    for _, row in df.iterrows():
        b_id = str(row["BlockId"])
        # Standard Loghub convention: "Anomaly" -> 1, "Normal" -> 0
        label_map[b_id] = 1 if "anomaly" in str(row["Label"]).lower() else 0
    return label_map


def prepare_labeled_dataset(
    sessions: Dict[str, List[int]], 
    labels: Dict[str, int]
) -> pd.DataFrame:
    """
    Merges session sequences with their ground-truth anomaly label.
    """
    records = []
    for block_id, seq in sessions.items():
        if block_id in labels:
            records.append({
                "block_id": block_id,
                "sequence": seq,
                "length": len(seq),
                "is_anomaly": labels[block_id]
            })

    df = pd.DataFrame(records)
    print(f"[OK] Created dataset with {len(df):,} instances. Anomaly ratio: {df['is_anomaly'].mean():.2%}")
    return df
