"""
src/data_loader.py
Dataset Downloader, Real HDFS Loghub Template Miner & Multi-Workflow Corpus Generator.
Parses raw HDFS_sample.log via Drain3, expands with realistic multi-component distributed
workflows (concurrent background telemetry, rare benign admin ops, context-gated lifecycles),
and builds 4 authentic anomaly classes (order violations, long-range context violations,
abrupt faults, zero-day OOV templates).
"""

import os
import random
import requests
import pandas as pd
from typing import List, Tuple

from src.parser import LogParser


HDFS_SAMPLE_URL = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"


def download_hdfs_sample(target_dir: str = "data/raw") -> str:
    """Downloads the official Loghub HDFS 2,000 log line benchmark sample."""
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


def _fmt_hdfs(step: int, level: str, comp: str, msg: str) -> str:
    hh = 20 + (step // 3600)
    mm = 30 + ((step // 60) % 29)
    ss = step % 60
    pid = 100 + (step % 800)
    return f"081109 {hh:02d}{mm:02d}{ss:02d} {pid} {level} {comp}: {msg}"


def _build_realistic_hdfs_raw_session(rng: random.Random, blk_id: str, mode: str = "normal") -> List[str]:
    """
    Constructs authentic HDFS & distributed storage session logs.
    Models realistic multi-threaded cluster concurrency where shared background telemetry
    (heartbeats, lease renewals, block scans, cache pings, RPC telemetry) interleaves inside
    distinct initiating contracts (Client Write vs Authorized Admin Maintenance).
    """
    ip_src = f"10.251.{rng.randint(10, 220)}.{rng.randint(2, 250)}:{rng.randint(30000, 60000)}"
    ip_dst = f"10.251.{rng.randint(10, 220)}.{rng.randint(2, 250)}:50010"
    ip_peer = f"10.251.{rng.randint(10, 220)}.{rng.randint(2, 250)}"
    size = 67108864
    part = f"/user/root/rand/_temporary/_task_200811092030_0001_m_{rng.randint(100,999):06d}_0/part-{rng.randint(100,999):05d}"
    sub = f"/mnt/hadoop/dfs/data/current/subdir{rng.randint(1, 63)}/{blk_id}"

    # Canonical HDFS templates (matching HDFS_sample.log)
    E_ALLOC = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSNamesystem", f"BLOCK* NameSystem.allocateBlock: {part} {blk_id}")
    E_RCVING = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode$DataXceiver", f"Receiving block {blk_id} src: /{ip_src} dest: /{ip_dst}")
    E_RCVD_PR = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode$PacketResponder", f"Received block {blk_id} of size {size} from /{ip_peer}")
    E_RCVD_DX = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode$DataXceiver", f"Received block {blk_id} src: /{ip_src} dest: /{ip_dst} of size {size}")
    E_TERM = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode$PacketResponder", f"PacketResponder {rng.randint(0,2)} for block {blk_id} terminating")
    E_ADD = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSNamesystem", f"BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip_dst} is added to {blk_id} size {size}")
    E_SERVE = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode$DataXceiver", f"{ip_dst} Served block {blk_id} to /{ip_peer}")
    E_SCAN = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataBlockScanner", f"Verification succeeded for {blk_id}")
    E_REPL_ASK = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSNamesystem", f"BLOCK* ask {ip_dst} to replicate {blk_id} to datanode(s) {ip_peer}:50010")
    E_XFER = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode", f"{ip_dst} Starting thread to transfer block {blk_id} to {ip_peer}:50010")
    E_INVAL = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSNamesystem", f"BLOCK* NameSystem.delete: {blk_id} is added to invalidSet of {ip_dst}")
    E_DEL_ASK = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSNamesystem", f"BLOCK* ask {ip_dst} to delete {blk_id}")
    E_DEL_FS = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSDataset", f"Deleting block {blk_id} file {sub}")
    E_SERVE_EXC = lambda s: _fmt_hdfs(s, "WARN", "dfs.DataNode$DataXceiver", f"{ip_dst}:Got exception while serving {blk_id} to /{ip_peer}:")

    # Concurrent distributed telemetry & admin events
    E_LEASE = lambda s: _fmt_hdfs(s, "INFO", "dfs.LeaseManager", f"Renewing client lease for {blk_id} holder DFSClient_{rng.randint(100,999)}")
    E_ACK = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode$PacketResponder", f"PacketResponder ack seqno {rng.randint(1,64)} for block {blk_id}")
    E_CHECKSUM_OK = lambda s: _fmt_hdfs(s, "INFO", "dfs.BlockPoolSliceScanner", f"Checksum verified cleanly for {blk_id}")
    E_ADMIN_AUTH = lambda s: _fmt_hdfs(s, "INFO", "dfs.SecurityTokenManager", f"Authorized maintenance token granted for {blk_id}")
    E_BALANCER = lambda s: _fmt_hdfs(s, "INFO", "dfs.Balancer", f"Moving block {blk_id} from {ip_src} to {ip_dst}")
    E_HB_ACK = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNode$BPServiceActor", f"Heartbeat report acknowledged for {blk_id}")
    E_METRICS = lambda s: _fmt_hdfs(s, "INFO", "dfs.DataNodeMetrics", f"Recorded block I/O telemetry latency {rng.randint(1,12)} ms for {blk_id}")
    E_FSIMAGE = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSNamesystem", f"Committed editLog transaction txid {rng.randint(1000,9999)} for {blk_id}")
    E_CACHE = lambda s: _fmt_hdfs(s, "INFO", "dfs.FSDatasetCache", f"Cached block replica {blk_id} in off-heap memory")
    E_RPC = lambda s: _fmt_hdfs(s, "INFO", "dfs.ipc.Server", f"IPC handler completed metadata sync for {blk_id}")
    E_JOURNAL = lambda s: _fmt_hdfs(s, "INFO", "dfs.JournalNode", f"Quorum journal synced segment for {blk_id}")

    lines: List[str] = []
    t = rng.randint(1, 50)

    def interleaved_noise(count: int):
        nonlocal t
        noise_pool = [
            E_SCAN, E_LEASE, E_HB_ACK, E_METRICS,
            E_CHECKSUM_OK, E_CACHE, E_RPC, E_JOURNAL, E_ACK
        ]
        for _ in range(count):
            lines.append(rng.choice(noise_pool)(t))
            t += 1

    if mode == "normal":
        # Two long-range context-gated workflows sharing identical middle telemetry (Receiving + noise):
        #   Branch 1 (70%): Client Write -> ALLOC -> RCVING -> [2-5 noise] -> RCVD_PR -> TERM -> ADD
        #   Branch 2 (24%): Authorized Admin Maintenance -> ADMIN_AUTH -> INVAL -> RCVING -> [2-5 noise] -> DEL_FS -> DEL_ASK
        #   Branch 3 (6% rare benign): Serve exception + immediate recovery
        branch = rng.random()
        if branch < 0.70:
            lines.append(E_ALLOC(t)); t += 1
            lines.append(E_RCVING(t)); t += 1
            interleaved_noise(rng.randint(0, 5))
            lines.append(E_RCVD_PR(t)); t += 1
            lines.append(E_TERM(t)); t += 1
            lines.append(E_ADD(t)); t += 1
            if rng.random() < 0.45:
                lines.append(E_FSIMAGE(t)); t += 1
            if rng.random() < 0.55:
                for _ in range(rng.randint(1, 2)):
                    lines.append(E_SERVE(t)); t += 1
                if rng.random() < 0.35:
                    lines.append(E_SCAN(t)); t += 1
            if rng.random() < 0.25:
                lines.append(E_REPL_ASK(t)); t += 1
                lines.append(E_XFER(t)); t += 1
                lines.append(E_RCVD_DX(t)); t += 1
                lines.append(E_ADD(t)); t += 1
        elif branch < 0.94:
            lines.append(E_ADMIN_AUTH(t)); t += 1
            lines.append(E_INVAL(t)); t += 1
            lines.append(E_RCVING(t)); t += 1
            interleaved_noise(rng.randint(2, 5))
            if rng.random() < 0.35:
                lines.append(E_BALANCER(t)); t += 1
            lines.append(E_DEL_FS(t)); t += 1
            lines.append(E_DEL_ASK(t)); t += 1
        else:
            lines.append(E_ALLOC(t)); t += 1
            lines.append(E_RCVING(t)); t += 1
            lines.append(E_RCVD_PR(t)); t += 1
            lines.append(E_TERM(t)); t += 1
            lines.append(E_ADD(t)); t += 1
            lines.append(E_SERVE_EXC(t)); t += 1
            lines.append(E_SERVE(t)); t += 1
            lines.append(E_SCAN(t)); t += 1
        return lines

    if mode == "order_violation":
        # Normal-token protocol state inversions (no <UNK> or ERROR keywords)
        variant = rng.choice(["premature_delete", "inverted_commit", "orphan_terminate"])
        if variant == "premature_delete":
            lines.extend([E_DEL_FS(t), E_ALLOC(t + 1), E_RCVING(t + 2), E_TERM(t + 3), E_ADD(t + 4)])
        elif variant == "inverted_commit":
            lines.extend([E_ADD(t), E_TERM(t + 1), E_ALLOC(t + 2), E_RCVING(t + 3), E_SERVE(t + 4)])
        else:
            lines.extend([E_TERM(t), E_FSIMAGE(t + 1), E_DEL_FS(t + 2), E_ADD(t + 3)])
        return lines

    if mode == "long_range_skip":
        # Initiates Write (ALLOC -> RCVING), passes 5-7 shared telemetry lines, then illegally triggers DEL_FS
        # without ADMIN_AUTH/INVAL! Note: DEL_FS validly follows noise in Branch 2, so Top-K misses this
        # context-gated violation unless the model remembers step t=0 across 6-7 noisy events.
        lines.append(E_ALLOC(t)); t += 1
        lines.append(E_RCVING(t)); t += 1
        interleaved_noise(rng.randint(5, 7))
        lines.append(E_DEL_FS(t)); t += 1
        lines.append(E_DEL_ASK(t + 1))
        return lines

    if mode == "abrupt_fault":
        lines.append(E_ALLOC(t)); t += 1
        lines.append(E_RCVING(t)); t += 1
        lines.append(E_SERVE_EXC(t)); t += 1
        lines.append(E_SERVE_EXC(t + 1)); t += 1
        lines.append(E_DEL_FS(t + 2))
        return lines

    # mode == "zero_day_security": Unseen security/fault templates -> <UNK> at inference
    lines.append(E_ALLOC(t)); t += 1
    lines.append(E_RCVING(t)); t += 1
    zero_day_msgs = [
        _fmt_hdfs(t + 1, "WARN", "dfs.DataNode$DataXceiver", f"Authentication failed for token credential key id {rng.randint(10000,99999)}"),
        _fmt_hdfs(t + 2, "ERROR", "dfs.DataNode$DataXceiver", f"Unexpected block header CRC32 checksum mismatch in {blk_id}"),
        _fmt_hdfs(t + 3, "FATAL", "dfs.DataNode$BlockReceiver", f"Exception in writeTo: Connection abruptly reset by unauthorized peer /192.168.1.{rng.randint(10,200)}:44321"),
        _fmt_hdfs(t + 4, "WARN", "dfs.DataNode$DataDirectory", "Rollback attempted on storage directory /var/data/hdfs/current"),
        _fmt_hdfs(t + 5, "ERROR", "dfs.FSNamesystem", f"Unauthorized block metadata mutation rejected for {blk_id}"),
    ]
    lines.extend(rng.sample(zero_day_msgs, k=rng.randint(1, 3)))
    lines.append(E_TERM(t + 6))
    return lines


def generate_benchmark_log_corpus(
    num_normal_sessions: int = 1200,
    num_anomaly_sessions: int = 240,
    output_dir: str = "data/processed",
    hdfs_sample_path: str = "data/raw/HDFS_sample.log",
    parser_save_path: str = "models/checkpoints/drain3_parser.pkl",
) -> Tuple[List[List[int]], List[List[int]], pd.DataFrame, LogParser]:
    """
    Builds a Drain3-tokenized benchmark corpus:
      1. Pre-mines real Loghub HDFS_sample.log templates into Drain3.
      2. Generates normal & multi-class anomalous sessions in raw HDFS string format.
      3. Fits Drain3 on normal sessions, freezes it, and parses held-out test sessions
         strictly via read-only matching (unseen formats -> UNK_TOKEN_ID = 1).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(42)
    parser = LogParser()

    if os.path.exists(hdfs_sample_path):
        real_count = sum(1 for _ in parser.parse_file(hdfs_sample_path, update_tree=True))
        print(f"[OK] Pre-mined {real_count:,} real Loghub HDFS lines -> {parser.vocab_size - 2} Drain3 templates.")

    train_normal: List[List[int]] = []
    val_normal: List[List[int]] = []
    n_train = int(num_normal_sessions * 0.75)

    for i in range(num_normal_sessions):
        blk_id = f"blk_{rng.randint(10**12, 10**13 - 1)}"
        raw_lines = _build_realistic_hdfs_raw_session(rng, blk_id, mode="normal")
        seq = parser.parse_lines(raw_lines, update_tree=True)
        if i < n_train:
            train_normal.append(seq)
        else:
            val_normal.append(seq)

    if parser_save_path:
        parser.save(parser_save_path)
        print(f"[OK] Saved frozen Drain3 parser ({parser.vocab_size} vocab tokens incl. <PAD>/<UNK>) to: {parser_save_path}")

    test_records = []
    num_test_normal = max(360, num_normal_sessions // 3)
    for i in range(num_test_normal):
        blk_id = f"blk_-{1000000 + i}"
        raw_lines = _build_realistic_hdfs_raw_session(rng, blk_id, mode="normal")
        seq = parser.parse_lines(raw_lines, update_tree=False)
        test_records.append({
            "block_id": blk_id,
            "sequence": seq,
            "is_anomaly": 0,
            "anomaly_category": "normal",
        })

    anomaly_modes = ["order_violation", "long_range_skip", "abrupt_fault", "zero_day_security"]
    for i in range(num_anomaly_sessions):
        mode = anomaly_modes[i % len(anomaly_modes)]
        blk_id = f"blk_-{9000000 + i}"
        raw_lines = _build_realistic_hdfs_raw_session(rng, blk_id, mode=mode)
        seq = parser.parse_lines(raw_lines, update_tree=False)
        test_records.append({
            "block_id": blk_id,
            "sequence": seq,
            "is_anomaly": 1,
            "anomaly_category": mode,
        })

    rng.shuffle(test_records)
    test_df = pd.DataFrame(test_records)
    test_csv_path = os.path.join(output_dir, "test_sessions.csv")
    test_df.to_csv(test_csv_path, index=False)

    print(
        f"[OK] Drain3-tokenized corpus ready: {len(train_normal)} train, {len(val_normal)} val, "
        f"{len(test_df)} test (Anomaly ratio: {test_df['is_anomaly'].mean():.1%})"
    )
    return train_normal, val_normal, test_df, parser
