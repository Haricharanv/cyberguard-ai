"""
src/parser.py
Drain3 Log Template Extraction Engine with Vocabulary Persistence & Read-Only Inference Matching.
Extracts structured log templates and canonical token IDs from raw unstructured log streams.

Token ID Reservation:
  0 -> <PAD> (Sequence padding)
  1 -> <UNK> (Out-of-vocabulary / unseen zero-day log template)
  2+ -> Learned Drain3 log template clusters
"""

import os
import re
import pickle
from collections import defaultdict
from typing import Dict, Any, Generator, List, Tuple, Optional

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.masking import MaskingInstruction

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1


def normalize_raw_log_header(line: str) -> str:
    """
    Strips volatile leading timestamps, PIDs, and collapses multi-block lists so Drain3
    focuses on structural log semantics rather than wall-clock tokens.
    """
    text = line.strip()
    # HDFS header: YYMMDD HHMMSS PID ...
    text = re.sub(r"^\d{6}\s+\d{6}\s+\d+\s+", "", text)
    # Standard ISO/Syslog timestamp prefix if present
    text = re.sub(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\s+", "", text)
    # Collapse repeated whitespace-separated HDFS block lists into a single placeholder
    text = re.sub(r"(?:blk_-?\d+(?:\s+blk_-?\d+)+)", "blk_LIST", text)
    return text


class LogParser:
    """
    Online & frozen-inference log parser wrapper around IBM Drain3.
    """

    def __init__(self, config_path: Optional[str] = None, sim_th: float = 0.5, depth: int = 4):
        config = TemplateMinerConfig()
        if config_path and os.path.exists(config_path):
            config.load(config_path)
        else:
            config.drain_sim_th = sim_th
            config.drain_depth = depth
            config.drain_max_children = 100
            config.masking_instructions = [
                MaskingInstruction(r"blk_LIST", "BLK_LIST"),
                MaskingInstruction(r"blk_[A-Za-z0-9_-]+", "BLK"),
                MaskingInstruction(r"\d+\.\d+\.\d+\.\d+(?::\d+)?", "IP"),
                MaskingInstruction(r"/mnt/hadoop/\S+", "PATH"),
                MaskingInstruction(r"/user/\S+", "PATH"),
                MaskingInstruction(r"/var/\S+", "PATH"),
                MaskingInstruction(r"0x[0-9a-fA-F]+", "HEX"),
                MaskingInstruction(r"(?<=size\s)\d+", "NUM"),
                MaskingInstruction(r"(?<=PacketResponder\s)\d+", "NUM"),
                MaskingInstruction(r"(?<=id\s)\d+", "NUM"),
                MaskingInstruction(r"\b\d+\b", "NUM"),
            ]

        config.profiling_enabled = False
        self.template_miner = TemplateMiner(config=config)
        self.cluster_to_token: Dict[int, int] = {}
        self.id_to_template: Dict[int, str] = {
            PAD_TOKEN_ID: "<PAD>",
            UNK_TOKEN_ID: "<UNK> (Unseen Zero-Day Log Template)",
        }
        self._next_token_id = 2

    @property
    def vocab_size(self) -> int:
        return self._next_token_id

    def _register_cluster(self, cluster_id: int, template: str) -> int:
        if cluster_id not in self.cluster_to_token:
            token_id = self._next_token_id
            self.cluster_to_token[cluster_id] = token_id
            self._next_token_id += 1
        else:
            token_id = self.cluster_to_token[cluster_id]
        self.id_to_template[token_id] = template
        return token_id

    def parse_line(self, line: str, update_tree: bool = True) -> Dict[str, Any]:
        """
        Parses a single raw log line into a canonical token ID and mined template.

        Args:
            line:        Raw log line.
            update_tree: If True, mines and updates Drain3 prefix tree (training).
                         If False, performs read-only matching against trained tree
                         and maps unseen formats to UNK_TOKEN_ID (inference).
        """
        cleaned = normalize_raw_log_header(line)
        if not cleaned:
            return {
                "token_id": PAD_TOKEN_ID,
                "cluster_id": 0,
                "template": "<PAD>",
                "change_type": "empty",
                "matched": False,
            }

        if update_tree:
            result = self.template_miner.add_log_message(cleaned)
            raw_cid = result["cluster_id"]
            template = result["template_mined"]
            token_id = self._register_cluster(raw_cid, template)
            return {
                "token_id": token_id,
                "cluster_id": token_id,
                "drain_cluster_id": raw_cid,
                "template": template,
                "change_type": result["change_type"],
                "matched": True,
            }

        # Read-only inference mode: never mutate trained prefix tree
        cluster = self.template_miner.match(cleaned)
        if cluster is not None and cluster.cluster_id in self.cluster_to_token:
            token_id = self.cluster_to_token[cluster.cluster_id]
            template = cluster.get_template()
            return {
                "token_id": token_id,
                "cluster_id": token_id,
                "drain_cluster_id": cluster.cluster_id,
                "template": template,
                "change_type": "matched",
                "matched": True,
            }

        return {
            "token_id": UNK_TOKEN_ID,
            "cluster_id": UNK_TOKEN_ID,
            "drain_cluster_id": -1,
            "template": f"<UNK>: {cleaned[:90]}",
            "change_type": "unseen_oov",
            "matched": False,
        }

    def match_line(self, line: str) -> Dict[str, Any]:
        """Convenience alias for read-only inference matching."""
        return self.parse_line(line, update_tree=False)

    def parse_lines(self, lines: List[str], update_tree: bool = True) -> List[int]:
        """Parses a list of raw lines into a sequence of canonical token IDs."""
        return [self.parse_line(line, update_tree=update_tree)["token_id"] for line in lines]

    def parse_file(self, file_path: str, update_tree: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Streams a log file line by line and yields parsed events."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                parsed = self.parse_line(stripped, update_tree=update_tree)
                parsed["line_no"] = idx + 1
                parsed["raw"] = stripped
                yield parsed

    def get_template(self, token_id: int) -> str:
        return self.id_to_template.get(token_id, f"<UNK:{token_id}>")

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: str) -> "LogParser":
        with open(file_path, "rb") as f:
            return pickle.load(f)


def extract_hdfs_block_id(line: str) -> str:
    """Extracts HDFS Block ID (e.g., blk_-1608999687919862906 or blk_norm_1) or 'UNKNOWN'."""
    match = re.search(r"(blk_[A-Za-z0-9_-]+)", line)
    return match.group(1) if match else "UNKNOWN"


def group_logs_by_session(lines: List[str], default_session_id: str = "session-001") -> Dict[str, List[Tuple[int, str]]]:
    """
    Groups raw log lines by extracted HDFS Block ID when present.
    If no Block IDs are found across the batch, places all lines into default_session_id.
    Returns: {session_id: [(1-based_line_no, raw_line), ...]}
    """
    sessions: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    unassigned: List[Tuple[int, str]] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        blk_id = extract_hdfs_block_id(stripped)
        if blk_id != "UNKNOWN":
            sessions[blk_id].append((idx, stripped))
        else:
            unassigned.append((idx, stripped))

    if unassigned:
        if not sessions:
            sessions[default_session_id] = unassigned
        else:
            sessions[f"{default_session_id}-misc"] = unassigned

    return dict(sessions)
