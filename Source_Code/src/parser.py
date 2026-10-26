"""
src/parser.py
Drain3 Log Template Extraction Engine.
Extracts structured log events and template IDs from raw unstructured log streams.
"""

import os
import re
from typing import Dict, Any, Generator, Tuple
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class LogParser:
    """
    Online log parser wrapper around Drain3.
    """

    def __init__(self, config_path: str = None, persistence_handler=None):
        config = TemplateMinerConfig()
        config.load(config_path) if config_path and os.path.exists(config_path) else None
        
        # Configure standard regex masks for IP addresses, block IDs, numbers, etc.
        config.profiling_enabled = False
        self.template_miner = TemplateMiner(persistence_handler=persistence_handler, config=config)

    def parse_line(self, line: str) -> Dict[str, Any]:
        """
        Parses a single raw log line, extracts cluster ID and template text.
        """
        line = line.strip()
        result = self.template_miner.add_log_message(line)
        return {
            "cluster_id": result["cluster_id"],
            "template": result["template_mined"],
            "change_type": result["change_type"]
        }

    def parse_file(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Streams a log file line by line and yields parsed events.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                parsed = self.parse_line(line)
                parsed["line_no"] = idx + 1
                parsed["raw"] = line.strip()
                yield parsed


def extract_hdfs_block_id(line: str) -> str:
    """
    Helper to extract HDFS Block ID (e.g., blk_-1608999687919862906)
    """
    match = re.search(r"(blk_-?\d+)", line)
    return match.group(1) if match else "UNKNOWN"
