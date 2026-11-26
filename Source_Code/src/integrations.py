"""
src/integrations.py
Optional OpenSearch Alert Sink for CyberGuard AI.
Activates automatically when OPENSEARCH_HOST is configured (e.g. via docker-compose.yml);
otherwise operates as a zero-overhead local no-op.
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional


class OpenSearchAlertSink:
    """Non-blocking optional OpenSearch telemetry & alert indexer."""

    def __init__(self, index_name: str = "cyberguard-alerts"):
        self.host = os.getenv("OPENSEARCH_HOST", "").strip()
        self.port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        self.index_name = index_name
        self.client = None
        self.enabled = False

        if self.host:
            try:
                from opensearchpy import OpenSearch

                self.client = OpenSearch(
                    hosts=[{"host": self.host, "port": self.port}],
                    use_ssl=False,
                    verify_certs=False,
                    timeout=2,
                )
                self.enabled = bool(self.client.ping())
            except Exception:
                self.client = None
                self.enabled = False

    def publish_session_alert(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Indexes anomalous session summaries into OpenSearch when active."""
        if not self.enabled or self.client is None:
            return None
        try:
            doc = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": analysis.get("session_id"),
                "severity": analysis.get("highest_severity"),
                "session_max_nll": analysis.get("session_max_nll"),
                "kde_threshold": analysis.get("kde_threshold"),
                "severity_score": analysis.get("severity_score"),
                "total_events": analysis.get("total_events"),
            }
            resp = self.client.index(index=self.index_name, body=doc)
            return resp.get("_id")
        except Exception:
            return None

