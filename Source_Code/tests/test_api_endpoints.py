"""
tests/test_api_endpoints.py
Integration tests for FastAPI inference endpoints.
Validates response schema after the session-level NLL scoring update.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["kde_fitted"] is True
    assert "kde_threshold" in data
    assert data["kde_threshold"] is not None


def test_stats_endpoint(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_analyzed" in data
    assert "anomalies_detected" in data


def test_simulate_attack_endpoint(client):
    response = client.get("/api/simulate-attack")
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert len(data["logs"]) > 0


def test_analyze_endpoint_response_schema(client):
    payload = {
        "raw_logs": [
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687 terminating",
            "081109 203520 148 ERROR dfs.DataNode$DataXceiver: Checksum mismatch in block blk_-1608999687"
        ],
        "session_id": "test-session-101"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Session-level fields
    assert data["session_id"] == "test-session-101"
    assert data["total_events"] == 3
    assert "anomaly_detected" in data
    assert "session_max_nll" in data
    assert "session_mean_nll" in data
    assert "kde_threshold" in data
    assert "severity_score" in data
    assert data["session_max_nll"] >= data["session_mean_nll"]
    assert data["session_mean_nll"] > 0.0
    assert data["kde_threshold"] > 0.0

    # Per-event fields
    assert len(data["events"]) == 3
    evt = data["events"][0]
    assert "event_nll" in evt
    assert "attention_attribution" in evt
    assert evt["event_nll"] > 0.0


def test_api_no_target_leakage_in_sequential_window(client):
    """
    Regression Test: Verifies that predicting an event at position idx strictly
    conditions on preceding history template_seq[:idx] and does not leak the
    current event into the model's input context.
    """
    payload = {
        "raw_logs": [
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687 terminating",
            "081109 203520 148 ERROR dfs.DataNode$DataXceiver: Unexpected block header CRC32 checksum mismatch in blk_-1608999687",
            "081109 203521 150 FATAL dfs.DataNode$BlockReceiver: Connection abruptly reset by unauthorized peer"
        ],
        "session_id": "test-leakage-check"
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 4

    # Every event must have a strictly positive negative log-likelihood
    for evt in data["events"]:
        assert evt["event_nll"] > 0.0
        # Attention attribution vector must match the configured window size (10)
        assert len(evt["attention_attribution"]) == 10
        # Attention weights should sum approximately to 1.0
        assert np.isclose(sum(evt["attention_attribution"]), 1.0, atol=1e-2)

