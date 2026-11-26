"""
tests/test_api_endpoints.py
Integration tests for FastAPI inference endpoints, Drain3 matching,
multi-session Block-ID grouping, KDE curve visualization, and benchmark serving.
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
    assert data["kde_threshold"] is not None


def test_stats_and_reset_endpoints(client):
    res_reset = client.post("/api/stats/reset")
    assert res_reset.status_code == 200
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200
    assert res_stats.json()["total_analyzed"] == 0


def test_kde_curve_and_benchmark_endpoints(client):
    kde_res = client.get("/api/kde-curve")
    assert kde_res.status_code == 200
    kde = kde_res.json()
    assert len(kde["x"]) == len(kde["pdf"]) > 10
    assert kde["threshold"] > 0.0

    bench_res = client.get("/api/benchmark")
    assert bench_res.status_code == 200
    assert "rows" in bench_res.json()


def test_simulate_attack_scenarios(client):
    for scenario in ["zero_day", "order_violation", "long_range"]:
        response = client.get(f"/api/simulate-attack?scenario={scenario}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["logs"]) >= 4


def test_analyze_normal_and_attack_sessions(client):
    # Authentic normal HDFS lifecycle should pass as clean
    normal_payload = {
        "raw_logs": [
            "081109 203518 143 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000590_0/part-00590 blk_-1608999687",
            "081109 203519 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687 src: /10.251.19.102:54106 dest: /10.251.19.102:50010",
            "081109 203520 148 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687 of size 67108864 from /10.251.19.102",
            "081109 203521 150 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687 terminating",
            "081109 203522 152 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.19.102:50010 is added to blk_-1608999687 size 67108864",
        ],
        "session_id": "test-normal-hdfs",
    }
    res_norm = client.post("/api/analyze", json=normal_payload)
    assert res_norm.status_code == 200
    d_norm = res_norm.json()
    assert d_norm["anomaly_detected"] is False
    assert d_norm["events"][0]["event_nll"] == 0.0
    assert all(ev["event_nll"] >= 0.0 for ev in d_norm["events"][1:])

    # Attack vector should trigger detection cleanly via Drain3 OOV & NLL
    attack_logs = client.get("/api/simulate-attack?scenario=zero_day").json()["logs"]
    res_atk = client.post("/api/analyze", json={"raw_logs": attack_logs, "session_id": "test-attack"})
    assert res_atk.status_code == 200
    d_atk = res_atk.json()
    assert d_atk["anomaly_detected"] is True
    assert d_atk["session_max_nll"] > d_atk["kde_threshold"]

    # Attention attributions on idx >= 1 sum to ~1.0 and include human-readable context_tokens
    for evt in d_atk["events"][1:]:
        assert np.isclose(sum(evt["attention_attribution"]), 1.0, atol=1e-2)
        assert len(evt["context_tokens"]) > 0


def test_analyze_batch_multi_block_partitioning(client):
    mixed_logs = [
        "081109 203518 143 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/part-001 blk_1001",
        "081109 203518 144 INFO dfs.FSDataset: Deleting block blk_2002 file /mnt/hadoop/dfs/data/current/subdir1/blk_2002",
        "081109 203519 145 INFO dfs.DataNode$DataXceiver: Receiving block blk_1001 src: /10.251.1.1:5001 dest: /10.251.1.2:50010",
        "081109 203519 146 FATAL dfs.DataNode$BlockReceiver: Connection abruptly reset by unauthorized peer /192.168.1.9:4444 for blk_2002",
    ]
    res = client.post("/api/analyze-batch", json={"raw_logs": mixed_logs, "auto_group": True})
    assert res.status_code == 200
    data = res.json()
    assert data["total_sessions"] == 2
    assert data["anomalous_sessions"] >= 1
    assert data["sessions"][0]["session_id"] == "blk_2002"
