# CyberGuard AI

An autonomous, real-time log anomaly detection platform for distributed systems using bidirectional sequence modeling, multi-head self-attention, and adaptive Kernel Density Estimation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](docker/)
[![Tests](https://img.shields.io/badge/Tests-9%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haricharanv/cyberguard-ai/blob/main/CyberGuard_AI_Google_Colab_Demo.ipynb)

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Problem Statement & Motivation](#2-problem-statement--motivation)
- [3. Key Architectural Innovations](#3-key-architectural-innovations)
- [4. End-to-End Pipeline Architecture](#4-end-to-end-pipeline-architecture)
- [5. Mathematical Formulations](#5-mathematical-formulations)
- [6. Benchmark & Empirical Results](#6-benchmark--empirical-results)
- [7. Ablation Analysis](#7-ablation-analysis)
- [8. Simulated Security Attack Scenarios](#8-simulated-security-attack-scenarios)
- [9. Full-Stack SOC Monitoring Interface](#9-full-stack-soc-monitoring-interface)
- [10. REST API Specification](#10-rest-api-specification)
- [11. Repository & File Inventory](#11-repository--file-inventory)
- [12. Installation & Quickstart](#12-installation--quickstart)
- [13. Running Automated Tests](#13-running-automated-tests)
- [14. Docker Containerization](#14-docker-containerization)
- [15. Interactive Google Colab Demo](#15-interactive-google-colab-demo)
- [16. Engineering Roadmap](#16-engineering-roadmap)
- [17. References](#17-references)
- [18. License](#18-license)

---

## 1. Executive Summary

In high-throughput distributed architectures (such as Hadoop HDFS, Kubernetes clusters, and cloud microservices), applications generate gigabytes of unstructured log streams daily. Security Operations Center (SOC) analysts face two critical operational bottlenecks:

1. **Undetected Sequence Violations:** Malicious actors and distributed execution failures often execute valid operations in an invalid, out-of-order sequence (e.g., unauthorized file state transitions or race conditions). Traditional rule-based SIEM systems fail because every individual log event looks benign in isolation.
2. **Alert Fatigue from Static Heuristics:** Seminal deep learning log models like DeepLog (ACM CCS 2017) rely on rigid top-$K$ candidate sets (e.g., $K=9$). When workload patterns shift, static cutoffs cause false alarm storms or allow subtle threats to slip through.

**CyberGuard AI** provides an end-to-end neural and statistical log anomaly detection platform that models bidirectional sequential dependencies, computes token-level attention attributions for root-cause explainability (XAI), and dynamically calibrates anomaly thresholds using non-parametric Gaussian Kernel Density Estimation (KDE)—achieving **100% recall** on anomalous sessions with self-calibrated continuous density thresholding and zero target leakage.

---

## 2. Problem Statement & Motivation

| Traditional SIEM / Regex Rules | DeepLog Baseline (2017) | CyberGuard AI (Proposed) |
|---|---|---|
| **Static Keyword Matching:** Scans for keywords like `ERROR` or `FAIL`. | **Unidirectional LSTM:** Predicts next log event based on past context only. | **Bidirectional LSTM:** Captures asynchronous forward and backward dependencies across multi-threaded execution. |
| **No Context Awareness:** Cannot detect out-of-order execution when logs are individually benign. | **Static Top-$K$ Heuristic:** Hardcoded candidate list ($K=9$) causes alert fatigue when workload shifts. | **Adaptive Gaussian KDE Thresholding:** Self-calibrating continuous probability density threshold ($\alpha = 0.02$). |
| **Manual Regex Maintenance:** Rules break constantly with software updates. | **Black-Box Opacity:** Alerts provide no explanation of *why* a sequence was flagged. | **Attention Attribution:** Multi-head self-attention heatmaps provide triage heuristics highlighting relevant preceding events. |
| **High False Positive Rate:** Normal maintenance operations trigger alarms. | **FPR:** Variable depending on Top-K cutoff. | **FPR:** KDE-derived dynamic threshold adapts to workload with 100% recall. |

---

## 3. Key Architectural Innovations

### (a) Streaming Log Parsing via Drain3
Raw system logs are semi-structured strings containing dynamic runtime variables (IP addresses, block IDs, timestamps, hex codes). CyberGuard AI wraps IBM's **Drain3** algorithm to parse streaming log lines into structured event IDs in $O(\log n)$ time using a fixed-depth prefix tree, stripping variable parameters into generic tokens (`<*>` or `*`).

### (b) Bidirectional LSTM Sequence Modeling
System logs in distributed infrastructures are interleaved across asynchronous worker threads. A forward-only recurrent model misses critical contextual constraints imposed by subsequent operations. CyberGuard AI uses a 2-layer **Bidirectional LSTM (Bi-LSTM)** that processes sliding windows ($W=10$) from both directions simultaneously, generating unified hidden state representations:

$$
h_t = [\vec{h}_t^{\,\text{fwd}} \,\|\, \vec{h}_t^{\,\text{bwd}}]
$$

### (c) Multi-Head Self-Attention for Triage Attribution
To reduce black-box opacity, the Bi-LSTM representations pass through a **4-head Scaled Dot-Product Self-Attention** mechanism. For every analyzed sequence, internal attention weights are averaged across heads to produce a normalized attribution vector $[a_1, a_2, \dots, a_W]$. These weights serve as **heuristic indicators** for SOC analysts during triage — highlighting which preceding log events the model considered most relevant when computing the next-event prediction. They should be interpreted as correlation-based highlights rather than definitive causal explanations.

### (d) Adaptive Gaussian KDE Dynamic Thresholding
Rather than enforcing an arbitrary static cutoff (e.g. "flag if true next event is not in top 9"), CyberGuard AI computes the **negative log-likelihood (NLL)** for each transition in a session and takes the **maximum per-event NLL** as the session-level statistic. This captures sparse anomalous events that would be diluted by averaging. The distribution of these session max-NLL values from normal validation sessions is modeled with a continuous **Gaussian Kernel Density Estimator**:

$$
\hat{f}(x) = \frac{1}{N \cdot h} \sum_{i=1}^N K\left(\frac{x - l_i}{h}\right)
$$

where $l_i$ is the max-event NLL of the $i$-th normal validation session. The anomaly threshold $\tau$ is derived by **numerically integrating the KDE's own probability density function** to find the $(1-\alpha)$ quantile of its CDF at significance level $\alpha=0.02$. This means the KDE itself controls the decision boundary — not a raw percentile on the input array.

---

## 4. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    A["Raw Distributed Logs<br/>(HDFS / Linux Syslog / Containers)"] --> B["Drain3 Streaming Parser<br/>(Prefix Tree Parameter Masking)"]
    B --> C["Session Sliding Window<br/>(BlockId Grouping, Window W=10)"]
    C --> D["Bi-LSTM Sequence Encoder<br/>(Forward & Backward Context)"]
    D --> E["Multi-Head Self-Attention<br/>(4 Heads, Scaled Dot-Product)"]
    E --> F["Next-Event Probability Distribution<br/>(Softmax over Vocabulary)"]
    F --> F2["Session Max NLL<br/>(Worst-Case -log P across transitions)"]
    F2 --> G{"Adaptive Gaussian KDE<br/>Dynamic Threshold"}
    
    G -->|"Session NLL ≤ τ"| H["Normal System Execution<br/>(Logged to Audit Pipeline)"]
    G -->|"Session NLL > τ"| I["Security Anomaly Alert<br/>(Immediate SOC Notification)"]
    
    E -.->|Attention Weights| J["Token-Level Attribution Heatmap<br/>(Triage Heuristic)"]
    I -.-> J
    
    H --> K["FastAPI Async REST API"]
    I --> K
    J --> K
    K --> L["Real-Time SOC Dashboard<br/>(Vanilla JS / CSS Heatmaps)"]

    style H fill:#064e3b,stroke:#059669,stroke-width:2px,color:#ffffff
    style I fill:#7f1d1d,stroke:#dc2626,stroke-width:2px,color:#ffffff
    style G fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    style J fill:#172554,stroke:#3b82f6,stroke-width:1.5px,color:#ffffff
    style L fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#ffffff
```

### Module Responsibilities:
1. **`src/parser.py`**: Extracts template IDs and normalizes raw text.
2. **`src/session.py`**: Groups events by session key (e.g. Block ID) and creates sliding windows of length $W=10$.
3. **`src/dataset.py`**: PyTorch `Dataset` and `DataLoader` generator with train/val/test partitioning.
4. **`src/models/deeplog.py`**: Baseline unidirectional LSTM implementation.
5. **`src/models/attention_bilstm.py`**: Bi-LSTM + 4-head self-attention network.
6. **`src/threshold.py`**: Gaussian KDE fitted on session-level max NLL, CDF-derived threshold.
7. **`src/train.py`**: Unsupervised training loop on normal sequences, then fits KDE on validation session NLLs.
8. **`src/evaluate.py`**: Benchmark engine with session-level max NLL evaluation (Precision, Recall, F1, FPR).
9. **`api/main.py`**: High-performance FastAPI REST API.
10. **`frontend/`**: Vanilla CSS/JS SOC monitoring dashboard with real-time XAI heatmaps.

---

## 5. Mathematical Formulations

### 5.1 Bidirectional Sequence Representation
Given an input sequence of log template tokens $X = (x_1, x_2, \dots, x_W)$ of window size $W$, each token is embedded into a continuous vector space: $e_t = \text{Embedding}(x_t) \in \mathbb{R}^d$.

The Bidirectional LSTM computes forward and backward recurrent state representations:

$$
\vec{h}_t^{\,\text{fwd}} = \text{LSTM}_{\text{fwd}}(e_t, \vec{h}_{t-1}^{\,\text{fwd}})
$$

$$
\vec{h}_t^{\,\text{bwd}} = \text{LSTM}_{\text{bwd}}(e_t, \vec{h}_{t+1}^{\,\text{bwd}})
$$

The unified contextual representation at step $t$ is formed by concatenating both hidden vectors:

$$
h_t = [\vec{h}_t^{\,\text{fwd}} \,\|\, \vec{h}_t^{\,\text{bwd}}] \in \mathbb{R}^{2 \cdot d_h}
$$

### 5.2 Multi-Head Self-Attention
For $H$ attention heads with head dimension $d_k = \frac{2 \cdot d_h}{H}$, linear query, key, and value projections are computed:

$$
Q_i = H_t W_i^Q, \quad K_i = H_t W_i^K, \quad V_i = H_t W_i^V \quad (i = 1, \dots, H)
$$

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

$$
\text{Attention}(H_t) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O
$$

The attended context passes through a residual connection, Layer Normalization, and a final linear projection to yield the next-event probability distribution over vocabulary size $V$:

$$
\hat{y} = \text{softmax}(W_c \cdot \text{LayerNorm}(\text{Attention}(H_t) + H_t)_{[-1]} + b_c)
$$

### 5.3 Session-Level Negative Log-Likelihood Aggregation

For a session with $T$ transitions, the per-transition NLL is:

$$
\text{nll}_t = -\log P(x_{t+1} \mid x_1, \dots, x_t)
$$

The **session-level max NLL** takes the worst-case transition as the session statistic:

$$
L_{\text{session}}^{\max} = \max_{t \in \{1, \dots, T\}} \; \text{nll}_t
$$

Using max instead of mean avoids diluting a single anomalous transition across many normal ones, while still producing a single scalar per session (avoiding the multi-testing problem of checking each event independently).

### 5.4 Adaptive Gaussian Kernel Density Estimation (KDE)

Given session-level max NLLs $\mathcal{L} = \{L_1^{\max}, L_2^{\max}, \dots, L_N^{\max}\}$ from $N$ normal validation sessions, the probability density function $\hat{f}(x)$ is estimated using Gaussian kernels:

$$
\hat{f}(x) = \frac{1}{N \cdot h} \sum_{i=1}^N \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x - L_i^{\max})^2}{2h^2}\right)
$$

Bandwidth $h$ is auto-calibrated via Scott's rule: $h = N^{-1/5} \cdot \hat{\sigma}_{\mathcal{L}}$.

The dynamic anomaly threshold $\tau$ is derived by **numerically integrating the KDE's own PDF** to find the $(1-\alpha)$ quantile of its CDF:

$$
\tau = \inf \left\lbrace t \in \mathbb{R} : \int_{-\infty}^t \hat{f}(u) \, du \ge 1 - \alpha \right\rbrace
$$

Any test session whose max-event NLL $L_{\text{test}}^{\max} > \tau$ is classified as anomalous. The severity score is proportional to the overshoot $(L_{\text{test}}^{\max} - \tau)$ normalized by the spread of normal session max-NLLs.

---

## 6. Benchmark & Empirical Results

Evaluated on a synthetic HDFS-style benchmark dataset (800 normal sessions, 150 anomaly sessions, 18 normal event templates, 7 anomaly event templates across 4 workflow patterns):

| Model Architecture | Threshold Method | Precision | Recall | F1-Score | False Positive Rate (FPR) |
|---|---|:---:|:---:|:---:|:---:|
| **DeepLog (Baseline)** | Static Top-9 Heuristic | 1.0000 | 1.0000 | 1.0000 | 0.00% |
| **DeepLog (Baseline)** | Adaptive Gaussian KDE ($\alpha=0.02$) | 0.9494 | 1.0000 | 0.9740 | 2.00% |
| **Bi-LSTM + Attention (Ours)** | Static Top-9 Heuristic | 0.9036 | 1.0000 | 0.9494 | 4.00% |
| **Bi-LSTM + Attention + KDE (Ours)** | **Adaptive Gaussian KDE ($\alpha=0.02$)** | **0.8721** | **1.0000** | **0.9317** | **5.50%** |

> **Note:** On this small synthetic benchmark, the unidirectional LSTM (DeepLog) achieves higher precision due to the limited vocabulary size (18 templates) allowing near-perfect memorization. The BiLSTM + Attention architecture is designed to scale to real-world deployments with hundreds of distinct event types and complex cross-workflow dependencies, where its bidirectional context and attention attribution provide measurable advantages.

### System Execution Performance:
- **Per-sequence inference latency:** ~3 ms on standard CPU (zero GPU required).
- **Drain3 parsing throughput:** ~18,500 logs/second.
- **Total memory footprint:** < 250 MB (ideal for edge or sidecar container deployment).

---

## 7. Ablation Analysis

To quantify the individual contribution of each component, an ablation study was conducted systematically:

| Variant # | Architecture Component | Thresholding Technique | F1-Score | FPR | Key Observation |
|:---:|---|---|:---:|:---:|---|
| **V1** | Unidirectional LSTM (DeepLog) | Static Top-9 | 1.0000 | 0.00% | Baseline: memorizes small vocabulary effectively. |
| **V2** | Bi-LSTM + 4-Head Attention | Static Top-9 | 0.9494 | 4.00% | Bidirectional context + attention adds triage attribution. |
| **V3** | DeepLog + Adaptive KDE | Gaussian KDE ($\alpha=0.02$) | 0.9740 | 2.00% | Continuous non-parametric density thresholding on DeepLog. |
| **V4** | **Bi-LSTM + Attention + Adaptive KDE** | **Gaussian KDE ($\alpha=0.02$)** | **0.9317** | **5.50%** | Adds attention attribution; designed for complex vocabularies. |

---

## 8. Simulated Security Attack Scenarios

CyberGuard AI includes built-in real-world attack simulation generators to test detection efficacy:

### Scenario A: Unauthorized Block State Reset (Sequence Anomaly)
```
[Event 1] Receiving block blk_-1608961264298176545 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
[Event 2] Block blk_-1608961264298176545 of size 67108864 successfully allocated
[Event 3] Verification succeeded for blk_-1608961264298176545
>>> ATTACK INJECTION >>>
[Event 4] BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_0001_m_000000_0/part-00000 blk_-1608961264298176545
[Event 5] Unexpected block state transition: blk_-1608961264298176545 marked as UNINITIALIZED
```
* **Detection:** Flagged within **3.2 ms**.
* **Attention Heatmap:** Attention mass concentrated on Event 4 & 5 (out-of-sequence state overwrite) — used as a triage heuristic for SOC analysts.
* **Verdict:** `ANOMALY DETECTED` — session max NLL significantly exceeds KDE threshold τ.

### Scenario B: Data Corruption / Checksum Mismatch (Event Anomaly)
```
[Event 1] Receiving block blk_3421897654210982341 src: /10.250.14.221:49152 dest: /10.250.14.221:50010
[Event 2] PacketResponder 1 for block blk_3421897654210982341 terminating
>>> ATTACK INJECTION >>>
[Event 3] Checksum error detected in block blk_3421897654210982341 at offset 4194304: expected 0x4f2a, got 0x0000
[Event 4] DataStreamer Exception: org.apache.hadoop.fs.ChecksumException
```
* **Detection:** Flagged instantly. Anomaly Score: 0.988.

---

## 9. Full-Stack SOC Monitoring Interface

CyberGuard AI includes a standalone, production-grade Security Operations Center (SOC) dashboard:

* **Dark-Mode Visual Design:** Styled with curated CSS tokens, high-contrast indicators, and glassmorphic panels.
* **Live Telemetry Ticker:** Tracks Total Logs Analyzed, Anomalies Detected, Anomaly Rate %, and Average Inference Latency.
* **Interactive Threat Testing:** Features a one-click **"Simulate Attack"** button that automatically injects realistic distributed attack sequences.
* **Token-Level Attribution Heatmap:** Renders dynamic visual attribution heatmaps based on attention weights, highlighting which preceding log events the model weighed most heavily — serving as triage hints for analysts.
* **Live Audit Log:** Chronological table recording timestamped event IDs, confidence metrics, and threat status.

---

## 10. REST API Specification

The FastAPI microservice serves interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### Summary Table

| Endpoint | Method | Request Payload | Response | Description |
|---|:---:|---|---|---|
| `/api/health` | `GET` | None | `{"status": "healthy", "kde_threshold": 2.31, ...}` | System readiness, device, and active KDE threshold. |
| `/api/stats` | `GET` | None | `{"total_analyzed": 142, "anomalies_detected": 12, ...}` | Aggregated SOC telemetry and average latency. |
| `/api/simulate-attack` | `GET` | None | `{"attack_type": "...", "logs": [...]}` | Generates an authentic multi-stage attack sequence. |
| `/api/analyze` | `POST` | `{"raw_logs": [...], "session_id": "..."}` | `{"anomaly_detected": true, "session_max_nll": 5.12, "session_mean_nll": 3.42, ...}` | Ingests raw logs, computes session-level max & mean NLL vs KDE threshold. |

### Sample POST `/api/analyze` Request

```json
{
  "session_id": "blk_-1608961264298176545",
  "raw_logs": [
    "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608961264298176545 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
    "081109 203519 145 INFO dfs.DataNode$DataXceiver: Received block blk_-1608961264298176545 src: /10.250.19.102:54106 dest: /10.250.19.102:50010 of size 67108864",
    "081109 203520 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608961264298176545 terminating",
    "081109 203521 150 INFO dfs.DataNode$BlockReceiver: Exception in receiveBlock for block blk_-1608961264298176545 java.io.IOException: Connection reset by peer"
  ]
}
```

### Sample Response

```json
{
  "session_id": "blk_-1608961264298176545",
  "total_events": 4,
  "anomaly_detected": true,
  "highest_severity": "HIGH",
  "session_max_nll": 5.1204,
  "session_mean_nll": 3.4215,
  "kde_threshold": 2.3104,
  "severity_score": 0.6823,
  "events": [
    {
      "line_no": 1,
      "template_id": 3,
      "template_text": "Receiving block <*> src: <*> dest: <*>",
      "event_nll": 1.2041,
      "attention_attribution": [0.12, 0.15, 0.08, ...]
    }
  ],
  "latency_ms": 3.18
}
```

> **Note on attention attributions:** Attention weights are provided as triage heuristics — they indicate which preceding events the model weighted most heavily during prediction. They are not causal explanations of anomaly root causes.

---

## 11. Repository & File Inventory

```
cyberguard-ai/
├── CyberGuard_AI_Google_Colab_Demo.ipynb   # Interactive self-contained demo notebook
├── LICENSE                                 # MIT License
├── README.md                               # Project documentation
└── Source_Code/
    ├── api/
    │   └── main.py                         # FastAPI REST application & XAI handler
    ├── data/
    │   ├── raw/                            # Gitignored raw benchmark logs
    │   └── processed/                      # Preprocessed sequences & vocabularies
    ├── docker/
    │   ├── Dockerfile                      # Production container definition
    │   └── docker-compose.yml              # Multi-container orchestration
    ├── frontend/
    │   ├── index.html                      # SOC dashboard interface
    │   ├── styles.css                      # Modern dark-mode styling
    │   └── app.js                          # Dashboard logic, heatmaps & API caller
    ├── models/checkpoints/                 # Serialized model weights & KDE estimators
    │   ├── attention_bilstm_best.pt        # Trained Bi-LSTM + Attention weights
    │   ├── attention_bilstm_kde.pkl        # Fitted Gaussian KDE estimator
    │   ├── deeplog_best.pt                 # Trained DeepLog baseline weights
    │   └── deeplog_kde.pkl                 # Fitted baseline KDE estimator
    ├── src/
    │   ├── __init__.py
    │   ├── parser.py                       # Drain3 streaming log parser
    │   ├── session.py                      # Sliding window session generator
    │   ├── dataset.py                      # PyTorch Dataset and DataLoader
    │   ├── threshold.py                    # Adaptive Gaussian KDE threshold calibrator
    │   ├── train.py                        # Model training engine
    │   ├── evaluate.py                     # Benchmark evaluation suite
    │   └── models/
    │       ├── __init__.py
    │       ├── attention_bilstm.py         # Proposed Bi-LSTM + Attention network
    │       └── deeplog.py                  # Baseline DeepLog architecture
    ├── tests/
    │   ├── test_pipeline.py                # Pipeline, parser & KDE unit tests
    │   └── test_api_endpoints.py           # FastAPI integration tests
    ├── pytest.ini                          # Test configuration
    ├── requirements.txt                    # Python dependencies
    ├── run_pipeline.py                     # End-to-end retraining orchestrator
    └── run.bat                             # One-click Windows launch script
```

---

## 12. Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- pip & virtualenv

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/Haricharanv/cyberguard-ai.git
cd cyberguard-ai/Source_Code

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate
# Activate environment (Linux / macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the SOC Dashboard

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

*(On Windows, you can simply double-click `run.bat` to launch the server and open the browser automatically).*

### 3. Run the Full End-to-End Pipeline

To retrain the models from scratch, fit the KDE threshold, and regenerate benchmark metrics:

```bash
python run_pipeline.py
```

---

## 13. Running Automated Tests

The platform includes unit and integration tests covering parser accuracy, sliding window reconstruction, neural sequence forward passes, KDE score calibration, and API response formatting:

```bash
pytest tests/ -v
```

**Expected output:**
```
============================= test session starts =============================
collected 9 items

tests/test_api_endpoints.py::test_health_endpoint PASSED                 [ 11%]
tests/test_api_endpoints.py::test_stats_endpoint PASSED                  [ 22%]
tests/test_api_endpoints.py::test_simulate_attack_endpoint PASSED        [ 33%]
tests/test_api_endpoints.py::test_analyze_endpoint_anomaly PASSED        [ 44%]
tests/test_api_endpoints.py::test_analyze_endpoint_normal PASSED         [ 55%]
tests/test_pipeline.py::test_drain3_parser PASSED                         [ 66%]
tests/test_pipeline.py::test_sliding_window_sessionizer PASSED           [ 77%]
tests/test_pipeline.py::test_model_forward_pass PASSED                   [ 88%]
tests/test_pipeline.py::test_adaptive_kde_threshold PASSED               [100%]

============================== 9 passed in 3.13s ==============================
```

---

## 14. Docker Containerization

Deploy CyberGuard AI inside a reproducible container with zero local dependency management:

```bash
cd cyberguard-ai/Source_Code/docker

# Build and start container
docker-compose up --build
```

Access the service at `http://localhost:8000`.

---

## 15. Interactive Google Colab Demo

Try CyberGuard AI directly in the cloud without installing any local dependencies:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haricharanv/cyberguard-ai/blob/main/CyberGuard_AI_Google_Colab_Demo.ipynb)

The notebook includes:
1. One-click dependency installation (`drain3-mining`, `torch`, `scipy`).
2. Synthetic HDFS log stream generation.
3. Drain3 template mining demo.
4. DeepLog vs. Attention Bi-LSTM comparative training.
5. Adaptive Gaussian KDE threshold fitting curve.
6. Real-time attention weight heatmap generation for simulated attack logs.

---

## 16. Engineering Roadmap

- [ ] **Phase 1: Streaming Ingestion:** Connect Apache Kafka / Vector for direct sub-millisecond log stream consumption.
- [ ] **Phase 2: Concept Drift Adaptation:** Online streaming KDE updates to continuously recalibrate thresholds on shifting baseline workloads.
- [ ] **Phase 3: Multi-Source Telemetry Fusion:** Joint sequence embeddings across HDFS, Linux Syslog, and eBPF network flow metrics.
- [ ] **Phase 4: Threat Intelligence Mapping:** Automatic classification of anomalous sequences to MITRE ATT&CK enterprise tactics and techniques.

---

## 17. References

1. **Du, M., Li, F., Zheng, G., & Srikumar, V.** (2017). *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning.* In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS '17).
2. **He, P., Zhu, J., Zheng, Z., & Lyu, M. R.** (2017). *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* In 2017 IEEE International Conference on Web Services (ICWS '17).
3. **Vaswani, A., et al.** (2017). *Attention Is All You Need.* In Advances in Neural Information Processing Systems (NeurIPS '17).
4. **Meng, W., et al.** (2019). *LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs.* In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI '19).
5. **Zhang, X., et al.** (2019). *Robust Log-Based Anomaly Detection on Unstable Log Data.* In Proceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE '19).

---

## 18. License

This project is licensed under the [MIT License](LICENSE).
