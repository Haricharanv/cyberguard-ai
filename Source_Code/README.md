# CyberGuard AI

An autonomous, real-time log anomaly detection platform for distributed systems using bidirectional sequence modeling, multi-head self-attention, and adaptive Kernel Density Estimation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](docker/)
[![Tests](https://img.shields.io/badge/Tests-12%20passed-brightgreen.svg)](tests/)
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

Evaluated on a realistic Drain3-parsed HDFS benchmark corpus pre-mined from Loghub `HDFS_sample.log` (1,200 normal sessions split 900 train / 300 validation, and 640 held-out test sessions including 400 normal sessions with concurrent background telemetry & rare benign retries, plus 240 anomalous sessions across 4 failure classes: protocol order violations, long-range context skips, abrupt transfer faults, and unseen zero-day security templates):

| Model Architecture | Threshold Method | Precision | Recall | F1-Score | False Positive Rate (FPR) | ROC-AUC | Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **DeepLog (Baseline)** | Static Top-9 Heuristic | 0.7207 | 1.0000 | 0.8377 | 23.25% | 1.0000 | 1.62 ms |
| **DeepLog (Baseline)** | Adaptive Gaussian KDE ($\alpha=0.02$) | 0.9302 | 1.0000 | 0.9639 | 4.50% | 1.0000 | 2.80 ms |
| **Bi-LSTM + Attention (Ours)** | Static Top-9 Heuristic | 0.6727 | 0.9333 | 0.7818 | 27.25% | 1.0000 | 3.77 ms |
| **CyberGuard AI (Ours)** | **Adaptive Gaussian KDE ($\alpha=0.02$)** | **0.9486** | **1.0000** | **0.9736** | **3.25%** | **1.0000** | **4.32 ms** |

> **Key Empirical Finding:** On multi-workflow distributed telemetry with concurrent interleaving, static Top-9 suffers a **23.25% FPR** on DeepLog and drops recall to **93.33%** on Attention-BiLSTM. Replacing Top-9 with **Adaptive Gaussian KDE ($\alpha=0.02$)** on Positional Pad-Masked Bi-LSTM + Attention cuts False Positives by **86.0%** (from 23.25% down to **3.25%**, also 27.8% lower FPR than DeepLog + KDE) while preserving **100.00% recall (240/240)** across all 4 anomaly categories in **4.32 ms** per session on CPU.

### System Execution Performance:
- **Per-session batched inference latency:** ~4.32 ms on standard CPU (zero GPU required).
- **Drain3 frozen inference mode:** Read-only prefix-tree matching (`<UNK>` zero-day isolation with zero hardcoded keyword rules).
- **Total memory footprint:** < 250 MB (ideal for edge or sidecar container deployment).

---

## 7. Ablation Analysis

To quantify the individual contribution of each component, an ablation study was conducted systematically:

| Variant # | Architecture Component | Thresholding Technique | F1-Score | FPR | Key Observation |
|:---:|---|---|:---:|:---:|---|
| **V1** | Unidirectional LSTM (DeepLog) | Static Top-9 | 0.8377 | 23.25% | Static cutoff over-flags concurrent telemetry branches. |
| **V2** | Positional Pad-Masked Bi-LSTM + 4-Head Attention | Static Top-9 | 0.7818 | 27.25% | Sharper logits let stealthy violations slip inside Top-9. |
| **V3** | DeepLog + Adaptive KDE | Gaussian KDE ($\alpha=0.02$) | 0.9639 | 4.50% | Continuous CDF thresholding cuts FPR by >80% vs Top-9. |
| **V4** | **CyberGuard AI (Bi-LSTM + Attention + Adaptive KDE)** | **Gaussian KDE ($\alpha=0.02$)** | **0.9736** | **3.25%** | **Best overall precision, F1, lowest FPR (3.25%), 100% recall + XAI.** |

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

CyberGuard AI includes a standalone, interactive Security Operations Center (SOC) workbench:

* **Dark-Mode Command Center:** Styled with curated CSS tokens, high-contrast indicators, and glassmorphic panels.
* **Live Gaussian KDE & NLL Spike Plots:** Pure inline SVG charts plotting the fitted normal validation density $\hat{f}(x)$, threshold $\tau$, live session marker, and per-event NLL spike bars.
* **Multi-Scenario Attack & File Ingestion:** One-click simulation across **Zero-Day Security Intrusion**, **Stealthy INFO Order Violation**, and **Long-Range Lifecycle Skip**, plus drag/upload `.log` file analysis with automatic `blk_*` Block-ID session partitioning and live Loghub HDFS stream replay.
* **Human-Readable XAI Triage Panel:** Auto-generated root-cause diagnostic summary paired with pad-masked multi-head attention tiles showing exact line numbers and Drain3 templates.

---

## 10. REST API Specification

The FastAPI microservice serves interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### Summary Table

| Endpoint | Method | Request Payload | Response | Description |
|---|:---:|---|---|---|
| `/api/health` | `GET` | None | `{"status": "healthy", "kde_threshold": 3.5831, ...}` | System readiness, device, OpenSearch status, and active KDE threshold. |
| `/api/stats` | `GET` | None | `{"total_analyzed": 142, "anomalies_detected": 12, ...}` | Aggregated SOC telemetry and average latency (plus `/api/stats/reset`). |
| `/api/kde-curve` | `GET` | None | `{"x": [...], "pdf": [...], "threshold": 3.5831}` | Sampled Gaussian KDE PDF coordinates for live plotting. |
| `/api/benchmark` | `GET` | None | `{"available": true, "rows": [...]}` | Live empirical benchmark table served from `benchmark_results.csv`. |
| `/api/sample-hdfs` | `GET` | Query `?limit=60` | `{"source": "Loghub HDFS_2k.log", "logs": [...]}` | Streams authentic lines from `HDFS_sample.log` for replay demos. |
| `/api/simulate-attack` | `GET` | Query `?scenario=...` | `{"scenario": "...", "attack_type": "...", "logs": [...]}` | Multi-scenario attack vectors (`zero_day`, `order_violation`, `long_range`). |
| `/api/analyze` | `POST` | `{"raw_logs": [...], "session_id": "..."}` | `{"anomaly_detected": true, "triage_summary": "...", ...}` | Parses via frozen Drain3, computes batched NLL vs KDE $\tau$ and XAI context. |
| `/api/analyze-batch` | `POST` | `{"raw_logs": [...], "auto_group": true}` | `{"total_sessions": N, "sessions": [...]}` | Partitions mixed logs by `blk_*` Block ID and ranks anomalous sessions first. |
| `/api/analyze-file` | `POST` | Multipart `.log`/`.txt` | `{"total_sessions": N, "sessions": [...]}` | Analyzes uploaded log files with automatic Block-ID sessionization. |

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
    ├── models/checkpoints/                 # Serialized model weights, Drain3 & KDE estimators
    │   ├── attention_bilstm_best.pt        # Trained Bi-LSTM + Attention weights
    │   ├── attention_bilstm_kde.pkl        # Fitted Gaussian KDE estimator
    │   ├── deeplog_best.pt                 # Trained DeepLog baseline weights
    │   ├── deeplog_kde.pkl                 # Fitted baseline KDE estimator
    │   └── drain3_parser.pkl               # Frozen Drain3 tree & vocabulary state
    ├── src/
    │   ├── __init__.py
    │   ├── parser.py                       # Drain3 parser (online + frozen OOV matching)
    │   ├── session.py                      # Sliding window session generator
    │   ├── dataset.py                      # Causal PyTorch Dataset and DataLoader
    │   ├── threshold.py                    # Adaptive Gaussian KDE threshold calibrator
    │   ├── train.py                        # Model training engine
    │   ├── evaluate.py                     # Batched benchmark evaluation suite
    │   ├── integrations.py                 # Optional OpenSearch alert sink
    │   └── models/
    │       ├── __init__.py
    │       ├── attention_bilstm.py         # Positional Pad-Masked Bi-LSTM + Attention
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
collected 12 items

tests/test_api_endpoints.py::test_health_endpoint PASSED                 [  8%]
tests/test_api_endpoints.py::test_stats_and_reset_endpoints PASSED       [ 16%]
tests/test_api_endpoints.py::test_kde_curve_and_benchmark_endpoints PASSED [ 25%]
tests/test_api_endpoints.py::test_simulate_attack_scenarios PASSED       [ 33%]
tests/test_api_endpoints.py::test_analyze_normal_and_attack_sessions PASSED [ 41%]
tests/test_api_endpoints.py::test_analyze_batch_multi_block_partitioning PASSED [ 50%]
tests/test_pipeline.py::test_drain3_parser_and_frozen_oov_match PASSED   [ 58%]
tests/test_pipeline.py::test_group_logs_by_session PASSED                [ 66%]
tests/test_pipeline.py::test_sliding_window_causal_alignment PASSED      [ 75%]
tests/test_pipeline.py::test_model_forward_pass_and_pad_masking PASSED   [ 83%]
tests/test_pipeline.py::test_adaptive_kde_threshold PASSED               [ 91%]
tests/test_pipeline.py::test_session_nll_computation PASSED              [100%]

============================== 12 passed in 5.25s ==============================
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
