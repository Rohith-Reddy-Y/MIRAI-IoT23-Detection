# MIRAI — IoT-23 Malware Detection System

🛡️ **Two-Stage IoT Malware Detection with Attack Variant Identification**

A machine learning-powered network traffic analysis system that detects IoT malware using an ensemble of XGBoost models trained on the [IoT-23 dataset](https://www.stratosphereips.org/datasets-iot23).

![Dashboard](https://img.shields.io/badge/Dashboard-Live-00d4aa?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-ff6f00?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)

## 🚀 Overview

MIRAI uses a **two-stage detection pipeline**:

1. **Stage 1 — Binary Detection**: An ensemble of 8 selected XGBoost models (ROC-AUC > 0.85) votes on whether each network packet is **benign** or **malicious**
2. **Stage 2 — Variant Identification**: 20 multiclass XGBoost models classify malicious packets into specific **attack variants** (C&C, DDoS, Port Scan, Okiru, Torii, etc.)

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Binary Models Trained | 20 / 23 datasets |
| Binary Models Selected | 8 (ROC-AUC > 0.85) |
| Multiclass Models | 20 (all variants covered) |
| Best Binary ROC-AUC | 1.0000 (Dataset 5, 17, 20) |
| Best Multiclass Accuracy | 100.00% (Dataset 5, 13) |
| Attack Variants Detected | 14 unique types |
| Total Training Rows | ~266M rows |

## 🧬 Detected Attack Variants

| Variant | Description |
|---------|-------------|
| 📡 C&C Communication | Botnet command & control traffic |
| 💓 C&C Heartbeat | Periodic keepalive signals |
| 💔 C&C Heartbeat Attack | Heartbeat combined with attack |
| 📥 C&C File Download | Malicious payload download |
| 🤖 Mirai C&C | Mirai botnet-specific C&C |
| 🌀 Torii C&C | Torii botnet C&C (advanced IoT malware) |
| 💥 DDoS Attack | Distributed Denial of Service |
| 🔎 Horizontal Port Scan | Network reconnaissance |
| ⚡ Port Scan Attack | Aggressive port scanning |
| 🐛 Okiru Malware | Okiru/Satori botnet variant |
| ⚔️ Attack | Generic malicious traffic |
| 📥 File Download | Suspicious file download |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  User Upload                     │
│              (CSV / Parquet file)                 │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│         Feature Engineering (19 features)        │
│   duration, orig_bytes, resp_bytes, proto_*,     │
│   service_*, local_orig, local_resp, ...         │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│     Stage 1: Binary Ensemble (8 models)          │
│        Confidence-weighted voting                │
│     Threshold: 0.75 → Benign / Malicious        │
└──────────────────┬──────────────────────────────┘
                   ▼ (malicious only)
┌─────────────────────────────────────────────────┐
│    Stage 2: Multiclass Models (20 models)        │
│     Per-dataset variant classification           │
│     → Attack variant + confidence score          │
└─────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
MIRAI-IoT23-Detection/
├── server.py               # Flask backend — model loading & prediction
├── dashboard.html           # Frontend — main dashboard page
├── dashboard.js             # Frontend — rendering & live detection logic
├── dashboard.css            # Frontend — premium dark theme styling
├── dashboard_data.js        # Frontend — static dataset metrics & stats
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── cap PPt/
│   ├── IOT23_Models/        # Binary XGBoost models (per dataset)
│   ├── IOT23_Models_Multiclass/  # Multiclass XGBoost models
│   ├── IOT23_Evaluations/   # Binary evaluation metrics
│   ├── IOT23_Evaluations_Multiclass/  # Multiclass evaluation metrics
│   ├── Feature_Selection_for_addition_of_multiclass.py
│   ├── Final_model_training_including_y_multiclass.py
│   └── model_testing_phase_including_multiclass.py
```

## 🛠️ Setup & Run

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MIRAI-IoT23-Detection.git
cd MIRAI-IoT23-Detection

# Install dependencies
pip install -r requirements.txt

# Run the server
python server.py
```

### Access
Open your browser and navigate to: **http://localhost:8080**

## 📐 Feature Engineering

19 features extracted from IoT-23 Zeek connection logs. Each model dynamically selects a subset (13–19) based on feature importance:

| Feature | Type | Description |
|---------|------|-------------|
| `duration` | Numeric | Connection duration |
| `orig_bytes` | Numeric | Bytes sent by originator |
| `resp_bytes` | Numeric | Bytes sent by responder |
| `orig_pkts` | Numeric | Packets from originator |
| `resp_pkts` | Numeric | Packets from responder |
| `orig_ip_bytes` | Numeric | IP-level bytes (originator) |
| `resp_ip_bytes` | Numeric | IP-level bytes (responder) |
| `missed_bytes` | Numeric | Missed bytes in connection |
| `local_orig` | Boolean | Originator is local |
| `local_resp` | Boolean | Responder is local |
| `proto_tcp` | One-hot | TCP protocol |
| `proto_udp` | One-hot | UDP protocol |
| `proto_icmp` | One-hot | ICMP protocol |
| `service_-` | One-hot | No identified service |
| `service_dns` | One-hot | DNS service |
| `service_http` | One-hot | HTTP service |
| `service_ssl` | One-hot | SSL/TLS service |
| `service_ssh` | One-hot | SSH service |
| `service_irc` | One-hot | IRC service |

## 🎯 Model Selection

**Binary (Stage 1)**: Only 8 models with ROC-AUC > 0.85 are used. Models trained on extremely imbalanced datasets (99.99% malicious) produce random predictions and are excluded.

**Multiclass (Stage 2)**: All 20 models are included to ensure complete attack variant coverage. Confidence-weighted voting ensures high-accuracy models dominate.

## 📜 License

This project is part of a Capstone Project (2025-26). Built with the [IoT-23 Dataset](https://www.stratosphereips.org/datasets-iot23) by Stratosphere Laboratory.

## 🙏 Acknowledgments

- **IoT-23 Dataset** — Stratosphere Laboratory, Czech Technical University
- **XGBoost** — Distributed Gradient Boosting Library
- **Chart.js** — Interactive chart rendering
- **Flask** — Python web framework
