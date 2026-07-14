# Fraud Shield AI — Local Fraud Detection System

Enterprise-grade AI-powered fraud detection system that runs **fully offline** using local LLMs via Ollama. No data leaves your machine, no API keys required.

## Overview

This system analyzes financial transactions through a multi-model pipeline:

- **FinBERT** — Financial sentiment analysis and behavioral anomaly detection (local HuggingFace model)
- **Local LLM (Ollama)** — Fraud reasoning engine using gemma2:2b (or any Ollama-hosted model)
- **FAISS RAG** — Historical fraud pattern retrieval and similarity search
- **FastAPI** — REST API backend
- **Gradio** — SOC-style monitoring dashboard

## Architecture

```
Transaction Data (CSV)
       │
FinBERT Analyzer (local)    Ollama LLM (local, gemma2:2b)
       │                            │
       ▼                            ▼
   Sentiment Scores ──► Orchestrator ◄── RAG (FAISS)
                            │
                            ▼
                     Verdict (APPROVE / REVIEW / BLOCK)
                            │
                            ▼
                   FastAPI + Gradio Dashboard
```

## Components

| Module | File | Role |
|---|---|---|
| Data Processing | `data_loader.py` | CSV ingestion, validation, batch processing |
| FinBERT Analysis | `finbert_analyzer.py` | Financial sentiment + anomaly detection (local) |
| LLM Reasoning | `deepseek_detector.py` | Fraud reasoning via Ollama (gemma2:2b) |
| RAG System | `rag_system.py` | FAISS vector DB for historical fraud patterns |
| FastAPI Backend | `main.py` | REST API orchestrating the full pipeline |
| Gradio Dashboard | `dashboard.py` | SOC-style monitoring cockpit |
| Simulation | `simulation_manager.py` | Real-time transaction stream generator |

## Setup

### 1. Install Ollama and pull a model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the default model (gemma2:2b)
ollama pull gemma2:2b

# Start Ollama server
ollama serve
```

Keep the Ollama server running in the background — the fraud detector connects to it at `http://localhost:11434/v1`.

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/mdhasim-1406/Fraud-shield-AI.git
cd Fraud-shield-AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the system

```bash
# Terminal 1: FastAPI backend
python main.py

# Terminal 2: Gradio dashboard
python dashboard.py
```

API available at `http://localhost:8000`, dashboard at `http://localhost:7860`.

### 4. Quick test

```bash
# Single transaction analysis
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_001",
    "user_id": "USER_001",
    "transaction_amount": 1500.0,
    "transaction_type": "Online",
    "timestamp": "2025-01-15T10:30:00",
    "account_balance": 2000.0,
    "device_type": "Mobile",
    "location": "New York",
    "merchant_category": "Electronics",
    "ip_address_flag": 0,
    "previous_fraudulent_activity": 0,
    "daily_transaction_count": 3,
    "avg_transaction_amount_7d": 250.0,
    "failed_transaction_count_7d": 1,
    "card_type": "Visa",
    "card_age": 365,
    "transaction_distance": 0.0,
    "authentication_method": "Biometric",
    "risk_score": 0.3,
    "is_weekend": 0
  }'
```

## Running Tests

```bash
python test_system.py
```

## Configuration

Edit `config.py` or set environment variables:

```bash
# Change model (default: gemma2:2b)
export LLM_MODEL="qwen2.5:3b"

# Change Ollama endpoint
export OLLAMA_BASE_URL="http://localhost:11434/v1"
```

No `.env` file or API key needed — everything runs locally.
