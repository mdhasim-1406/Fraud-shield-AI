# 🛡️ Fraud Detection System

**Enterprise-grade AI-powered fraud detection system** using DeepSeek V3.1, FinBERT, and RAG for real-time transaction analysis.

## 🚀 Project Overview

This is a complete, production-ready fraud detection system that analyzes financial transactions using advanced AI models:

- **DeepSeek V3.1** - Advanced reasoning engine for fraud pattern analysis
- **FinBERT** - Financial sentiment analysis and behavioral anomaly detection
- **FAISS RAG** - Historical fraud pattern retrieval and context augmentation
- **FastAPI** - High-performance REST API backend
- **Gradio** - Professional web dashboard for analysts

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fraud Detection System                       │
├─────────────────────────────────────────────────────────────────┤
│  📊 Data Layer     │  🤖 AI Models      │  🔍 Analysis Pipeline │
├─────────────────────────────────────────────────────────────────┤
│  • Data Loader     │  • FinBERT         │  • Transaction Ingestion│
│  • CSV Processing  │  • DeepSeek V3.1   │  • Sentiment Analysis   │
│  • Batch Processing│  • RAG System      │  • Fraud Reasoning     │
│  • Validation      │  • Embeddings      │  • Pattern Matching    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API & Interface Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  • FastAPI Backend │  • Gradio Dashboard │  • Real-time Updates  │
│  • REST Endpoints  │  • Visual Analytics │  • Analyst Tools      │
│  • Async Processing│  • Interactive UI   │  • Export Functions   │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Components

### 1. **Data Processing Module** (`data_loader.py`)
- Loads and validates transaction data from CSV
- Handles missing values and data type validation
- Generates Chain-of-Thought prompts for AI analysis
- Supports batch processing for efficiency

### 2. **FinBERT Sentiment Analysis** (`finbert_analyzer.py`)
- Financial sentiment analysis using ProsusAI/finbert
- Behavioral anomaly detection algorithms
- Risk factor identification and scoring
- Real-time transaction context building

### 3. **DeepSeek V3.1 Engine** (`deepseek_detector.py`)
- Advanced fraud reasoning with DeepSeek V3.1
- Multi-shot prompt engineering with few-shot examples
- Self-consistency analysis for high-value transactions
- Structured JSON output parsing

### 4. **RAG Knowledge Base** (`rag_system.py`)
- FAISS vector database for fraud pattern storage
- Historical fraud case retrieval and similarity search
- Prompt augmentation with relevant fraud context
- Knowledge base statistics and analytics

### 5. **FastAPI Backend** (`main.py`)
- Production-grade REST API with async support
- Complete fraud detection pipeline orchestration
- Batch processing capabilities
- Real-time health monitoring and statistics

### 6. **Gradio Dashboard** (`dashboard.py`)
- Professional three-panel analyst interface
- Real-time transaction stream monitoring
- Interactive investigation workspace
- Demo mode and export functionality

### 7. **Testing Suite** (`test_system.py`)
- Comprehensive unit and integration tests
- Performance benchmarking tools
- Accuracy validation against datasets
- Error handling and edge case testing

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- 8GB+ RAM recommended
- GPU optional (for faster FinBERT processing)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Fraud-shield-AI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Edit `.env` file:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get your API key from [OpenRouter](https://openrouter.ai/)

### 4. Verify Installation
```bash
# Test data loading
python data_loader.py

# Test FinBERT analyzer
python finbert_analyzer.py

# Test RAG system
python rag_system.py
```

## 🚀 Quick Start

### Start the Complete System

1. **Start FastAPI Backend:**
```bash
python main.py
```
API will be available at `http://localhost:8000`

2. **Start Gradio Dashboard:**
```bash
python dashboard.py
```
Dashboard will be available at `http://localhost:7860`

3. **Run Tests:**
```bash
python test_system.py
```

### Basic API Usage

#### Single Transaction Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_id": "TXN_123",
       "user_id": "USER_456",
       "transaction_amount": 1500.0,
       "transaction_type": "Online",
       "merchant_category": "Electronics",
       "account_balance": 2000.0,
       "device_type": "Mobile",
       "location": "New York",
       "authentication_method": "Biometric",
       "daily_transaction_count": 3,
       "avg_transaction_amount_7d": 250.0,
       "failed_transaction_count_7d": 1,
       "card_age": 365,
       "transaction_distance": 0.0,
       "ip_address_flag": 0,
       "previous_fraudulent_activity": 0,
       "risk_score": 0.3,
       "is_weekend": 0
     }'
```

#### Batch Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "transactions": [/* array of transactions */],
       "max_concurrent": 10
     }'
```

## 📊 Performance Benchmarks

The system is designed to meet enterprise performance requirements:

| Component | Target | Typical Performance |
|-----------|--------|-------------------|
| **Data Loading** | < 5 seconds | ~1-2 seconds |
| **FinBERT Analysis** | < 10 seconds | ~3-5 seconds |
| **RAG Search** | < 5 seconds | ~0.5-1 seconds |
| **DeepSeek Fast** | < 2 seconds | ~1-1.5 seconds |
| **DeepSeek Deep** | < 30 seconds | ~5-15 seconds |
| **End-to-End** | < 2 seconds | ~1-2 seconds |

### Performance Testing
```bash
# Run comprehensive benchmarks
python test_system.py

# Individual component benchmarks
python -c "
from test_system import *
benchmark_data_loading()
benchmark_finbert_analysis()
benchmark_rag_system()
"
```

## 🔧 Configuration

### Environment Variables (`.env`)
```bash
# DeepSeek API Configuration
OPENROUTER_API_KEY=your_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Model Selection
FAST_MODEL=deepseek/deepseek-chat
REASONING_MODEL=deepseek/deepseek-r5

# Performance Settings
API_TIMEOUT=30
MAX_RETRIES=3
BATCH_SIZE=10

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

### Configuration Management (`config.py`)
```python
from config import config

# Access settings
api_key = config.settings.OPENROUTER_API_KEY
timeout = config.settings.API_TIMEOUT
batch_size = config.settings.BATCH_SIZE

# Validate configuration
config.validate_config()
```

## 📈 Key Features

### 🎯 **Advanced AI Analysis**
- **Chain-of-Thought Prompting** - Structured reasoning for fraud detection
- **Few-Shot Learning** - Examples for improved accuracy
- **Self-Consistency** - Multiple analyses for high-value transactions
- **Task-Specific Outputs** - Structured JSON responses

### 🔍 **Comprehensive Fraud Detection**
- **Behavioral Analysis** - User pattern and anomaly detection
- **Sentiment Analysis** - Financial context interpretation
- **Historical Matching** - Similar fraud pattern identification
- **Risk Scoring** - Multi-factor risk assessment

### ⚡ **Enterprise Performance**
- **Async Processing** - Concurrent transaction analysis
- **Batch Operations** - Efficient bulk processing
- **Connection Pooling** - Optimized API usage
- **Caching Strategies** - Improved response times

### 🛡️ **Production Ready**
- **Error Handling** - Comprehensive exception management
- **Logging** - Structured logging with configurable levels
- **Health Monitoring** - System status and metrics
- **Input Validation** - Pydantic model validation

## 🧪 Testing

### Run All Tests
```bash
# Run with pytest
pytest test_system.py -v

# Run specific test classes
pytest test_system.py::TestDataLoader -v
pytest test_system.py::TestFinBERTAnalyzer -v
pytest test_system.py::TestDeepSeekDetector -v
pytest test_system.py::TestRAGSystem -v
```

### Test Coverage Areas
- ✅ **Unit Tests** - Individual component functionality
- ✅ **Integration Tests** - End-to-end pipeline validation
- ✅ **Performance Tests** - Benchmarking and optimization
- ✅ **Error Handling** - Edge cases and failure scenarios
- ✅ **Mock Tests** - API testing without external dependencies

### Manual Testing Checklist

#### API Endpoints
- [ ] `GET /` - Root endpoint
- [ ] `GET /health` - Health check
- [ ] `POST /api/v1/analyze` - Single transaction analysis
- [ ] `POST /api/v1/batch` - Batch processing
- [ ] `GET /api/v1/stats` - System statistics
- [ ] `GET /api/v1/transaction/{id}` - Transaction lookup
- [ ] `POST /api/v1/feedback` - Analyst feedback

#### Dashboard Features
- [ ] Transaction stream display
- [ ] Real-time analysis results
- [ ] Demo mode functionality
- [ ] Export report generation
- [ ] Auto-refresh capabilities

## 📊 API Documentation

### Core Endpoints

#### Analyze Single Transaction
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "transaction_id": "string",
  "user_id": "string",
  "transaction_amount": 1500.0,
  "transaction_type": "Online",
  "merchant_category": "Electronics",
  "account_balance": 2000.0,
  "device_type": "Mobile",
  "location": "New York",
  "authentication_method": "Biometric",
  "daily_transaction_count": 3,
  "avg_transaction_amount_7d": 250.0,
  "failed_transaction_count_7d": 1,
  "card_age": 365,
  "transaction_distance": 0.0,
  "ip_address_flag": 0,
  "previous_fraudulent_activity": 0,
  "risk_score": 0.3,
  "is_weekend": 0
}
```

**Response:**
```json
{
  "transaction_id": "TXN_123",
  "fraud_probability": 0.75,
  "risk_level": "HIGH",
  "finbert_sentiment": {
    "sentiment_positive": 0.1,
    "sentiment_negative": 0.7,
    "sentiment_neutral": 0.2,
    "confidence": 0.8,
    "sentiment_label": "negative",
    "anomaly_flags": ["high_amount_balance_ratio"],
    "anomaly_score": 0.6,
    "risk_factors": {...},
    "context_text": "...",
    "overall_risk_score": 0.65
  },
  "deepseek_reasoning": {
    "fraud_probability": 0.8,
    "risk_level": "HIGH",
    "reasoning_steps": [
      "Step 1: High amount-to-balance ratio indicates potential fraud",
      "Step 2: Transaction pattern deviates from user behavior",
      "Step 3: Authentication method shows anomaly",
      "Step 4: Merchant category risk assessment",
      "Step 5: Final fraud probability determination"
    ],
    "red_flags": ["High amount", "Pattern deviation"],
    "confidence": 0.85,
    "recommendation": "BLOCK",
    "analysis_mode": "fast"
  },
  "similar_frauds": [
    {
      "transaction_id": "TXN_SIM_001",
      "similarity_score": 0.85,
      "fraud_indicators": ["high_amount_balance_ratio"],
      "amount": 1200.0,
      "merchant_category": "Electronics"
    }
  ],
  "final_verdict": "BLOCK",
  "processing_time": 1.23,
  "analysis_timestamp": "2024-01-15 10:30:45"
}
```

#### Batch Analysis
```http
POST /api/v1/batch
Content-Type: application/json

{
  "transactions": [
    {
      "transaction_id": "TXN_001",
      "transaction_amount": 1500.0,
      ...
    },
    {
      "transaction_id": "TXN_002",
      "transaction_amount": 75.0,
      ...
    }
  ],
  "max_concurrent": 10
}
```

## 🎛️ Dashboard Usage

### Analyst Workflow

1. **Monitor Transaction Stream**
   - View real-time transactions with color-coded risk levels
   - Click on high-risk transactions for detailed analysis

2. **Investigate Suspicious Transactions**
   - Load transaction details in investigation workspace
   - Review AI analysis results and reasoning
   - Check similar historical fraud cases

3. **Make Decisions**
   - Use Approve/Review/Block buttons for quick actions
   - Add analyst notes for documentation
   - Export detailed reports for compliance

4. **System Monitoring**
   - Monitor system performance and model status
   - Track fraud detection rates and processing times
   - View knowledge base statistics

### Demo Mode
```bash
# Run demo with sample transactions
python dashboard.py
# Navigate to http://localhost:7860
# Click "Demo Mode" to see system capabilities
```

## 🔧 Advanced Configuration

### Customizing AI Models

#### FinBERT Configuration
```python
from finbert_analyzer import FinBERTAnalyzer

# Custom model configuration
analyzer = FinBERTAnalyzer(
    model_name="ProsusAI/finbert",
    device="cuda"  # or "cpu"
)
```

#### DeepSeek Configuration
```python
from deepseek_detector import DeepSeekFraudDetector

detector = DeepSeekFraudDetector(
    api_key="your_key",
    fast_model="deepseek/deepseek-chat",
    reasoning_model="deepseek/deepseek-r5",
    timeout=30,
    max_retries=3
)
```

#### RAG Configuration
```python
from rag_system import FraudKnowledgeBase

kb = FraudKnowledgeBase(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    index_path="models/custom_index.faiss",
    embedding_dim=384
)
```

## 📈 Monitoring & Analytics

### System Metrics
- **Transaction Processing Rate** - Transactions per second
- **Fraud Detection Accuracy** - Precision and recall metrics
- **Average Response Time** - API endpoint performance
- **Model Confidence Scores** - AI model certainty levels
- **Knowledge Base Growth** - Historical fraud patterns stored

### Health Checks
```bash
# API Health
curl http://localhost:8000/health

# Dashboard Status
curl http://localhost:8000/api/v1/stats

# Knowledge Base Stats
curl http://localhost:8000/api/v1/knowledge-base/stats
```

## 🚨 Troubleshooting

### Common Issues

#### API Connection Errors
```bash
# Check if server is running
curl http://localhost:8000/health

# Verify API key in .env
cat .env

# Check server logs
tail -f logs/fraud_detection.log
```

#### Model Loading Issues
```bash
# Test individual components
python data_loader.py
python finbert_analyzer.py
python rag_system.py

# Check memory usage
python -c "import torch; print(torch.cuda.is_available())"
```

#### Performance Issues
```bash
# Run performance benchmarks
python test_system.py

# Check system resources
htop  # or top

# Monitor GPU usage (if applicable)
nvidia-smi
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest test_system.py -v --cov

# Format code
black .
isort .
```

### Code Structure
```
Fraud-shield-AI/
├── data_loader.py      # Data processing and validation
├── finbert_analyzer.py # Financial sentiment analysis
├── deepseek_detector.py# AI reasoning engine
├── rag_system.py       # Historical pattern matching
├── main.py             # FastAPI backend
├── dashboard.py        # Gradio web interface
├── test_system.py      # Testing suite
├── config.py           # Configuration management
├── requirements.txt    # Dependencies
├── .env               # Environment variables
└── README.md          # This file
```

## 📄 License

This project is designed for enterprise fraud detection applications. Please ensure compliance with relevant financial regulations and data protection laws.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review test outputs for error details
3. Check API response codes and error messages
4. Verify all dependencies are installed correctly

## 🎯 Success Metrics

### Performance Targets ✅
- **Transaction Processing**: < 2 seconds per transaction
- **Batch Processing**: Support up to 50 concurrent transactions
- **API Response Time**: Sub-second for health checks
- **Dashboard Load Time**: < 5 seconds
- **System Availability**: 99.9% uptime target

### Accuracy Targets 📊
- **Fraud Detection Rate**: > 90% recall for known fraud patterns
- **False Positive Rate**: < 5% for legitimate transactions
- **Model Confidence**: > 80% average confidence scores
- **Pattern Matching**: > 85% similarity for relevant cases

---

**Built with ❤️ for enterprise fraud detection**
*DeepSeek V3.1 • FinBERT • FAISS • FastAPI • Gradio*
