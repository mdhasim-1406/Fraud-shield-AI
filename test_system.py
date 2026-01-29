"""
Testing Suite for Fraud Detection System
Component 7: Comprehensive testing and validation
"""
import pytest
import pytest_asyncio
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

# Import our modules for testing
from data_loader import TransactionDataLoader, load_transaction_data
from finbert_analyzer import FinBERTAnalyzer, analyze_financial_sentiment
from deepseek_detector import DeepSeekFraudDetector, analyze_fraud_with_deepseek
from rag_system import FraudKnowledgeBase, retrieve_similar_frauds
from config import config

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataLoader:
    """Test cases for data loading functionality"""

    def test_load_transaction_data_success(self):
        """Test successful data loading"""
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        assert df is not None
        assert len(df) > 0
        assert 'Transaction_ID' in df.columns
        assert 'Fraud_Label' in df.columns
        assert len(df.columns) == 21  # Expected number of columns

    def test_load_transaction_data_missing_file(self):
        """Test error handling for missing file"""
        loader = TransactionDataLoader("nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            loader.load_transaction_data()

    def test_row_to_fraud_analysis_prompt(self):
        """Test prompt generation from transaction row"""
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        if len(df) > 0:
            row = df.iloc[0]
            prompt_data = loader.row_to_fraud_analysis_prompt(row)

            assert 'transaction_id' in prompt_data
            assert 'analysis_prompt' in prompt_data
            assert 'amount' in prompt_data
            assert 'is_high_value' in prompt_data
            assert len(prompt_data['analysis_prompt']) > 0

    def test_prepare_batch_transactions(self):
        """Test batch processing functionality"""
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        batches = list(loader.prepare_batch_transactions(batch_size=5))

        assert len(batches) > 0
        assert all(isinstance(batch, list) for batch in batches)
        assert all(len(transaction) > 0 for batch in batches for transaction in [batch])

    def test_get_transaction_by_id(self):
        """Test transaction retrieval by ID"""
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        if len(df) > 0:
            first_id = df.iloc[0]['Transaction_ID']
            transaction = loader.get_transaction_by_id(first_id)

            assert transaction is not None
            assert transaction['transaction_id'] == first_id

    def test_get_dataset_stats(self):
        """Test dataset statistics generation"""
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        stats = loader.get_dataset_stats()

        assert 'total_transactions' in stats
        assert 'fraud_transactions' in stats
        assert 'fraud_rate' in stats
        assert stats['total_transactions'] > 0
        assert 0 <= stats['fraud_rate'] <= 1

class TestFinBERTAnalyzer:
    """Test cases for FinBERT sentiment analysis"""

    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction for testing"""
        return {
            'Transaction_ID': 'TXN_TEST',
            'User_ID': 'USER_TEST',
            'Transaction_Amount': 1500.0,
            'Transaction_Type': 'Online',
            'Account_Balance': 1000.0,
            'Device_Type': 'Mobile',
            'Merchant_Category': 'Electronics',
            'Daily_Transaction_Count': 5,
            'Avg_Transaction_Amount_7d': 200.0,
            'Failed_Transaction_Count_7d': 2,
            'Authentication_Method': 'Biometric',
            'IP_Address_Flag': 0,
            'Card_Age': 15,
            'Transaction_Distance': 500.0,
            'Is_Weekend': 0
        }

    def test_analyzer_initialization(self):
        """Test FinBERT analyzer initialization"""
        analyzer = FinBERTAnalyzer()

        assert analyzer.model is not None
        assert analyzer.tokenizer is not None
        assert analyzer.sentiment_pipeline is not None

    def test_build_financial_context(self, sample_transaction):
        """Test financial context building"""
        analyzer = FinBERTAnalyzer()

        context = analyzer.build_financial_context(sample_transaction)

        assert len(context) > 0
        assert 'USER_TEST' in context
        assert '1500.0' in context
        assert 'Electronics' in context

    def test_sentiment_analysis(self, sample_transaction):
        """Test sentiment analysis functionality"""
        analyzer = FinBERTAnalyzer()

        context = analyzer.build_financial_context(sample_transaction)
        sentiment_result = analyzer.get_finbert_sentiment(context)

        assert sentiment_result.positive >= 0
        assert sentiment_result.negative >= 0
        assert sentiment_result.neutral >= 0
        assert sentiment_result.confidence >= 0
        assert sentiment_result.label in ['positive', 'negative', 'neutral']

        # Probabilities should sum to approximately 1
        total_prob = sentiment_result.positive + sentiment_result.negative + sentiment_result.neutral
        assert 0.99 <= total_prob <= 1.01

    def test_anomaly_detection(self, sample_transaction):
        """Test behavioral anomaly detection"""
        analyzer = FinBERTAnalyzer()

        context = analyzer.build_financial_context(sample_transaction)
        sentiment_result = analyzer.get_finbert_sentiment(context)
        anomaly_result = analyzer.detect_behavioral_anomalies(sample_transaction, sentiment_result)

        assert isinstance(anomaly_result.anomaly_flags, list)
        assert isinstance(anomaly_result.anomaly_score, float)
        assert 0 <= anomaly_result.anomaly_score <= 1
        assert isinstance(anomaly_result.risk_factors, dict)

    def test_complete_analysis(self, sample_transaction):
        """Test complete transaction analysis"""
        analyzer = FinBERTAnalyzer()

        result = analyzer.analyze_transaction_sentiment(sample_transaction)

        expected_keys = [
            'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'confidence', 'sentiment_label', 'anomaly_flags', 'anomaly_score',
            'risk_factors', 'context_text', 'overall_risk_score'
        ]

        for key in expected_keys:
            assert key in result

        assert 0 <= result['overall_risk_score'] <= 1

class TestDeepSeekDetector:
    """Test cases for DeepSeek fraud detection"""

    @pytest.fixture
    def sample_prompt(self):
        """Sample analysis prompt for testing"""
        return """
        Transaction ID: TXN_TEST
        Amount: $1500.00
        Type: Online
        Account Balance: $1000.00
        Daily Transaction Count: 5
        7-Day Average Amount: $200.00
        Failed Transactions (7d): 2
        Device Type: Mobile
        Merchant Category: Electronics
        """

    def test_detector_initialization(self):
        """Test DeepSeek detector initialization"""
        # Skip if no API key
        if not config.settings.OPENROUTER_API_KEY:
            pytest.skip("No API key available")

        detector = DeepSeekFraudDetector(api_key=config.settings.OPENROUTER_API_KEY)

        assert detector.api_key == config.settings.OPENROUTER_API_KEY
        assert detector.fast_client is not None
        assert detector.reasoning_client is not None

    def test_system_prompt_creation(self):
        """Test system prompt generation"""
        if not config.settings.OPENROUTER_API_KEY:
            pytest.skip("No API key available")

        detector = DeepSeekFraudDetector(api_key=config.settings.OPENROUTER_API_KEY)

        system_prompt = detector._create_system_prompt()

        assert len(system_prompt) > 0
        assert 'fraud detection analyst' in system_prompt.lower()
        assert 'JSON' in system_prompt

    def test_prompt_creation(self, sample_prompt):
        """Test analysis prompt creation"""
        if not config.settings.OPENROUTER_API_KEY:
            pytest.skip("No API key available")

        detector = DeepSeekFraudDetector(api_key=config.settings.OPENROUTER_API_KEY)

        fast_prompt = detector._create_fast_analysis_prompt(sample_prompt)
        deep_prompt = detector._create_deep_analysis_prompt(sample_prompt)

        assert len(fast_prompt) > len(sample_prompt)
        assert len(deep_prompt) > len(sample_prompt)
        assert sample_prompt in fast_prompt
        assert sample_prompt in deep_prompt

    @patch('deepseek_detector.OpenAI')
    def test_fast_analysis_mock(self, mock_openai, sample_prompt):
        """Test fast analysis with mocked API"""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "fraud_probability": 0.7,
            "risk_level": "HIGH",
            "reasoning_steps": ["Step 1", "Step 2"],
            "red_flags": ["High amount"],
            "confidence": 0.8,
            "recommendation": "REVIEW"
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        detector = DeepSeekFraudDetector(api_key="test_key")
        result = detector.analyze_transaction_fast(sample_prompt)

        assert result.fraud_probability == 0.7
        assert result.risk_level == "HIGH"
        assert len(result.reasoning_steps) == 2
        assert "High amount" in result.red_flags

    def test_parse_llm_response_valid(self):
        """Test parsing of valid LLM response"""
        if not config.settings.OPENROUTER_API_KEY:
            pytest.skip("No API key available")

        detector = DeepSeekFraudDetector(api_key=config.settings.OPENROUTER_API_KEY)

        valid_response = json.dumps({
            "fraud_probability": 0.6,
            "risk_level": "MEDIUM",
            "reasoning_steps": ["Analysis step 1", "Analysis step 2"],
            "red_flags": ["Suspicious pattern"],
            "confidence": 0.75,
            "recommendation": "REVIEW"
        })

        result = detector._parse_llm_response(valid_response)

        assert result.fraud_probability == 0.6
        assert result.risk_level == "MEDIUM"
        assert len(result.reasoning_steps) == 2

    def test_parse_llm_response_invalid(self):
        """Test parsing of invalid LLM response"""
        if not config.settings.OPENROUTER_API_KEY:
            pytest.skip("No API key available")

        detector = DeepSeekFraudDetector(api_key=config.settings.OPENROUTER_API_KEY)

        invalid_response = "This is not JSON"
        result = detector._parse_llm_response(invalid_response)

        assert result.fraud_probability == 0.5  # Default value
        assert result.risk_level == "MEDIUM"
        assert "Error parsing" in result.reasoning_steps[0]

class TestRAGSystem:
    """Test cases for RAG knowledge base"""

    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization"""
        kb = FraudKnowledgeBase()

        assert kb.embedding_model is not None
        assert kb.index is not None
        assert kb.embedding_dim == 384
        assert len(kb.fraud_cases) == 0

    def test_fraud_case_creation(self):
        """Test fraud case extraction and creation"""
        kb = FraudKnowledgeBase()

        # Sample transaction row
        sample_row = pd.Series({
            'Transaction_ID': 'TXN_TEST',
            'Transaction_Amount': 1000.0,
            'Account_Balance': 500.0,
            'Daily_Transaction_Count': 3,
            'Avg_Transaction_Amount_7d': 200.0,
            'Failed_Transaction_Count_7d': 2,
            'Transaction_Distance': 1500.0,
            'IP_Address_Flag': 1,
            'Card_Age': 15,
            'Is_Weekend': 1,
            'Merchant_Category': 'Electronics',
            'Device_Type': 'Mobile',
            'Location': 'New York',
            'Card_Type': 'Visa',
            'Authentication_Method': 'Password',
            'Previous_Fraudulent_Activity': 0,
            'Risk_Score': 0.8
        })

        indicators = kb._extract_fraud_indicators(sample_row)
        risk_factors = kb._extract_risk_factors(sample_row)
        user_behavior = kb._extract_user_behavior(sample_row)

        assert isinstance(indicators, list)
        assert isinstance(risk_factors, dict)
        assert isinstance(user_behavior, dict)
        assert "high_amount_balance_ratio" in indicators  # 1000/500 = 2.0 > 0.8
        assert risk_factors['amount_balance_ratio'] == 2.0

    def test_case_text_creation(self):
        """Test searchable text creation for fraud cases"""
        kb = FraudKnowledgeBase()

        sample_row = pd.Series({
            'Transaction_ID': 'TXN_TEST',
            'Transaction_Amount': 1000.0,
            'Merchant_Category': 'Electronics',
            'Location': 'New York',
            'Device_Type': 'Mobile',
            'Card_Type': 'Visa',
            'Authentication_Method': 'Biometric',
            'Risk_Score': 0.8,
            'Daily_Transaction_Count': 3,
            'Failed_Transaction_Count_7d': 2,
            'Transaction_Distance': 1500.0,
            'Card_Age': 15,
            'Avg_Transaction_Amount_7d': 200.0,
            'Previous_Fraudulent_Activity': 0
        })

        case_text = kb._create_case_text(sample_row)

        assert len(case_text) > 0
        assert 'TXN_TEST' in case_text
        assert '1000.0' in case_text
        assert 'Electronics' in case_text

    def test_embedding_generation(self):
        """Test transaction embedding generation"""
        kb = FraudKnowledgeBase()

        test_text = "Test transaction for embedding"
        embedding = kb.get_transaction_embedding(test_text)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == kb.embedding_dim

    def test_knowledge_base_building(self):
        """Test building knowledge base from dataset"""
        # Load sample data
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        kb = FraudKnowledgeBase()
        fraud_count = kb.build_fraud_knowledge_base(df)

        # Should find some fraud cases
        assert fraud_count >= 0
        assert len(kb.fraud_cases) == fraud_count

        if fraud_count > 0:
            assert kb.index.ntotal == fraud_count

    def test_similarity_search(self):
        """Test similar fraud case retrieval"""
        # Load sample data and build knowledge base
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        kb = FraudKnowledgeBase()
        kb.build_fraud_knowledge_base(df)

        if len(kb.fraud_cases) > 0:
            # Get embedding for first transaction
            first_case = kb.fraud_cases[0]
            query_embedding = first_case.embedding

            # Search for similar cases
            similar_cases = kb.retrieve_similar_frauds(query_embedding, k=3)

            assert isinstance(similar_cases, list)
            assert len(similar_cases) <= 3

            if similar_cases:
                assert all(hasattr(case, 'similarity_score') for case in similar_cases)
                assert all(0 <= case.similarity_score <= 1 for case in similar_cases)

    def test_prompt_augmentation(self):
        """Test RAG prompt augmentation"""
        kb = FraudKnowledgeBase()

        # Create mock similar cases
        mock_case = Mock()
        mock_case.transaction_id = "TXN_MOCK"
        mock_case.amount = 1000.0
        mock_case.merchant_category = "Electronics"
        mock_case.fraud_indicators = ["high_amount"]
        mock_case.risk_factors = {"amount_balance_ratio": 2.0}
        mock_case.user_behavior = {"avg_7d_amount": 200.0, "previous_fraud": 0}

        similar_case = Mock()
        similar_case.case = mock_case
        similar_case.similarity_score = 0.8
        similar_case.matching_indicators = ["high_amount"]

        base_prompt = "Analyze this transaction for fraud"
        augmented_prompt = kb.augment_prompt_with_rag(base_prompt, [similar_case])

        assert len(augmented_prompt) > len(base_prompt)
        assert "TXN_MOCK" in augmented_prompt
        assert "historical fraud cases" in augmented_prompt

class TestIntegration:
    """Integration tests for complete system"""

    def test_end_to_end_pipeline(self):
        """Test complete fraud detection pipeline"""
        # Load sample data
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        if len(df) == 0:
            pytest.skip("No data available for testing")

        # Get a sample transaction
        sample_row = df.iloc[0]
        transaction_dict = sample_row.to_dict()

        # Test FinBERT analysis
        finbert_analyzer = FinBERTAnalyzer()
        finbert_result = finbert_analyzer.analyze_transaction_sentiment(transaction_dict)

        assert 'overall_risk_score' in finbert_result
        assert 0 <= finbert_result['overall_risk_score'] <= 1

        # Test RAG system
        kb = FraudKnowledgeBase()
        kb.build_fraud_knowledge_base(df)

        transaction_text = finbert_analyzer.build_financial_context(transaction_dict)
        embedding = kb.get_transaction_embedding(transaction_text)
        similar_cases = kb.retrieve_similar_frauds(embedding, k=3)

        assert isinstance(similar_cases, list)

        # Test prompt augmentation
        augmented_prompt = kb.augment_prompt_with_rag(transaction_text, similar_cases)
        assert len(augmented_prompt) >= len(transaction_text)

    def test_performance_benchmarks(self):
        """Test system performance benchmarks"""
        # Load sample data
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        if len(df) == 0:
            pytest.skip("No data available for testing")

        # Test data loading performance
        start_time = time.time()
        test_loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        test_df = test_loader.load_transaction_data()
        load_time = time.time() - start_time

        assert load_time < 5.0  # Should load within 5 seconds
        assert len(test_df) > 0

        # Test FinBERT performance
        finbert_analyzer = FinBERTAnalyzer()
        sample_transaction = df.iloc[0].to_dict()

        start_time = time.time()
        finbert_result = finbert_analyzer.analyze_transaction_sentiment(sample_transaction)
        finbert_time = time.time() - start_time

        assert finbert_time < 10.0  # Should complete within 10 seconds
        assert 'overall_risk_score' in finbert_result

        # Test RAG performance
        kb = FraudKnowledgeBase()
        kb.build_fraud_knowledge_base(df)

        transaction_text = finbert_analyzer.build_financial_context(sample_transaction)
        embedding = kb.get_transaction_embedding(transaction_text)

        start_time = time.time()
        similar_cases = kb.retrieve_similar_frauds(embedding, k=3)
        rag_time = time.time() - start_time

        assert rag_time < 5.0  # Should complete within 5 seconds

    def test_accuracy_validation(self):
        """Test accuracy against known fraud labels"""
        # Load sample data
        loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
        df = loader.load_transaction_data()

        if len(df) == 0:
            pytest.skip("No data available for testing")

        # Test a few transactions and compare against known labels
        finbert_analyzer = FinBERTAnalyzer()
        correct_predictions = 0
        total_predictions = min(10, len(df))  # Test first 10 transactions

        for i in range(total_predictions):
            transaction_dict = df.iloc[i].to_dict()
            actual_label = transaction_dict.get('Fraud_Label', 0)

            # Get FinBERT risk score
            finbert_result = finbert_analyzer.analyze_transaction_sentiment(transaction_dict)
            predicted_fraud = finbert_result.get('overall_risk_score', 0) >= 0.6

            if predicted_fraud == actual_label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions

        # We expect reasonable accuracy (this is just a basic test)
        assert accuracy >= 0.3  # At least 30% accuracy on small sample

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_values_handling(self):
        """Test handling of missing values in data"""
        # Create test data with missing values
        test_data = {
            'Transaction_ID': ['TXN_1', 'TXN_2'],
            'Transaction_Amount': [100.0, None],
            'Fraud_Label': [0, 1]
        }

        df = pd.DataFrame(test_data)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            loader = TransactionDataLoader(temp_file)
            result_df = loader.load_transaction_data()

            # Should handle missing values gracefully
            assert len(result_df) == 2
            assert not result_df['Transaction_Amount'].isnull().any()

        finally:
            os.unlink(temp_file)

    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        # Create test data with wrong types
        test_data = {
            'Transaction_ID': ['TXN_1', 'TXN_2'],
            'Transaction_Amount': ['invalid', 100.0],  # String instead of float
            'Fraud_Label': [0, 1]
        }

        df = pd.DataFrame(test_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            loader = TransactionDataLoader(temp_file)
            # Should handle or report data type issues
            result_df = loader.load_transaction_data()
            assert result_df is not None

        finally:
            os.unlink(temp_file)

    def test_empty_dataset(self):
        """Test handling of empty dataset"""
        # Create empty CSV
        test_data = {
            'Transaction_ID': [],
            'Transaction_Amount': [],
            'Fraud_Label': []
        }

        df = pd.DataFrame(test_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            loader = TransactionDataLoader(temp_file)
            result_df = loader.load_transaction_data()

            # Should handle empty dataset
            assert len(result_df) == 0

        finally:
            os.unlink(temp_file)

class TestConfiguration:
    """Test configuration management"""

    def test_config_validation(self):
        """Test configuration validation"""
        # Test with missing API key
        original_key = config.settings.OPENROUTER_API_KEY
        config.settings.OPENROUTER_API_KEY = ""

        try:
            with pytest.raises(ValueError):
                config.validate_config()
        finally:
            config.settings.OPENROUTER_API_KEY = original_key

    def test_settings_access(self):
        """Test settings access"""
        settings = config.get_settings()

        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
        assert settings.BATCH_SIZE == 10
        assert settings.HIGH_VALUE_THRESHOLD == 1000.0

# Performance benchmarking functions
def benchmark_data_loading():
    """Benchmark data loading performance"""
    start_time = time.time()

    loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
    df = loader.load_transaction_data()

    end_time = time.time()
    duration = end_time - start_time

    print(f"Data loading: {duration:.3f}s for {len(df)} transactions")
    return duration

def benchmark_finbert_analysis():
    """Benchmark FinBERT analysis performance"""
    finbert_analyzer = FinBERTAnalyzer()

    # Load sample data
    loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
    df = loader.load_transaction_data()

    if len(df) == 0:
        return 0.0

    sample_transaction = df.iloc[0].to_dict()

    start_time = time.time()
    result = finbert_analyzer.analyze_transaction_sentiment(sample_transaction)
    end_time = time.time()

    duration = end_time - start_time
    print(f"FinBERT analysis: {duration:.3f}s")
    return duration

def benchmark_rag_system():
    """Benchmark RAG system performance"""
    # Load sample data
    loader = TransactionDataLoader("synthetic_fraud_dataset.csv")
    df = loader.load_transaction_data()

    kb = FraudKnowledgeBase()
    kb.build_fraud_knowledge_base(df)

    if len(kb.fraud_cases) == 0:
        return 0.0

    # Test similarity search
    query_case = kb.fraud_cases[0]
    start_time = time.time()
    similar_cases = kb.retrieve_similar_frauds(query_case.embedding, k=3)
    end_time = time.time()

    duration = end_time - start_time
    print(f"RAG similarity search: {duration:.3f}s")
    return duration

# Main test runner
if __name__ == "__main__":
    # Run performance benchmarks
    print("🚀 Running Performance Benchmarks...")
    print("=" * 50)

    try:
        data_load_time = benchmark_data_loading()
        finbert_time = benchmark_finbert_analysis()
        rag_time = benchmark_rag_system()

        print("\n📊 Performance Summary:")
        print(f"Data Loading: {data_load_time:.3f}s")
        print(f"FinBERT Analysis: {finbert_time:.3f}s")
        print(f"RAG Search: {rag_time:.3f}s")

        total_time = data_load_time + finbert_time + rag_time
        print(f"Total: {total_time:.3f}s")

        # Performance targets
        targets_met = {
            "Data Loading < 5s": data_load_time < 5.0,
            "FinBERT < 10s": finbert_time < 10.0,
            "RAG < 5s": rag_time < 5.0,
            "Total < 20s": total_time < 20.0
        }

        print("\n🎯 Performance Targets:")
        for target, met in targets_met.items():
            status = "✅ MET" if met else "❌ NOT MET"
            print(f"{status}: {target}")

    except Exception as e:
        print(f"❌ Benchmark error: {e}")

    print("\n" + "=" * 50)
    print("🧪 Running Unit Tests...")

    # Run pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])
