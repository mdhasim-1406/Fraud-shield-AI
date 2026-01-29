"""
FinBERT Sentiment Analysis Module for Fraud Detection System
Component 2: Financial sentiment and behavioral anomaly detection
"""
import torch
import logging
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Structured sentiment analysis result"""
    positive: float
    negative: float
    neutral: float
    confidence: float
    label: str

@dataclass
class AnomalyResult:
    """Structured anomaly detection result"""
    anomaly_flags: List[str]
    anomaly_score: float
    risk_factors: Dict[str, float]

class FinBERTAnalyzer:
    """FinBERT-based financial sentiment and anomaly detection"""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto"):
        """
        Initialize FinBERT analyzer

        Args:
            model_name: Hugging Face model name
            device: Device to run model on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.sentiment_pipeline = None

        # Load model and tokenizer
        self._load_model()

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self) -> None:
        """Load FinBERT model and tokenizer"""
        try:
            logger.info(f"Loading FinBERT model: {self.model_name} on {self.device}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # positive, negative, neutral
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            # Create sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )

            logger.info("FinBERT model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading FinBERT model: {str(e)}")
            raise

    def build_financial_context(self, row: Dict[str, Any]) -> str:
        """
        Build natural language description of financial behavior

        Args:
            row: Transaction data dictionary

        Returns:
            str: Formatted financial context description
        """
        try:
            # Extract key information from transaction row
            user_id = row.get('User_ID', 'Unknown')
            transaction_type = row.get('Transaction_Type', 'Unknown')
            amount = row.get('Transaction_Amount', 0)
            merchant_category = row.get('Merchant_Category', 'Unknown')
            account_balance = row.get('Account_Balance', 0)
            daily_count = row.get('Daily_Transaction_Count', 0)
            avg_amount_7d = row.get('Avg_Transaction_Amount_7d', 0)
            failed_count_7d = row.get('Failed_Transaction_Count_7d', 0)
            auth_method = row.get('Authentication_Method', 'Unknown')
            device_type = row.get('Device_Type', 'Unknown')

            # Build contextual description
            context = f"""
            User {user_id} conducting {transaction_type} transaction for ${amount:.2f} at {merchant_category} merchant.
            Account balance: ${account_balance:.2f}. Recent activity: {daily_count} transactions today,
            average ${avg_amount_7d:.2f} over 7 days, {failed_count_7d} failed attempts.
            Authentication via {auth_method} from {device_type} device.
            """

            return context.strip()

        except Exception as e:
            logger.error(f"Error building financial context: {str(e)}")
            return f"Transaction analysis for user {row.get('User_ID', 'Unknown')}"

    def get_finbert_sentiment(self, transaction_text: str) -> SentimentResult:
        """
        Get FinBERT sentiment scores for transaction text

        Args:
            transaction_text: Input text for sentiment analysis

        Returns:
            SentimentResult: Structured sentiment analysis result
        """
        try:
            if not self.sentiment_pipeline:
                raise ValueError("Model not loaded. Call _load_model() first.")

            # Get sentiment scores
            results = self.sentiment_pipeline(transaction_text)

            # Extract scores for each sentiment
            sentiment_scores = {}
            for result in results[0]:
                sentiment_scores[result['label'].lower()] = result['score']

            # Create result object
            sentiment_result = SentimentResult(
                positive=sentiment_scores.get('positive', 0.0),
                negative=sentiment_scores.get('negative', 0.0),
                neutral=sentiment_scores.get('neutral', 0.0),
                confidence=max(sentiment_scores.values()) if sentiment_scores else 0.0,
                label=max(sentiment_scores, key=sentiment_scores.get) if sentiment_scores else 'neutral'
            )

            return sentiment_result

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Return neutral sentiment on error
            return SentimentResult(
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                confidence=0.0,
                label='neutral'
            )

    def detect_behavioral_anomalies(self, row: Dict[str, Any],
                                   sentiment_scores: SentimentResult) -> AnomalyResult:
        """
        Detect behavioral anomalies in transaction patterns

        Args:
            row: Transaction data dictionary
            sentiment_scores: FinBERT sentiment analysis results

        Returns:
            AnomalyResult: Structured anomaly detection result
        """
        try:
            anomaly_flags = []
            risk_factors = {}

            # 1. High transaction amount vs account balance ratio
            amount = row.get('Transaction_Amount', 0)
            balance = row.get('Account_Balance', 0)

            if balance > 0:
                amount_balance_ratio = amount / balance
                if amount_balance_ratio > 0.8:  # Using 80% of account balance
                    anomaly_flags.append("High amount-to-balance ratio")
                    risk_factors['amount_balance_ratio'] = amount_balance_ratio

            # 2. Transaction distance analysis
            transaction_distance = row.get('Transaction_Distance', 0)
            if transaction_distance > 1000:  # Distance > 1000 km
                anomaly_flags.append("Unusual transaction distance")
                risk_factors['transaction_distance'] = transaction_distance

            # 3. Failed transaction spike
            failed_count = row.get('Failed_Transaction_Count_7d', 0)
            daily_count = row.get('Daily_Transaction_Count', 0)

            if failed_count > 0 and daily_count > 0:
                failed_ratio = failed_count / (failed_count + daily_count)
                if failed_ratio > 0.3:  # More than 30% failed transactions
                    anomaly_flags.append("High failed transaction ratio")
                    risk_factors['failed_transaction_ratio'] = failed_ratio

            # 4. Device type mismatch (if we had user profile data)
            device_type = row.get('Device_Type', 'Unknown')
            if device_type in ['Unknown', 'Other']:
                anomaly_flags.append("Unusual device type")
                risk_factors['device_anomaly'] = 1.0

            # 5. IP address flags combined with negative sentiment
            ip_flag = row.get('IP_Address_Flag', 0)
            if ip_flag == 1 and sentiment_scores.negative > 0.6:
                anomaly_flags.append("Suspicious IP with negative sentiment")
                risk_factors['ip_sentiment_risk'] = sentiment_scores.negative

            # 6. Weekend transaction analysis
            is_weekend = row.get('Is_Weekend', 0)
            if is_weekend == 1 and row.get('Merchant_Category') in ['Gambling', 'Adult Entertainment']:
                anomaly_flags.append("Weekend high-risk merchant")
                risk_factors['weekend_risk_merchant'] = 1.0

            # 7. Card age vs transaction amount
            card_age = row.get('Card_Age', 0)
            if card_age < 30 and amount > 500:  # New card, high amount
                anomaly_flags.append("New card high-value transaction")
                risk_factors['new_card_high_amount'] = amount / 500.0

            # Calculate overall anomaly score
            base_score = len(anomaly_flags) * 0.2  # Base score from flags

            # Add weighted risk factors
            for factor, value in risk_factors.items():
                if factor in ['amount_balance_ratio', 'failed_transaction_ratio']:
                    base_score += min(value * 0.3, 0.3)  # Cap at 0.3
                elif factor in ['transaction_distance']:
                    base_score += min(value / 10000, 0.2)  # Normalize distance
                else:
                    base_score += value * 0.1  # Other factors

            anomaly_score = min(base_score, 1.0)  # Cap at 1.0

            return AnomalyResult(
                anomaly_flags=anomaly_flags,
                anomaly_score=anomaly_score,
                risk_factors=risk_factors
            )

        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return AnomalyResult(
                anomaly_flags=["Error in analysis"],
                anomaly_score=0.5,
                risk_factors={}
            )

    def analyze_transaction_sentiment(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete sentiment and anomaly analysis for a transaction

        Args:
            row: Transaction data dictionary

        Returns:
            Dict containing complete analysis results
        """
        try:
            # Build financial context
            context_text = self.build_financial_context(row)

            # Get sentiment analysis
            sentiment_result = self.get_finbert_sentiment(context_text)

            # Detect anomalies
            anomaly_result = self.detect_behavioral_anomalies(row, sentiment_result)

            # Combine results
            analysis_result = {
                'sentiment_positive': sentiment_result.positive,
                'sentiment_negative': sentiment_result.negative,
                'sentiment_neutral': sentiment_result.neutral,
                'confidence': sentiment_result.confidence,
                'sentiment_label': sentiment_result.label,
                'anomaly_flags': anomaly_result.anomaly_flags,
                'anomaly_score': anomaly_result.anomaly_score,
                'risk_factors': anomaly_result.risk_factors,
                'context_text': context_text,
                'overall_risk_score': (sentiment_result.negative * 0.4 +
                                     anomaly_result.anomaly_score * 0.6)
            }

            return analysis_result

        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            return {
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 1.0,
                'confidence': 0.0,
                'sentiment_label': 'neutral',
                'anomaly_flags': ['Analysis error'],
                'anomaly_score': 0.5,
                'risk_factors': {},
                'context_text': '',
                'overall_risk_score': 0.5
            }

# Convenience function for easy sentiment analysis
def analyze_financial_sentiment(transaction_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for financial sentiment analysis

    Args:
        transaction_row: Transaction data dictionary

    Returns:
        Dict containing sentiment and anomaly analysis
    """
    analyzer = FinBERTAnalyzer()
    return analyzer.analyze_transaction_sentiment(transaction_row)

if __name__ == "__main__":
    # Test the FinBERT analyzer
    try:
        print("Testing FinBERT Analyzer...")

        # Initialize analyzer
        analyzer = FinBERTAnalyzer()

        # Sample transaction data
        sample_transaction = {
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

        # Test sentiment analysis
        print("\n1. Testing sentiment analysis...")
        context = analyzer.build_financial_context(sample_transaction)
        print(f"Context: {context[:100]}...")

        sentiment = analyzer.get_finbert_sentiment(context)
        print(f"Sentiment: {sentiment}")

        # Test anomaly detection
        print("\n2. Testing anomaly detection...")
        anomalies = analyzer.detect_behavioral_anomalies(sample_transaction, sentiment)
        print(f"Anomalies: {anomalies}")

        # Test complete analysis
        print("\n3. Testing complete analysis...")
        result = analyzer.analyze_transaction_sentiment(sample_transaction)
        print(f"Complete result keys: {list(result.keys())}")
        print(f"Overall risk score: {result['overall_risk_score']:.3f}")

        print("\n✅ FinBERT Analyzer test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing FinBERT analyzer: {e}")
        import traceback
        traceback.print_exc()
