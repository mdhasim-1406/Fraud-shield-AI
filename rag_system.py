"""
RAG System for Historical Fraud Pattern Retrieval
Component 4: Retrieval-Augmented Generation using FAISS for fraud context
"""
import faiss
import numpy as np
import logging
import json
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FraudCase:
    """Structured representation of a historical fraud case"""
    transaction_id: str
    fraud_indicators: List[str]
    risk_factors: Dict[str, Any]
    merchant_category: str
    amount: float
    user_behavior: Dict[str, Any]
    embedding: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class SimilarFraudCase:
    """Result from similarity search"""
    case: FraudCase
    similarity_score: float
    matching_indicators: List[str]

class FraudKnowledgeBase:
    """FAISS-based vector database for fraud pattern storage and retrieval"""

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "models/fraud_index.faiss",
                 embedding_dim: int = 384):
        """
        Initialize fraud knowledge base

        Args:
            embedding_model: Sentence transformer model name
            index_path: Path to save/load FAISS index
            embedding_dim: Dimension of embeddings
        """
        self.embedding_model_name = embedding_model
        self.index_path = index_path
        self.embedding_dim = embedding_dim

        # Initialize components
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.fraud_cases: List[FraudCase] = []
        self.case_mapping: Dict[str, int] = {}  # transaction_id -> index

        # Load embedding model
        self._load_embedding_model()

        # Load or create FAISS index
        self._setup_faiss_index()

    def _load_embedding_model(self) -> None:
        """Load sentence transformer model for embeddings"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def _setup_faiss_index(self) -> None:
        """Set up FAISS index - load existing or create new"""
        try:
            # Ensure models directory exists
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            if os.path.exists(self.index_path):
                try:
                    # Try to load existing index
                    logger.info(f"Loading existing FAISS index from {self.index_path}")
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Successfully loaded FAISS index with {self.index.ntotal if self.index else 0} vectors")
                except Exception as load_error:
                    # If loading fails, delete corrupted file and create new
                    logger.warning(f"Failed to load existing FAISS index: {load_error}")
                    logger.info("Deleting corrupted index file and creating new one")
                    try:
                        os.remove(self.index_path)
                    except Exception as del_error:
                        logger.warning(f"Could not delete corrupted index file: {del_error}")

                    # Create new index
                    logger.info("Creating new FAISS index")
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                # Create new index
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatL2(self.embedding_dim)

        except Exception as e:
            logger.error(f"Error setting up FAISS index: {e}")
            # Create fallback index even if there are issues
            logger.info("Creating fallback FAISS index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def _extract_fraud_indicators(self, row: pd.Series) -> List[str]:
        """Extract fraud indicators from transaction row"""
        indicators = []

        # Amount-based indicators
        amount = row.get('Transaction_Amount', 0)
        balance = row.get('Account_Balance', 0)

        if balance > 0:
            ratio = amount / balance
            if ratio > 0.8:
                indicators.append("high_amount_balance_ratio")
            elif ratio > 0.5:
                indicators.append("moderate_amount_balance_ratio")

        if amount > 1000:
            indicators.append("high_transaction_amount")

        # Pattern-based indicators
        daily_count = row.get('Daily_Transaction_Count', 0)
        avg_7d = row.get('Avg_Transaction_Amount_7d', 0)

        if daily_count > 5:
            indicators.append("high_daily_transaction_count")

        if avg_7d > 0 and amount > avg_7d * 3:
            indicators.append("amount_significantly_above_average")

        # Failed transaction indicators
        failed_count = row.get('Failed_Transaction_Count_7d', 0)
        if failed_count > 3:
            indicators.append("multiple_failed_attempts")

        # Device and location indicators
        if row.get('IP_Address_Flag', 0) == 1:
            indicators.append("ip_address_flag")

        if row.get('Transaction_Distance', 0) > 1000:
            indicators.append("unusual_transaction_distance")

        # Authentication indicators
        auth_method = row.get('Authentication_Method', 'Unknown')
        if auth_method in ['Password', 'Unknown']:
            indicators.append("weak_authentication")

        # Weekend and timing indicators
        if row.get('Is_Weekend', 0) == 1:
            indicators.append("weekend_transaction")

        # Card age indicators
        card_age = row.get('Card_Age', 0)
        if card_age < 30:
            indicators.append("new_card")
        elif card_age > 365:
            indicators.append("old_card")

        return indicators

    def _extract_risk_factors(self, row: pd.Series) -> Dict[str, Any]:
        """Extract risk factors from transaction row"""
        return {
            'amount_balance_ratio': row.get('Transaction_Amount', 0) / max(row.get('Account_Balance', 1), 1),
            'daily_transaction_count': row.get('Daily_Transaction_Count', 0),
            'failed_transaction_ratio': row.get('Failed_Transaction_Count_7d', 0) / max(row.get('Daily_Transaction_Count', 1), 1),
            'transaction_distance': row.get('Transaction_Distance', 0),
            'card_age_days': row.get('Card_Age', 0),
            'risk_score': row.get('Risk_Score', 0),
            'ip_flag': row.get('IP_Address_Flag', 0),
            'weekend_flag': row.get('Is_Weekend', 0)
        }

    def _extract_user_behavior(self, row: pd.Series) -> Dict[str, Any]:
        """Extract user behavior patterns from transaction row"""
        return {
            'avg_7d_amount': row.get('Avg_Transaction_Amount_7d', 0),
            'merchant_category': row.get('Merchant_Category', 'Unknown'),
            'device_type': row.get('Device_Type', 'Unknown'),
            'authentication_method': row.get('Authentication_Method', 'Unknown'),
            'location': row.get('Location', 'Unknown'),
            'card_type': row.get('Card_Type', 'Unknown'),
            'previous_fraud': row.get('Previous_Fraudulent_Activity', 0)
        }

    def _create_case_text(self, row: pd.Series) -> str:
        """Create searchable text representation of fraud case"""
        fraud_indicators = self._extract_fraud_indicators(row)
        risk_factors = self._extract_risk_factors(row)
        user_behavior = self._extract_user_behavior(row)

        # Build comprehensive case description
        case_text = f"""
        Fraud Case: Transaction {row.get('Transaction_ID', 'Unknown')}
        Amount: ${row.get('Transaction_Amount', 0):.2f}
        Merchant: {row.get('Merchant_Category', 'Unknown')}
        Location: {row.get('Location', 'Unknown')}
        Device: {row.get('Device_Type', 'Unknown')}
        Card Type: {row.get('Card_Type', 'Unknown')}
        Authentication: {row.get('Authentication_Method', 'Unknown')}

        Fraud Indicators: {', '.join(fraud_indicators)}
        Risk Score: {row.get('Risk_Score', 0):.3f}
        Amount-Balance Ratio: {risk_factors.get('amount_balance_ratio', 0):.3f}
        Daily Transactions: {risk_factors.get('daily_transaction_count', 0)}
        Failed Transaction Ratio: {risk_factors.get('failed_transaction_ratio', 0):.3f}
        Transaction Distance: {risk_factors.get('transaction_distance', 0):.1f} km
        Card Age: {risk_factors.get('card_age_days', 0)} days

        User Behavior Pattern: {user_behavior.get('avg_7d_amount', 0):.2f} average over 7 days,
        Previous Fraud: {'Yes' if user_behavior.get('previous_fraud', 0) == 1 else 'No'}
        """

        return case_text.strip()

    def build_fraud_knowledge_base(self, df: pd.DataFrame) -> int:
        """
        Build knowledge base from fraud cases in dataset

        Args:
            df: DataFrame containing transaction data

        Returns:
            int: Number of fraud cases added to knowledge base
        """
        try:
            logger.info("Building fraud knowledge base from dataset...")

            # Filter confirmed fraud cases
            fraud_df = df[df['Fraud_Label'] == 1].copy()

            if len(fraud_df) == 0:
                logger.warning("No fraud cases found in dataset")
                return 0

            logger.info(f"Found {len(fraud_df)} confirmed fraud cases")

            # Process each fraud case
            embeddings = []
            new_cases = []

            for _, row in fraud_df.iterrows():
                try:
                    # Create case text for embedding
                    case_text = self._create_case_text(row)

                    # Generate embedding using sentence-transformers
                    embedding = self.embedding_model.encode([case_text])[0]
                    embeddings.append(embedding)

                    # Extract case information
                    fraud_indicators = self._extract_fraud_indicators(row)
                    risk_factors = self._extract_risk_factors(row)
                    user_behavior = self._extract_user_behavior(row)

                    # Create fraud case object
                    fraud_case = FraudCase(
                        transaction_id=row.get('Transaction_ID', 'Unknown'),
                        fraud_indicators=fraud_indicators,
                        risk_factors=risk_factors,
                        merchant_category=row.get('Merchant_Category', 'Unknown'),
                        amount=row.get('Transaction_Amount', 0),
                        user_behavior=user_behavior,
                        embedding=embedding,
                        metadata=row.to_dict()
                    )

                    new_cases.append(fraud_case)

                except Exception as e:
                    logger.warning(f"Error processing fraud case {row.get('Transaction_ID', 'Unknown')}: {e}")
                    continue

            if not new_cases:
                logger.warning("No valid fraud cases could be processed")
                return 0

            # Convert embeddings to numpy array
            embedding_array = np.array(embeddings).astype('float32')

            # Add to FAISS index
            start_idx = len(self.fraud_cases)
            self.index.add(embedding_array)

            # Add cases to knowledge base
            self.fraud_cases.extend(new_cases)

            # Update case mapping
            for i, case in enumerate(new_cases):
                self.case_mapping[case.transaction_id] = start_idx + i

            # Save updated index
            self._save_faiss_index()

            logger.info(f"Successfully added {len(new_cases)} fraud cases to knowledge base")
            return len(new_cases)

        except Exception as e:
            logger.error(f"Error building fraud knowledge base: {e}")
            raise

    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def retrieve_similar_frauds(self,
                              transaction_embedding: np.ndarray,
                              k: int = 3) -> List[SimilarFraudCase]:
        """
        Retrieve top-k most similar historical fraud cases

        Args:
            transaction_embedding: Embedding vector for current transaction
            k: Number of similar cases to retrieve

        Returns:
            List of SimilarFraudCase objects with similarity scores
        """
        try:
            if len(self.fraud_cases) == 0:
                logger.warning("No fraud cases in knowledge base")
                return []

            # Ensure embedding is correct shape and type
            query_embedding = np.array([transaction_embedding]).astype('float32')

            # Search for similar cases
            distances, indices = self.index.search(query_embedding, min(k, len(self.fraud_cases)))

            # Convert distances to similarity scores (1 - normalized_distance)
            max_distance = np.max(distances[0]) if len(distances[0]) > 0 else 1
            similarities = 1 - (distances[0] / max_distance) if max_distance > 0 else distances[0]

            # Build results
            similar_cases = []
            for i, (distance, similarity) in enumerate(zip(distances[0], similarities)):
                if indices[0][i] < len(self.fraud_cases):
                    case = self.fraud_cases[indices[0][i]]

                    # Find matching indicators
                    matching_indicators = self._find_matching_indicators(case, query_embedding[0])

                    similar_case = SimilarFraudCase(
                        case=case,
                        similarity_score=float(similarity),
                        matching_indicators=matching_indicators
                    )

                    similar_cases.append(similar_case)

            # Sort by similarity score (highest first)
            similar_cases.sort(key=lambda x: x.similarity_score, reverse=True)

            logger.info(f"Retrieved {len(similar_cases)} similar fraud cases")
            return similar_cases[:k]

        except Exception as e:
            logger.error(f"Error retrieving similar frauds: {e}")
            return []

    def _find_matching_indicators(self, fraud_case: FraudCase,
                                 query_embedding: np.ndarray) -> List[str]:
        """Find indicators that match between fraud case and query"""
        # This is a simplified matching - in practice, you might use more sophisticated methods
        matching = []

        # Check for common risk factors
        if (fraud_case.risk_factors.get('amount_balance_ratio', 0) > 0.5 and
            np.mean(query_embedding) > 0.1):  # Simplified check
            matching.append("high_amount_balance_ratio")

        if fraud_case.risk_factors.get('failed_transaction_ratio', 0) > 0.2:
            matching.append("failed_transaction_pattern")

        if fraud_case.risk_factors.get('transaction_distance', 0) > 500:
            matching.append("unusual_location")

        if fraud_case.metadata.get('IP_Address_Flag', 0) == 1:
            matching.append("ip_address_anomaly")

        return matching

    def get_transaction_embedding(self, transaction_text: str) -> np.ndarray:
        """
        Generate embedding for transaction text

        Args:
            transaction_text: Text description of transaction

        Returns:
            np.ndarray: Embedding vector
        """
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not loaded, using fallback")
                # Return zero vector as fallback
                return np.zeros(self.embedding_dim)

            # Generate embedding using sentence-transformers
            embedding = self.embedding_model.encode([transaction_text])[0]
            return embedding

        except Exception as e:
            logger.error(f"Error generating transaction embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)

    def augment_prompt_with_rag(self,
                              base_prompt: str,
                              similar_cases: List[SimilarFraudCase]) -> str:
        """
        Augment DeepSeek prompt with retrieved fraud cases

        Args:
            base_prompt: Original transaction analysis prompt
            similar_cases: List of similar fraud cases

        Returns:
            str: Augmented prompt with historical context
        """
        try:
            if not similar_cases:
                return base_prompt

            # Build context from similar cases
            context_sections = []

            for i, similar_case in enumerate(similar_cases, 1):
                case = similar_case.case

                # Create case description
                case_description = f"""
                Case {i}: Transaction {case.transaction_id}
                - Amount: ${case.amount:.2f} at {case.merchant_category}
                - Fraud Indicators: {', '.join(case.fraud_indicators)}
                - Risk Factors: Amount-Balance Ratio {case.risk_factors.get('amount_balance_ratio', 0):.2f},
                  Failed Transaction Ratio {case.risk_factors.get('failed_transaction_ratio', 0):.2f}
                - User Behavior: {case.user_behavior.get('avg_7d_amount', 0):.2f} 7-day average,
                  Previous Fraud: {'Yes' if case.user_behavior.get('previous_fraud', 0) == 1 else 'No'}
                - Similarity Score: {similar_case.similarity_score:.3f}
                - Matching Indicators: {', '.join(similar_case.matching_indicators)}
                """

                context_sections.append(case_description.strip())

            # Combine all cases
            historical_context = "\n".join(context_sections)

            # Create augmented prompt
            augmented_prompt = f"""
            Based on these similar historical fraud cases:

            {historical_context}

            Now analyze the current transaction considering these patterns:

            {base_prompt}

            Pay special attention to:
            - Similar fraud indicators and how they manifest in the current case
            - Risk factor patterns that match the historical cases
            - User behavior similarities that suggest potential fraud
            - How the similarity scores indicate pattern strength

            Use these historical patterns to inform your fraud probability assessment.
            """

            return augmented_prompt.strip()

        except Exception as e:
            logger.error(f"Error augmenting prompt with RAG: {e}")
            return base_prompt

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            'total_fraud_cases': len(self.fraud_cases),
            'embedding_dimension': self.embedding_dim,
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_model': self.embedding_model_name,
            'common_indicators': self._get_common_indicators(),
            'merchant_categories': self._get_merchant_distribution(),
            'amount_ranges': self._get_amount_ranges()
        }

    def _get_common_indicators(self) -> Dict[str, int]:
        """Get frequency of fraud indicators"""
        indicator_counts = {}
        for case in self.fraud_cases:
            for indicator in case.fraud_indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

        # Return top 10 most common
        sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_indicators[:10])

    def _get_merchant_distribution(self) -> Dict[str, int]:
        """Get distribution of merchant categories in fraud cases"""
        merchant_counts = {}
        for case in self.fraud_cases:
            merchant = case.merchant_category
            merchant_counts[merchant] = merchant_counts.get(merchant, 0) + 1

        return dict(sorted(merchant_counts.items(), key=lambda x: x[1], reverse=True))

    def _get_amount_ranges(self) -> Dict[str, int]:
        """Get distribution of amount ranges in fraud cases"""
        ranges = {'0-100': 0, '100-500': 0, '500-1000': 0, '1000-5000': 0, '5000+': 0}

        for case in self.fraud_cases:
            amount = case.amount
            if amount <= 100:
                ranges['0-100'] += 1
            elif amount <= 500:
                ranges['100-500'] += 1
            elif amount <= 1000:
                ranges['500-1000'] += 1
            elif amount <= 5000:
                ranges['1000-5000'] += 1
            else:
                ranges['5000+'] += 1

        return ranges

# Convenience functions
def build_fraud_knowledge_base(df: pd.DataFrame) -> FraudKnowledgeBase:
    """
    Build fraud knowledge base from dataset

    Args:
        df: DataFrame containing transaction data

    Returns:
        FraudKnowledgeBase: Initialized and populated knowledge base
    """
    kb = FraudKnowledgeBase()
    kb.build_fraud_knowledge_base(df)
    return kb

def retrieve_similar_frauds(kb: FraudKnowledgeBase,
                          transaction_text: str,
                          k: int = 3) -> List[SimilarFraudCase]:
    """
    Retrieve similar fraud cases for a transaction

    Args:
        kb: FraudKnowledgeBase instance
        transaction_text: Text description of transaction
        k: Number of similar cases to retrieve

    Returns:
        List of similar fraud cases
    """
    embedding = kb.get_transaction_embedding(transaction_text)
    return kb.retrieve_similar_frauds(embedding, k)

if __name__ == "__main__":
    # Test the RAG system
    try:
        print("Testing RAG System...")

        # Load sample data
        from data_loader import TransactionDataLoader

        loader = TransactionDataLoader()
        df = loader.load_transaction_data()

        print(f"\n1. Loaded {len(df)} transactions")

        # Build knowledge base
        print("\n2. Building fraud knowledge base...")
        kb = FraudKnowledgeBase()
        fraud_count = kb.build_fraud_knowledge_base(df)

        print(f"Added {fraud_count} fraud cases to knowledge base")

        # Get knowledge base stats
        print("\n3. Knowledge base statistics:")
        stats = kb.get_knowledge_base_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test similarity search
        print("\n4. Testing similarity search...")
        sample_transaction = df.iloc[0]  # Use first transaction as example
        sample_text = kb._create_case_text(sample_transaction)

        print(f"Sample transaction text: {sample_text[:100]}...")

        embedding = kb.get_transaction_embedding(sample_text)
        similar_cases = kb.retrieve_similar_frauds(embedding, k=3)

        print(f"Found {len(similar_cases)} similar cases:")
        for i, case in enumerate(similar_cases, 1):
            print(f"  Case {i}: {case.case.transaction_id} (similarity: {case.similarity_score:.3f})")

        # Test prompt augmentation
        print("\n5. Testing prompt augmentation...")
        base_prompt = "Analyze this transaction for fraud: User making $500 purchase..."
        augmented_prompt = kb.augment_prompt_with_rag(base_prompt, similar_cases)

        print(f"Original prompt length: {len(base_prompt)}")
        print(f"Augmented prompt length: {len(augmented_prompt)}")
        print(f"Augmented prompt preview: {augmented_prompt[:200]}...")

        print("\n✅ RAG System test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        import traceback
        traceback.print_exc()
