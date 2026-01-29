"""
Data Processing Module for Fraud Detection System
Component 1: Robust data pipeline for transaction processing
"""
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionDataLoader:
    """Handles loading and processing of transaction data"""

    # Expected columns in the dataset
    EXPECTED_COLUMNS = [
        'Transaction_ID', 'User_ID', 'Transaction_Amount', 'Transaction_Type',
        'Timestamp', 'Account_Balance', 'Device_Type', 'Location',
        'Merchant_Category', 'IP_Address_Flag', 'Previous_Fraudulent_Activity',
        'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d',
        'Failed_Transaction_Count_7d', 'Card_Type', 'Card_Age',
        'Transaction_Distance', 'Authentication_Method', 'Risk_Score',
        'Is_Weekend', 'Fraud_Label'
    ]

    def __init__(self, filepath: str = "synthetic_fraud_dataset.csv"):
        """Initialize data loader with dataset path"""
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

    def load_transaction_data(self) -> pd.DataFrame:
        """
        Load transaction data from CSV file with validation

        Returns:
            pd.DataFrame: Loaded and validated transaction data

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data validation fails
        """
        try:
            logger.info(f"Loading transaction data from {self.filepath}")

            # Load CSV file
            self.df = pd.read_csv(self.filepath)

            # Validate columns
            missing_columns = set(self.EXPECTED_COLUMNS) - set(self.df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Handle missing values
            self.df = self._handle_missing_values()

            # Validate data types
            self._validate_data_types()

            # Convert timestamp column
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])

            logger.info(f"Successfully loaded {len(self.df)} transactions")
            return self.df

        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")
        except Exception as e:
            logger.error(f"Error loading transaction data: {str(e)}")
            raise

    def _handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = self.df.copy()

        # For numeric columns, fill with median
        numeric_columns = [
            'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count',
            'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d',
            'Card_Age', 'Transaction_Distance', 'Risk_Score'
        ]

        for col in numeric_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # For categorical columns, fill with mode
        categorical_columns = [
            'Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category',
            'Card_Type', 'Authentication_Method'
        ]

        for col in categorical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        # For boolean columns, fill with False (most conservative approach)
        boolean_columns = ['IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Is_Weekend']
        for col in boolean_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(0)

        return df

    def _validate_data_types(self) -> None:
        """Validate that columns have expected data types"""
        # Define expected types for each column
        expected_types = {
            'Transaction_ID': 'object',
            'User_ID': 'object',
            'Transaction_Amount': 'float64',
            'Transaction_Type': 'object',
            'Account_Balance': 'float64',
            'Device_Type': 'object',
            'Location': 'object',
            'Merchant_Category': 'object',
            'IP_Address_Flag': 'int64',
            'Previous_Fraudulent_Activity': 'int64',
            'Daily_Transaction_Count': 'int64',
            'Avg_Transaction_Amount_7d': 'float64',
            'Failed_Transaction_Count_7d': 'int64',
            'Card_Type': 'object',
            'Card_Age': 'int64',
            'Transaction_Distance': 'float64',
            'Authentication_Method': 'object',
            'Risk_Score': 'float64',
            'Is_Weekend': 'int64',
            'Fraud_Label': 'int64'
        }

        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if actual_type != expected_type:
                    logger.warning(f"Column {col}: expected {expected_type}, got {actual_type}")

    def row_to_fraud_analysis_prompt(self, row: pd.Series) -> Dict[str, Any]:
        """
        Transform a transaction row into a detailed fraud analysis prompt

        Args:
            row: Pandas Series representing a transaction row

        Returns:
            Dict containing transaction data and formatted prompt
        """
        # Build transaction metadata section
        metadata = f"""
        Transaction ID: {row.get('Transaction_ID', 'Unknown')}
        User ID: {row.get('User_ID', 'Unknown')}
        Amount: ${row.get('Transaction_Amount', 0):.2f}
        Type: {row.get('Transaction_Type', 'Unknown')}
        Timestamp: {row.get('Timestamp', 'Unknown')}
        Location: {row.get('Location', 'Unknown')}
        """

        # Build user behavior patterns section
        behavior = f"""
        Account Balance: ${row.get('Account_Balance', 0):.2f}
        Daily Transaction Count: {row.get('Daily_Transaction_Count', 0)}
        7-Day Average Amount: ${row.get('Avg_Transaction_Amount_7d', 0):.2f}
        Failed Transactions (7d): {row.get('Failed_Transaction_Count_7d', 0)}
        Previous Fraud Activity: {'Yes' if row.get('Previous_Fraudulent_Activity', 0) == 1 else 'No'}
        """

        # Build risk indicators section
        risk_indicators = f"""
        Device Type: {row.get('Device_Type', 'Unknown')}
        Merchant Category: {row.get('Merchant_Category', 'Unknown')}
        Card Type: {row.get('Card_Type', 'Unknown')}
        Card Age: {row.get('Card_Age', 0)} days
        Transaction Distance: {row.get('Transaction_Distance', 0):.2f} km
        Authentication Method: {row.get('Authentication_Method', 'Unknown')}
        IP Address Flag: {'Yes' if row.get('IP_Address_Flag', 0) == 1 else 'No'}
        Risk Score: {row.get('Risk_Score', 0):.4f}
        Is Weekend: {'Yes' if row.get('Is_Weekend', 0) == 1 else 'No'}
        """

        # Build contextual factors section
        context = f"""
        This transaction involves a {row.get('Transaction_Type', 'Unknown').lower()}
        transaction for ${row.get('Transaction_Amount', 0):.2f} at a {row.get('Merchant_Category', 'Unknown').lower()} merchant.
        The user has ${row.get('Account_Balance', 0):.2f} in their account and has made
        {row.get('Daily_Transaction_Count', 0)} transactions today with an average of
        ${row.get('Avg_Transaction_Amount_7d', 0):.2f} over the past 7 days.
        """

        # Create the complete analysis prompt using Chain-of-Thought structure
        analysis_prompt = f"""
        Analyze this financial transaction for potential fraud:

        TRANSACTION METADATA:
        {metadata}

        USER BEHAVIOR PATTERNS:
        {behavior}

        RISK INDICATORS:
        {risk_indicators}

        CONTEXTUAL INFORMATION:
        {context}

        Analyze this transaction step-by-step:
        1) Identify anomalies in transaction patterns, amounts, or behavior
        2) Assess risk factors including device, location, and authentication
        3) Compare to normal behavior patterns for this user
        4) Provide fraud probability score (0-1, where 1 is definitely fraud)
        5) Explain your reasoning for the fraud assessment

        Provide your analysis in a clear, structured format.
        """

        return {
            'transaction_id': row.get('Transaction_ID'),
            'raw_data': row.to_dict(),
            'analysis_prompt': analysis_prompt.strip(),
            'amount': row.get('Transaction_Amount', 0),
            'is_high_value': row.get('Transaction_Amount', 0) > 1000.0,
            'fraud_label': row.get('Fraud_Label', 0)
        }

    def prepare_batch_transactions(self, batch_size: int = 10) -> Iterator[List[Dict[str, Any]]]:
        """
        Prepare transaction batches for efficient processing

        Args:
            batch_size: Number of transactions per batch

        Yields:
            List of transaction prompt dictionaries
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_transaction_data() first.")

        transactions = []

        for _, row in self.df.iterrows():
            transaction_prompt = self.row_to_fraud_analysis_prompt(row)
            transactions.append(transaction_prompt)

            if len(transactions) >= batch_size:
                yield transactions
                transactions = []

        # Yield remaining transactions
        if transactions:
            yield transactions

    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific transaction by ID

        Args:
            transaction_id: The transaction ID to search for

        Returns:
            Transaction data dictionary or None if not found
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_transaction_data() first.")

        transaction_row = self.df[self.df['Transaction_ID'] == transaction_id]

        if transaction_row.empty:
            return None

        return self.row_to_fraud_analysis_prompt(transaction_row.iloc[0])

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistical information about the dataset

        Returns:
            Dictionary containing dataset statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_transaction_data() first.")

        return {
            'total_transactions': len(self.df),
            'fraud_transactions': int(self.df['Fraud_Label'].sum()),
            'legitimate_transactions': len(self.df) - int(self.df['Fraud_Label'].sum()),
            'fraud_rate': float(self.df['Fraud_Label'].mean()),
            'avg_amount': float(self.df['Transaction_Amount'].mean()),
            'max_amount': float(self.df['Transaction_Amount'].max()),
            'min_amount': float(self.df['Transaction_Amount'].min()),
            'unique_users': self.df['User_ID'].nunique(),
            'unique_merchants': self.df['Merchant_Category'].nunique(),
            'date_range': {
                'start': str(self.df['Timestamp'].min()),
                'end': str(self.df['Timestamp'].max())
            }
        }

# Convenience function for easy data loading
def load_transaction_data(filepath: str) -> pd.DataFrame:
    """
    Convenience function to load transaction data

    Args:
        filepath: Path to the CSV dataset file

    Returns:
        pd.DataFrame: Loaded transaction data
    """
    loader = TransactionDataLoader(filepath)
    return loader.load_transaction_data()

if __name__ == "__main__":
    # Test the data loader
    try:
        loader = TransactionDataLoader()
        df = loader.load_transaction_data()

        print("Dataset Statistics:")
        stats = loader.get_dataset_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nSample Transaction Prompt:")
        if len(df) > 0:
            sample_prompt = loader.row_to_fraud_analysis_prompt(df.iloc[0])
            print(sample_prompt['analysis_prompt'][:200] + "...")

        print("\nBatch Processing Test:")
        batch_count = 0
        for batch in loader.prepare_batch_transactions(batch_size=5):
            batch_count += 1
            print(f"Batch {batch_count}: {len(batch)} transactions")
            if batch_count >= 2:  # Just show first 2 batches
                break

    except Exception as e:
        print(f"Error testing data loader: {e}")
