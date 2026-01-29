"""
Real-Time Transaction Simulator for Fraud Detection System
Component: Stateful transaction generator with fraud playbooks (Async Version)
"""
import pandas as pd
import random
import asyncio
import httpx
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

def sanitize_payload_for_api(transaction: dict) -> dict:
    """
    Takes a raw transaction dictionary and cleans/casts every value to be
    JSON-serializable and compliant with the FastAPI Pydantic models.
    This is the single source of truth for data integrity.
    CRITICAL FIX: Ensures all required fields are present and correctly typed.
    """
    sanitized = {}

    # Start with the transaction data and clean it
    for key, value in transaction.items():
        if pd.isna(value):
            # Handle potential NaN values - use correct lowercase field names
            if key in ['transaction_amount', 'account_balance', 'avg_transaction_amount_7d', 'transaction_distance', 'risk_score']:
                sanitized[key] = 0.0
            else:
                sanitized[key] = 0
            continue

        if isinstance(value, (np.integer, np.int64)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            sanitized[key] = float(value)
        elif isinstance(value, datetime):
            sanitized[key] = value.isoformat()
        elif isinstance(value, (int, float, str, bool)):
            sanitized[key] = value
        else:
            # If it's any other type, convert it to a string as a fallback
            sanitized[key] = str(value)

    # CRITICAL FIX: Ensure ALL required fields are present with proper defaults
    # These fields are required by the Transaction Pydantic model in main.py

    # Required string fields
    required_strings = [
        'transaction_id', 'user_id', 'transaction_type', 'device_type',
        'location', 'merchant_category', 'card_type', 'authentication_method'
    ]
    for field in required_strings:
        if field not in sanitized or not isinstance(sanitized[field], str):
            # Generate sensible defaults
            if field == 'transaction_id':
                sanitized[field] = f"TXN_SIM_{int(asyncio.get_event_loop().time())}_{random.randint(1000, 9999)}"
            elif field == 'user_id':
                sanitized[field] = sanitized.get(field, f"USER_SIM_{random.randint(1000, 9999)}")
            elif field == 'transaction_type':
                sanitized[field] = sanitized.get(field, 'Online')
            elif field == 'device_type':
                sanitized[field] = sanitized.get(field, 'Mobile')
            elif field == 'location':
                sanitized[field] = sanitized.get(field, 'Unknown')
            elif field == 'merchant_category':
                sanitized[field] = sanitized.get(field, 'Retail')
            elif field == 'card_type':
                sanitized[field] = sanitized.get(field, 'Visa')
            elif field == 'authentication_method':
                sanitized[field] = sanitized.get(field, 'OTP')

    # Required float fields (must be > 0 for some, >= 0 for others)
    required_floats = [
        ('transaction_amount', 10.0), ('account_balance', 1000.0),
        ('avg_transaction_amount_7d', 100.0), ('transaction_distance', 0.0)
    ]
    for field, default in required_floats:
        if field not in sanitized or not isinstance(sanitized[field], (int, float)):
            sanitized[field] = sanitized.get(field, default)
        # Ensure transaction_amount > 0
        if field == 'transaction_amount' and sanitized[field] <= 0:
            sanitized[field] = default

    # Required int fields (with specific ranges)
    required_ints = [
        ('ip_address_flag', 0), ('previous_fraudulent_activity', 0),
        ('daily_transaction_count', 1), ('failed_transaction_count_7d', 0),
        ('card_age', 365), ('is_weekend', 0)
    ]
    for field, default in required_ints:
        if field not in sanitized or not isinstance(sanitized[field], int):
            sanitized[field] = int(sanitized.get(field, default))
        # Enforce valid ranges
        if field in ['ip_address_flag', 'previous_fraudulent_activity', 'is_weekend'] and sanitized[field] not in [0, 1]:
            sanitized[field] = default
        elif field in ['daily_transaction_count', 'failed_transaction_count_7d', 'card_age'] and sanitized[field] < 0:
            sanitized[field] = default

    # Special handling for timestamp - must be string
    if 'timestamp' in transaction:
        sanitized['timestamp'] = str(transaction['timestamp'])
    else:
        sanitized['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Special handling for risk_score - must be 0.0-1.0
    if 'risk_score' in transaction:
        risk_val = float(transaction['risk_score'])
        sanitized['risk_score'] = max(0.0, min(1.0, risk_val))
    else:
        sanitized['risk_score'] = 0.5  # Default medium risk

    # Optional fraud_label
    if 'fraud_label' not in sanitized or sanitized['fraud_label'] not in [0, 1]:
        sanitized['fraud_label'] = 0

    return sanitized

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserPersona:
    """Stateful user persona for realistic transaction generation"""

    def __init__(self, user_data: Dict[str, Any]):
        """Initialize persona from dataset row"""
        self.user_data = user_data.copy()
        self.last_location = user_data.get('Location', 'New York')
        self.last_txn_time = datetime.now()
        self.daily_txn_count = 0
        self.is_compromised = False
        self.compromise_time: Optional[datetime] = None
        self.device_history = [user_data.get('Device_Type', 'Mobile')]
        self.merchant_history = [user_data.get('Merchant_Category', 'Retail')]

    def update_location(self, new_location: str):
        """Update user's last location"""
        self.last_location = new_location
        self.last_txn_time = datetime.now()

    def increment_daily_count(self):
        """Increment daily transaction counter"""
        self.daily_txn_count += 1

    def mark_compromised(self):
        """Mark user account as compromised"""
        self.is_compromised = True
        self.compromise_time = datetime.now()

    def add_device(self, device: str):
        """Add device to user's device history"""
        if device not in self.device_history:
            self.device_history.append(device)

    def add_merchant(self, merchant: str):
        """Add merchant to user's merchant history"""
        if merchant not in self.merchant_history:
            self.merchant_history.append(merchant)

class FraudPlaybook:
    """Collection of fraud attack scenarios"""

    @staticmethod
    def impossible_travel(persona: UserPersona) -> Dict[str, Any]:
        """Playbook A: Impossible Travel Attack"""
        # Get current location and select distant location
        current_location = persona.last_location

        # Define location pairs that are geographically distant
        location_pairs = {
            'New York': ['London', 'Tokyo', 'Sydney', 'Mumbai'],
            'London': ['New York', 'Tokyo', 'Sydney', 'Mumbai'],
            'Tokyo': ['New York', 'London', 'Sydney', 'Mumbai'],
            'Sydney': ['New York', 'London', 'Tokyo', 'Mumbai'],
            'Mumbai': ['New York', 'London', 'Tokyo', 'Sydney']
        }

        distant_locations = location_pairs.get(current_location, ['London', 'Tokyo'])
        new_location = random.choice(distant_locations)

        # Generate transaction within minutes of last transaction
        txn_time = persona.last_txn_time + timedelta(minutes=random.randint(15, 45))

        # High amount transaction
        base_amount = persona.user_data.get('Avg_Transaction_Amount_7d', 100)
        amount = base_amount * random.uniform(3, 8)  # 3-8x normal amount

        return {
            'location': new_location,
            'transaction_amount': float(round(amount, 2)),
            'timestamp': txn_time.strftime("%Y-%m-%d %H:%M:%S"),
            'merchant_category': random.choice(['Luxury Goods', 'Electronics', 'Travel', 'Jewelry']),
            'device_type': random.choice(['Mobile', 'Desktop', 'Tablet']),
            'authentication_method': random.choice(['Password', 'OTP']),
            'transaction_distance': float(round(random.uniform(5000, 12000), 2)),  # 5000-12000 km
            'ip_address_flag': 1,  # Flag suspicious IP
            'daily_transaction_count': int(persona.daily_txn_count + 1),
            'failed_transaction_count_7d': int(random.randint(2, 5)),
            'risk_score': float(round(random.uniform(0.8, 0.95), 4))
        }

    @staticmethod
    def velocity_attack(persona: UserPersona) -> List[Dict[str, Any]]:
        """Playbook B: Velocity Attack with multiple rapid transactions"""
        transactions = []

        # Generate 5-8 rapid transactions
        num_transactions = random.randint(5, 8)

        for i in range(num_transactions):
            # Transactions within 10 seconds of each other
            txn_time = persona.last_txn_time + timedelta(seconds=i * random.randint(5, 15))

            # Small amounts at online merchants
            amount = random.uniform(10, 150)

            transaction = {
                'location': persona.last_location,
                'transaction_amount': float(round(amount, 2)),
                'timestamp': txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                'transaction_type': 'Online',
                'merchant_category': random.choice(['Online Retail', 'Digital Services', 'Subscription', 'Gaming']),
                'device_type': random.choice(persona.device_history + ['Mobile', 'Desktop']),
                'authentication_method': 'OTP',
                'transaction_distance': float(round(random.uniform(0, 100), 2)),
                'ip_address_flag': random.randint(0, 1),
                'daily_transaction_count': int(persona.daily_txn_count + i + 1),
                'failed_transaction_count_7d': int(random.randint(1, 3)),
                'risk_score': float(round(random.uniform(0.6, 0.8), 4))
            }
            transactions.append(transaction)

        return transactions

    @staticmethod
    def account_drain(persona: UserPersona) -> Dict[str, Any]:
        """Playbook C: Account Drain Attack"""
        # Massive transaction (10x+ normal amount)
        base_amount = persona.user_data.get('Avg_Transaction_Amount_7d', 100)
        amount = base_amount * random.uniform(10, 25)  # 10-25x normal amount

        # New merchant category never used before
        new_merchants = ['Gambling', 'Adult Entertainment', 'Luxury Jewelry', 'Cryptocurrency Exchange']
        current_merchants = persona.merchant_history
        available_merchants = [m for m in new_merchants if m not in current_merchants]
        merchant = random.choice(available_merchants) if available_merchants else random.choice(new_merchants)

        # New device type
        new_devices = ['Smart TV', 'Gaming Console', 'Unknown Device', 'IoT Device']
        current_devices = persona.device_history
        available_devices = [d for d in new_devices if d not in current_devices]
        device = random.choice(available_devices) if available_devices else random.choice(new_devices)

        return {
            'location': random.choice(['International', 'Offshore', 'Unknown']),
            'transaction_amount': float(round(amount, 2)),
            'timestamp': (datetime.now() + timedelta(minutes=random.randint(1, 10))).strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_type': 'Online',
            'merchant_category': merchant,
            'device_type': device,
            'authentication_method': 'Password',  # Weak authentication
            'transaction_distance': float(round(random.uniform(8000, 15000), 2)),  # Very distant
            'ip_address_flag': 1,
            'daily_transaction_count': int(persona.daily_txn_count + 1),
            'failed_transaction_count_7d': int(random.randint(3, 8)),
            'card_age': int(random.randint(1, 7)),  # Very new card
            'risk_score': float(round(random.uniform(0.85, 0.98), 4))
        }

class TransactionSimulator:
    """Main simulator class for generating realistic transaction streams"""

    def __init__(self, csv_path: str = "synthetic_fraud_dataset.csv"):
        """Initialize simulator with dataset"""
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.personas: List[UserPersona] = []
        self.api_url = "http://localhost:8000/api/v1/analyze"
        self.is_running = False

        # Load dataset
        self._load_dataset()

        # Create user personas
        self._create_personas()

    def _load_dataset(self) -> None:
        """Load transaction dataset"""
        try:
            logger.info(f"Loading dataset from {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)

            # CRITICAL FIX: Clean all NaN values immediately after loading
            logger.info("Cleaning dataset - replacing all NaN values with 0")
            self.df.fillna(0, inplace=True)

            # Additional data cleaning for string columns
            string_columns = ['Transaction_ID', 'User_ID', 'Transaction_Type', 'Device_Type',
                            'Location', 'Merchant_Category', 'Card_Type', 'Authentication_Method']
            for col in string_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).fillna('Unknown')

            logger.info(f"Loaded and cleaned {len(self.df)} transactions")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def _create_personas(self) -> None:
        """Create 20-30 user personas from dataset"""
        if self.df is None or len(self.df) == 0:
            raise ValueError("No dataset loaded")

        # Sample 25 random users for personas
        num_personas = min(25, len(self.df))
        sampled_rows = self.df.sample(n=num_personas, random_state=42)

        for _, row in sampled_rows.iterrows():
            persona = UserPersona(row.to_dict())
            self.personas.append(persona)

        logger.info(f"Created {len(self.personas)} user personas")

    def generate_normal_transaction(self, persona: UserPersona) -> Dict[str, Any]:
        """Generate a normal transaction based on persona's profile"""
        # Base transaction on user's historical data
        base_data = persona.user_data.copy()

        # Add some realistic variation
        amount_variation = random.uniform(0.8, 1.3)  # ±20% variation
        amount = base_data.get('Transaction_Amount', 100) * amount_variation

        # Slight location variation (nearby cities)
        location_variation = random.uniform(0, 1)
        if location_variation < 0.1:  # 10% chance of different location
            nearby_locations = {
                'New York': ['New Jersey', 'Connecticut', 'Philadelphia'],
                'London': ['Manchester', 'Birmingham', 'Bristol'],
                'Tokyo': ['Yokohama', 'Osaka', 'Nagoya'],
                'Sydney': ['Melbourne', 'Brisbane', 'Perth'],
                'Mumbai': ['Delhi', 'Bangalore', 'Pune']
            }
            current_location = persona.last_location
            nearby = nearby_locations.get(current_location, [current_location])
            new_location = random.choice(nearby)
            persona.update_location(new_location)
        else:
            new_location = persona.last_location

        # Update daily count
        persona.increment_daily_count()

        # Generate transaction
        transaction = {
            'transaction_id': f"TXN_SIM_{int(asyncio.get_event_loop().time())}_{random.randint(1000, 9999)}",
            'user_id': base_data.get('User_ID'),
            'transaction_amount': float(round(amount, 2)),
            'transaction_type': base_data.get('Transaction_Type', 'POS'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'account_balance': float(base_data.get('Account_Balance', 1000)),
            'device_type': base_data.get('Device_Type', 'Mobile'),
            'location': new_location,
            'merchant_category': base_data.get('Merchant_Category', 'Retail'),
            'ip_address_flag': int(base_data.get('IP_Address_Flag', 0)),
            'previous_fraudulent_activity': int(base_data.get('Previous_Fraudulent_Activity', 0)),
            'daily_transaction_count': int(persona.daily_txn_count),
            'avg_transaction_amount_7d': float(base_data.get('Avg_Transaction_Amount_7d', 100)),
            'failed_transaction_count_7d': int(base_data.get('Failed_Transaction_Count_7d', 0)),
            'card_type': base_data.get('Card_Type', 'Visa'),
            'card_age': int(base_data.get('Card_Age', 365)),
            'transaction_distance': float(round(random.uniform(0, 50), 2)),  # Local distance
            'authentication_method': base_data.get('Authentication_Method', 'Biometric'),
            'risk_score': float(round(random.uniform(0.1, 0.3), 4)),  # Low risk for normal transactions
            'is_weekend': int(base_data.get('Is_Weekend', 0)),
            'fraud_label': 0  # Normal transaction
        }

        return transaction

    def execute_fraud_playbook(self, persona: UserPersona) -> List[Dict[str, Any]]:
        """Execute random fraud playbook"""
        playbooks = ['impossible_travel', 'velocity_attack', 'account_drain']
        playbook = random.choice(playbooks)

        if playbook == 'impossible_travel':
            transaction = FraudPlaybook.impossible_travel(persona)
            transaction = self._ensure_complete_transaction_data(transaction, persona)
            return [transaction]

        elif playbook == 'velocity_attack':
            transactions = FraudPlaybook.velocity_attack(persona)
            # Complete all transactions with necessary data
            for txn in transactions:
                txn = self._ensure_complete_transaction_data(txn, persona)
            return transactions

        elif playbook == 'account_drain':
            transaction = FraudPlaybook.account_drain(persona)
            transaction = self._ensure_complete_transaction_data(transaction, persona)
            return [transaction]

        return []

    def _get_base_transaction_data(self, persona: UserPersona) -> Dict[str, Any]:
        """Get base transaction data for fraud scenarios"""
        return {
            'transaction_id': f"TXN_SIM_{int(asyncio.get_event_loop().time())}_{random.randint(1000, 9999)}",
            'user_id': persona.user_data.get('User_ID'),
            'transaction_type': 'Online',
            'account_balance': float(persona.user_data.get('Account_Balance', 1000)),
            'previous_fraudulent_activity': 1,  # Mark as suspicious
            'card_type': persona.user_data.get('Card_Type', 'Visa'),
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'fraud_label': 1  # Mark as fraud
        }

    def _ensure_complete_transaction_data(self, transaction: Dict[str, Any], persona: UserPersona) -> Dict[str, Any]:
        """Ensure transaction has all required fields to match the API schema"""
        base_data = persona.user_data

        # Fill in any missing required fields
        complete_transaction = transaction.copy()

        # Ensure all required fields are present with defaults if missing
        if 'transaction_id' not in complete_transaction:
            complete_transaction['transaction_id'] = f"TXN_SIM_{int(asyncio.get_event_loop().time())}_{random.randint(1000, 9999)}"

        if 'user_id' not in complete_transaction:
            complete_transaction['user_id'] = base_data.get('User_ID')

        if 'transaction_type' not in complete_transaction:
            complete_transaction['transaction_type'] = 'Online'

        if 'account_balance' not in complete_transaction:
            complete_transaction['account_balance'] = float(base_data.get('Account_Balance', 1000))

        if 'avg_transaction_amount_7d' not in complete_transaction:
            complete_transaction['avg_transaction_amount_7d'] = float(base_data.get('Avg_Transaction_Amount_7d', 100))

        if 'failed_transaction_count_7d' not in complete_transaction:
            complete_transaction['failed_transaction_count_7d'] = int(base_data.get('Failed_Transaction_Count_7d', 0))

        if 'card_age' not in complete_transaction:
            complete_transaction['card_age'] = int(base_data.get('Card_Age', 365))

        if 'card_type' not in complete_transaction:
            complete_transaction['card_type'] = base_data.get('Card_Type', 'Visa')

        if 'authentication_method' not in complete_transaction:
            complete_transaction['authentication_method'] = 'OTP'

        if 'is_weekend' not in complete_transaction:
            complete_transaction['is_weekend'] = 1 if datetime.now().weekday() >= 5 else 0

        if 'fraud_label' not in complete_transaction:
            complete_transaction['fraud_label'] = 1  # Assume fraud if it's a fraud playbook

        return complete_transaction

    def generate_fallback_analysis(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback analysis when API is rate limited or unavailable"""
        amount = transaction.get('transaction_amount', 0)
        risk_score = transaction.get('risk_score', 0)
        merchant_category = transaction.get('merchant_category', 'Unknown')
        device_type = transaction.get('device_type', 'Unknown')
        location = transaction.get('location', 'Unknown')

        # Simple rule-based analysis for fallback
        fraud_probability = 0.5  # Default medium risk

        # High-risk indicators
        high_risk_indicators = 0

        if amount > 1000:
            high_risk_indicators += 1
        if risk_score > 0.7:
            high_risk_indicators += 1
        if merchant_category in ['Gambling', 'Adult Entertainment', 'Cryptocurrency Exchange']:
            high_risk_indicators += 1
        if device_type in ['Unknown Device', 'IoT Device']:
            high_risk_indicators += 1
        if location in ['International', 'Offshore', 'Unknown']:
            high_risk_indicators += 1

        # Calculate probability based on risk indicators
        if high_risk_indicators >= 3:
            fraud_probability = 0.85
        elif high_risk_indicators >= 2:
            fraud_probability = 0.65
        elif high_risk_indicators >= 1:
            fraud_probability = 0.45
        else:
            fraud_probability = 0.15

        # Determine risk level and verdict
        if fraud_probability >= 0.8:
            risk_level = "CRITICAL"
            verdict = "BLOCK"
        elif fraud_probability >= 0.6:
            risk_level = "HIGH"
            verdict = "REVIEW"
        else:
            risk_level = "LOW"
            verdict = "APPROVE"

        return {
            'transaction_id': transaction.get('transaction_id', 'Unknown'),
            'fraud_probability': round(fraud_probability, 4),
            'risk_level': risk_level,
            'reasoning_steps': [
                f"Fallback analysis: {high_risk_indicators} high-risk indicators detected",
                f"Transaction amount: ${amount:.2f} - {'High' if amount > 1000 else 'Normal'} value",
                f"Merchant category: {merchant_category} - {'High' if merchant_category in ['Gambling', 'Adult Entertainment'] else 'Standard'} risk",
                f"Device type: {device_type} - {'Suspicious' if device_type in ['Unknown Device'] else 'Normal'}",
                f"Location: {location} - {'International' if location in ['International', 'Offshore'] else 'Domestic'}"
            ],
            'red_flags': [f"{high_risk_indicators} high-risk indicators"] if high_risk_indicators > 0 else [],
            'confidence': 0.7,  # Moderate confidence for fallback
            'recommendation': verdict,
            'analysis_mode': 'fallback'
        }

    async def send_transaction_to_api_async(self, transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send transaction to FastAPI for analysis asynchronously with intelligent filtering"""
        try:
            # PART 1: AGGRESSIVE API CALL REDUCTION
            # Only use DeepSeek API for high-value or suspicious transactions
            amount = transaction.get('transaction_amount', 0)
            risk_score = transaction.get('risk_score', 0)

            # Filter: Only expensive API calls for transactions that need it
            needs_api_analysis = (
                amount > 1000 or  # High-value transactions
                risk_score > 0.7 or  # Already suspicious transactions
                transaction.get('merchant_category') in ['Gambling', 'Adult Entertainment', 'Cryptocurrency Exchange']  # High-risk merchants
            )

            if not needs_api_analysis:
                # Use fallback analysis for regular transactions (NO API CALL)
                print(f"📊 Transaction {transaction['transaction_id']}: FALLBACK ANALYSIS (Probability: {risk_score:.3f})")
                return self.generate_fallback_analysis(transaction)

            # CRITICAL ARCHITECTURAL FIX: Use centralized sanitizer as single source of truth
            final_payload = sanitize_payload_for_api(transaction)

            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, json=final_payload, timeout=30.0)
                response.raise_for_status()  # This will raise an exception for 4xx/5xx errors

                # Log the successful result from the server's response
                result = response.json()
                print(f"📊 Transaction {transaction['transaction_id']}: {result.get('final_verdict', 'UNKNOWN')} (Probability: {result.get('fraud_probability', 0):.3f})")
                return result

        except httpx.RequestError as e:
            # Handle rate limiting (429) and other HTTP errors gracefully
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 429:
                    print(f"⚠️ Rate limited for transaction {transaction['transaction_id']} - using fallback")
                    return self.generate_fallback_analysis(transaction)
                elif e.response.status_code >= 500:
                    print(f"⚠️ Server error for transaction {transaction['transaction_id']} - using fallback")
                    return self.generate_fallback_analysis(transaction)

            # This will catch timeouts, connection errors, etc.
            print(f"❌ HTTP Request failed for transaction {transaction['transaction_id']}: {e}")
            return self.generate_fallback_analysis(transaction)
        except Exception as e:
            print(f"❌ Error for transaction {transaction['transaction_id']}: {e}")
            return self.generate_fallback_analysis(transaction)

async def run_simulation_loop():
    """The main non-blocking simulation loop with robust rate limit management and error handling."""
    print("🚀 Real-Time Transaction Simulator is starting...")
    print("📊 Rate Limit Management: Active (15-30 second delays)")
    print("🎯 Smart Filtering: Only high-value/suspicious transactions use DeepSeek API")
    print("🔄 Fallback System: Rule-based analysis when API unavailable")
    print("🛡️ Robust Error Handling: Task will restart automatically on crashes")

    simulator = TransactionSimulator()
    print(f"📊 Loaded {len(simulator.df)} transactions from dataset")
    print(f"👥 Created {len(simulator.personas)} user personas")

    simulator.is_running = True
    api_call_count = 0
    fallback_count = 0

    try:
        while simulator.is_running:
            try:  # ROBUST ERROR HANDLING: Wrap entire loop in try-catch
                # PART 2: DRAMATICALLY SLOWER SIMULATION FOR RATE LIMIT COMPLIANCE
                # Original: 1.0-4.0 seconds (12-30 API calls/hour)
                # Current: 2.0-5.0 seconds (7-18 API calls/hour)
                # NEW: 15.0-30.0 seconds (1.2-2.4 API calls/hour) - WELL UNDER 50/DAY LIMIT
                wait_time = random.uniform(15.0, 30.0)
                print(f"⏱️ Next transaction in {wait_time:.1f} seconds (rate limit management)")
                await asyncio.sleep(wait_time)

                # Select random persona
                persona = random.choice(simulator.personas)

                # 80% chance of normal transaction, 20% chance of fraud (Enhanced for Demo Impact)
                if random.random() < 0.20:  # 20% fraud chance - 4x more engaging demos!
                    print("🔴 FRAUD PLAYBOOK ACTIVATED")
                    transactions = simulator.execute_fraud_playbook(persona)

                    for transaction in transactions:
                        if transaction:
                            result = await simulator.send_transaction_to_api_async(transaction)

                            # Track API vs fallback usage
                            if result and result.get('analysis_mode') == 'fallback':
                                fallback_count += 1
                            else:
                                api_call_count += 1

                            # Update persona state
                            persona.update_location(transaction.get('location', persona.last_location))
                            persona.increment_daily_count()
                else:
                    # Generate normal transaction
                    transaction = simulator.generate_normal_transaction(persona)

                    if transaction:
                        result = await simulator.send_transaction_to_api_async(transaction)

                        # Track API vs fallback usage
                        if result and result.get('analysis_mode') == 'fallback':
                            fallback_count += 1
                        else:
                            api_call_count += 1

                        # Update persona state
                        persona.update_location(transaction.get('location', persona.last_location))

                # PART 3: ROBUST FALLBACK SYSTEM - Log system status
                if api_call_count + fallback_count > 0:
                    api_percentage = (api_call_count / (api_call_count + fallback_count)) * 100
                    print(f"📈 System Status: {api_call_count} API calls, {fallback_count} fallback analyses ({api_percentage:.1f}% API usage)")

            except KeyboardInterrupt:
                print(f"\n🛑 Simulation stopped by user. Final stats: {api_call_count} API calls, {fallback_count} fallback analyses")
                break
            except Exception as e:
                # ROBUST ERROR HANDLING: Catch and recover from any exception
                print(f"💥 CRITICAL ERROR in simulation loop: {e}. Restarting loop in 10 seconds.")
                print(f"� Current stats before crash: {api_call_count} API calls, {fallback_count} fallback analyses")
                logger.error(f"Simulation loop error: {e}")
                # Wait before restarting to prevent rapid crash loops
                await asyncio.sleep(10)
                continue  # Restart the loop instead of crashing

    except Exception as e:
        # Final exception handler for unexpected errors
        print(f"💥 FATAL ERROR: Simulation loop terminated unexpectedly: {e}")
        logger.error(f"Fatal simulation error: {e}")
    finally:
        simulator.is_running = False
        print("✅ Simulation loop terminated gracefully")
