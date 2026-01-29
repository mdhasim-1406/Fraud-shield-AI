"""
DeepSeek V3.1 Fraud Reasoning Engine for Fraud Detection System
Component 3: Advanced AI-powered fraud detection with DeepSeek V3.1
"""
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
import httpx
from openai import OpenAI, AsyncOpenAI
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FraudAnalysisResult:
    """Structured fraud analysis result from DeepSeek"""
    fraud_probability: float
    risk_level: str
    reasoning_steps: List[str]
    red_flags: List[str]
    confidence: float
    recommendation: str

@dataclass
class SelfConsistencyResult:
    """Result from self-consistency analysis"""
    final_probability: float
    agreement_score: float
    individual_results: List[FraudAnalysisResult]
    most_common_recommendation: str

class DeepSeekFraudDetector:
    """DeepSeek V3.1 based fraud detection with advanced reasoning"""

    def __init__(self,
                 api_key: str,
                 base_url: str = "https://openrouter.ai/api/v1",
                 fast_model: str = "deepseek/deepseek-chat-v3.1:free",
                 reasoning_model: str = "deepseek/deepseek-chat-v3.1:free",
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize DeepSeek fraud detector

        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
            fast_model: Fast model for quick analysis
            reasoning_model: Advanced model for deep reasoning
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize OpenAI-compatible clients
        self.fast_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "FraudShieldAI"}
        )

        self.reasoning_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers={"HTTP-Referer": "https://openrouter.ai", "X-Title": "FraudShieldAI"}
        )

        # Few-shot examples for prompt engineering
        self.few_shot_examples = self._create_few_shot_examples()

    def _create_few_shot_examples(self) -> str:
        """Create few-shot examples for fraud analysis"""
        return """
Example 1 - Legitimate Transaction:
Transaction: User USER_123 making a $45.67 purchase at Grocery Store using Visa card. Account balance: $1,234.56. Daily transactions: 2, 7-day average: $52.30, Failed attempts: 0. Authentication: Biometric from Mobile device. Location: New York. Card age: 365 days. Weekend: No.

Reasoning:
1) Transaction amount ($45.67) is reasonable compared to account balance ($1,234.56) - ratio of 3.7%
2) User has consistent transaction pattern with 7-day average of $52.30
3) No failed attempts indicates normal behavior
4) Biometric authentication from mobile device is user's typical method
5) Grocery store is normal merchant category for this user
6) Card has been active for 365 days showing established usage

Fraud Probability: 0.05
Risk Level: LOW
Red Flags: []
Recommendation: APPROVE

Example 2 - Obvious Fraud:
Transaction: User USER_456 making a $2,500.00 purchase at Luxury Jewelry using Amex card. Account balance: $500.00. Daily transactions: 8, 7-day average: $125.00, Failed attempts: 5. Authentication: Password from Unknown device. Location: International. Card age: 1 day. Weekend: Yes. IP Flag: Yes.

Reasoning:
1) Transaction amount ($2,500) is 500% of account balance ($500) - extremely high ratio
2) Sudden spike in daily transactions (8 vs 7-day average suggesting $125)
3) Multiple failed attempts (5) indicate suspicious activity
4) Unknown device with password authentication is unusual
5) International location with IP flag suggests potential compromise
6) Brand new card (1 day old) making high-value purchase

Fraud Probability: 0.95
Risk Level: CRITICAL
Red Flags: ["Extreme amount-to-balance ratio", "Transaction spike", "Failed attempts", "Unknown device", "International location", "New card high value"]
Recommendation: BLOCK

Example 3 - Ambiguous Case:
Transaction: User USER_789 making a $750.00 purchase at Electronics Store using Mastercard. Account balance: $2,000.00. Daily transactions: 1, 7-day average: $150.00, Failed attempts: 1. Authentication: OTP from Tablet device. Location: Different City. Card age: 45 days. Weekend: No.

Reasoning:
1) Transaction amount ($750) vs balance ($2,000) is 37.5% - moderately high but not extreme
2) Only 1 daily transaction vs 7-day average of $150 suggests possible pattern change
3) Single failed attempt could be user error or testing
4) OTP authentication from tablet is reasonable alternative method
5) Different city but not international, could be travel
6) Card is relatively new (45 days) but not brand new

Fraud Probability: 0.45
Risk Level: MEDIUM
Red Flags: ["Moderately high amount", "Pattern deviation", "Recent card"]
Recommendation: REVIEW
"""

    def _create_system_prompt(self) -> str:
        """Create system prompt for fraud analysis"""
        return """You are an expert fraud detection analyst specializing in financial transaction analysis. You have extensive experience in identifying suspicious patterns, behavioral anomalies, and fraud indicators.

Your task is to analyze financial transactions and provide structured fraud assessments following this exact format:

{
  "fraud_probability": <float 0-1>,
  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "reasoning_steps": [
    "Step 1: <specific analysis>",
    "Step 2: <specific analysis>",
    "Step 3: <specific analysis>",
    "Step 4: <specific analysis>",
    "Step 5: <specific analysis>"
  ],
  "red_flags": [<list of specific suspicious indicators>],
  "confidence": <float 0-1>,
  "recommendation": "<APPROVE|REVIEW|BLOCK>"
}

Analysis Guidelines:
- Consider amount-to-balance ratios, transaction patterns, device consistency
- Evaluate authentication methods, location data, and timing patterns
- Assess user behavior consistency and merchant category appropriateness
- Look for velocity patterns, failed attempt spikes, and unusual combinations
- Provide specific, actionable reasoning in each step

Always respond with valid JSON matching the exact schema above."""

    def _create_fast_analysis_prompt(self, transaction_data: str) -> str:
        """Create prompt for fast analysis mode"""
        return f"""{self.few_shot_examples}

Now analyze this transaction using the same step-by-step reasoning process:

{transaction_data}

Provide your analysis in the specified JSON format."""

    def _create_deep_analysis_prompt(self, transaction_data: str) -> str:
        """Create prompt for deep reasoning mode"""
        return f"""{self.few_shot_examples}

Now analyze this transaction requiring complex pattern detection and nuanced analysis:

{transaction_data}

Use extended thinking mode to consider:
- Complex behavioral patterns across multiple dimensions
- Subtle indicators that might be missed in quick analysis
- Historical context and trend analysis
- Multi-factor risk assessment combining all indicators
- Edge cases and unusual but legitimate scenarios

Provide your comprehensive analysis in the specified JSON format."""

    def _parse_llm_response(self, response_content: str) -> FraudAnalysisResult:
        """Parse LLM response into structured result"""
        try:
            # Clean response content
            cleaned_content = response_content.strip()

            # Handle potential markdown formatting
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]

            # Parse JSON response
            data = json.loads(cleaned_content)

            # Validate required fields
            required_fields = ['fraud_probability', 'risk_level', 'reasoning_steps',
                             'red_flags', 'confidence', 'recommendation']

            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing field in LLM response: {field}")

            return FraudAnalysisResult(
                fraud_probability=float(data.get('fraud_probability', 0.5)),
                risk_level=str(data.get('risk_level', 'MEDIUM')),
                reasoning_steps=[str(step) for step in data.get('reasoning_steps', [])],
                red_flags=[str(flag) for flag in data.get('red_flags', [])],
                confidence=float(data.get('confidence', 0.5)),
                recommendation=str(data.get('recommendation', 'REVIEW'))
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response content: {response_content}")
            # Return default result on parsing error
            return FraudAnalysisResult(
                fraud_probability=0.5,
                risk_level='MEDIUM',
                reasoning_steps=['Error parsing LLM response'],
                red_flags=['Response parsing failed'],
                confidence=0.0,
                recommendation='REVIEW'
            )
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return FraudAnalysisResult(
                fraud_probability=0.5,
                risk_level='MEDIUM',
                reasoning_steps=['Error in analysis'],
                red_flags=['Analysis error'],
                confidence=0.0,
                recommendation='REVIEW'
            )

    def analyze_transaction_fast(self, prompt: str) -> FraudAnalysisResult:
        """
        Fast analysis using DeepSeek chat model for sub-2-second decisions

        Args:
            prompt: Formatted transaction analysis prompt

        Returns:
            FraudAnalysisResult: Structured analysis result
        """
        try:
            logger.info("Starting fast fraud analysis...")

            # Create the analysis prompt
            analysis_prompt = self._create_fast_analysis_prompt(prompt)

            # Make API call with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self.fast_client.chat.completions.create(
                        model=self.fast_model,
                        messages=[
                            {"role": "system", "content": self._create_system_prompt()},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        temperature=0.1,  # Low temperature for consistency
                        max_tokens=1000
                    )

                    # Parse response
                    result_content = response.choices[0].message.content
                    result = self._parse_llm_response(result_content)

                    logger.info(f"Fast analysis completed. Fraud probability: {result.fraud_probability:.3f}")
                    return result

                except Exception as e:
                    logger.warning(f"Fast analysis attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise

        except Exception as e:
            logger.error(f"Error in fast analysis: {e}")
            return FraudAnalysisResult(
                fraud_probability=0.5,
                risk_level='MEDIUM',
                reasoning_steps=['Fast analysis failed'],
                red_flags=['Analysis error'],
                confidence=0.0,
                recommendation='REVIEW'
            )

    def analyze_transaction_deep(self, prompt: str) -> FraudAnalysisResult:
        """
        Deep analysis using reasoning model for complex cases

        Args:
            prompt: Formatted transaction analysis prompt

        Returns:
            FraudAnalysisResult: Structured analysis result
        """
        try:
            logger.info("Starting deep fraud analysis...")

            # Create the analysis prompt
            analysis_prompt = self._create_deep_analysis_prompt(prompt)

            # Make API call with retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self.reasoning_client.chat.completions.create(
                        model=self.reasoning_model,
                        messages=[
                            {"role": "system", "content": self._create_system_prompt()},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1500
                    )

                    # Parse response
                    result_content = response.choices[0].message.content
                    result = self._parse_llm_response(result_content)

                    logger.info(f"Deep analysis completed. Fraud probability: {result.fraud_probability:.3f}")
                    return result

                except Exception as e:
                    logger.warning(f"Deep analysis attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Error in deep analysis: {e}")
            return FraudAnalysisResult(
                fraud_probability=0.5,
                risk_level='MEDIUM',
                reasoning_steps=['Deep analysis failed'],
                red_flags=['Analysis error'],
                confidence=0.0,
                recommendation='REVIEW'
            )

    def analyze_with_self_consistency(self, prompt: str, num_analyses: int = 3) -> SelfConsistencyResult:
        """
        Run multiple analyses for self-consistency on high-value transactions

        Args:
            prompt: Formatted transaction analysis prompt
            num_analyses: Number of parallel analyses to run

        Returns:
            SelfConsistencyResult: Consensus-based result
        """
        try:
            logger.info(f"Starting self-consistency analysis with {num_analyses} parallel runs...")

            # Run multiple analyses in parallel (using fast mode for efficiency)
            individual_results = []
            for i in range(num_analyses):
                result = self.analyze_transaction_fast(prompt)
                individual_results.append(result)

            # Calculate agreement score
            probabilities = [r.fraud_probability for r in individual_results]
            recommendations = [r.recommendation for r in individual_results]

            # Calculate probability variance as agreement measure
            prob_variance = np.var(probabilities)
            agreement_score = max(0, 1 - prob_variance * 4)  # Scale variance to 0-1

            # Find most common recommendation
            recommendation_counts = {}
            for rec in recommendations:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

            most_common_recommendation = max(recommendation_counts, key=recommendation_counts.get)

            # Use mean probability for final result
            final_probability = np.mean(probabilities)

            # Create consensus result
            consensus_result = FraudAnalysisResult(
                fraud_probability=final_probability,
                risk_level=self._calculate_consensus_risk_level(individual_results),
                reasoning_steps=self._combine_reasoning_steps(individual_results),
                red_flags=self._combine_red_flags(individual_results),
                confidence=agreement_score,
                recommendation=most_common_recommendation
            )

            logger.info(f"Self-consistency analysis completed. Agreement: {agreement_score:.3f}")
            return SelfConsistencyResult(
                final_probability=final_probability,
                agreement_score=agreement_score,
                individual_results=individual_results,
                most_common_recommendation=most_common_recommendation
            )

        except Exception as e:
            logger.error(f"Error in self-consistency analysis: {e}")
            # Return first result as fallback
            fallback_result = individual_results[0] if individual_results else FraudAnalysisResult(
                fraud_probability=0.5, risk_level='MEDIUM', reasoning_steps=['Self-consistency failed'],
                red_flags=['Analysis error'], confidence=0.0, recommendation='REVIEW'
            )
            return SelfConsistencyResult(
                final_probability=fallback_result.fraud_probability,
                agreement_score=0.0,
                individual_results=individual_results,
                most_common_recommendation=fallback_result.recommendation
            )

    def _calculate_consensus_risk_level(self, results: List[FraudAnalysisResult]) -> str:
        """Calculate consensus risk level from multiple results"""
        risk_levels = [r.risk_level for r in results]
        risk_hierarchy = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}

        # Use highest risk level (most conservative)
        risk_scores = [risk_hierarchy.get(level, 2) for level in risk_levels]
        max_risk_score = max(risk_scores)
        return {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', 4: 'CRITICAL'}.get(max_risk_score, 'MEDIUM')

    def _combine_reasoning_steps(self, results: List[FraudAnalysisResult]) -> List[str]:
        """Combine reasoning steps from multiple analyses"""
        all_steps = []
        for result in results:
            all_steps.extend(result.reasoning_steps)

        # Remove duplicates while preserving order
        seen = set()
        unique_steps = []
        for step in all_steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)

        return unique_steps[:5]  # Limit to 5 steps

    def _combine_red_flags(self, results: List[FraudAnalysisResult]) -> List[str]:
        """Combine red flags from multiple analyses"""
        all_flags = []
        for result in results:
            all_flags.extend(result.red_flags)

        # Count frequency of each flag
        flag_counts = {}
        for flag in all_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

        # Return flags that appear in at least 2 analyses (majority consensus)
        min_analyses = max(2, len(results) // 2)
        consensus_flags = [flag for flag, count in flag_counts.items() if count >= min_analyses]

        return consensus_flags

    def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis function that chooses appropriate analysis mode

        Args:
            transaction_data: Complete transaction data dictionary

        Returns:
            Dict containing analysis results and metadata
        """
        try:
            # Extract transaction prompt data
            transaction_id = transaction_data.get('transaction_id', 'Unknown')
            prompt = transaction_data.get('analysis_prompt', '')
            amount = transaction_data.get('amount', 0)
            is_high_value = transaction_data.get('is_high_value', False)

            logger.info(f"Analyzing transaction {transaction_id} (Amount: ${amount})")

            # Choose analysis mode based on transaction value and complexity
            if is_high_value and amount > 1000:
                # Use self-consistency for high-value transactions
                logger.info("Using self-consistency analysis for high-value transaction")
                consistency_result = self.analyze_with_self_consistency(prompt)

                return {
                    'transaction_id': transaction_id,
                    'fraud_probability': consistency_result.final_probability,
                    'risk_level': self._calculate_consensus_risk_level(consistency_result.individual_results),
                    'reasoning_steps': self._combine_reasoning_steps(consistency_result.individual_results),
                    'red_flags': self._combine_red_flags(consistency_result.individual_results),
                    'confidence': consistency_result.agreement_score,
                    'recommendation': consistency_result.most_common_recommendation,
                    'analysis_mode': 'self_consistency',
                    'individual_results': [
                        {
                            'fraud_probability': r.fraud_probability,
                            'risk_level': r.risk_level,
                            'confidence': r.confidence,
                            'recommendation': r.recommendation
                        }
                        for r in consistency_result.individual_results
                    ]
                }
            else:
                # Use fast analysis for regular transactions
                logger.info("Using fast analysis for regular transaction")
                analysis_result = self.analyze_transaction_fast(prompt)

                return {
                    'transaction_id': transaction_id,
                    'fraud_probability': analysis_result.fraud_probability,
                    'risk_level': analysis_result.risk_level,
                    'reasoning_steps': analysis_result.reasoning_steps,
                    'red_flags': analysis_result.red_flags,
                    'confidence': analysis_result.confidence,
                    'recommendation': analysis_result.recommendation,
                    'analysis_mode': 'fast'
                }

        except Exception as e:
            logger.error(f"Error in main analysis: {e}")
            return {
                'transaction_id': transaction_data.get('transaction_id', 'Unknown'),
                'fraud_probability': 0.5,
                'risk_level': 'MEDIUM',
                'reasoning_steps': ['Analysis error occurred'],
                'red_flags': ['System error'],
                'confidence': 0.0,
                'recommendation': 'REVIEW',
                'analysis_mode': 'error',
                'error': str(e)
            }

# Convenience function for easy fraud detection
def analyze_fraud_with_deepseek(transaction_data: Dict[str, Any],
                               api_key: str,
                               **kwargs) -> Dict[str, Any]:
    """
    Convenience function for fraud analysis with DeepSeek

    Args:
        transaction_data: Transaction data dictionary
        api_key: OpenRouter API key
        **kwargs: Additional arguments for DeepSeekFraudDetector

    Returns:
        Dict containing fraud analysis results
    """
    detector = DeepSeekFraudDetector(api_key=api_key, **kwargs)
    return detector.analyze_transaction(transaction_data)

if __name__ == "__main__":
    # Test the DeepSeek detector (requires API key)
    import os

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("Please set your OpenRouter API key in the .env file")
        exit(1)

    try:
        print("Testing DeepSeek Fraud Detector...")

        # Initialize detector
        detector = DeepSeekFraudDetector(api_key=api_key)

        # Sample transaction data
        sample_transaction = {
            'transaction_id': 'TXN_TEST_DEEPSEEK',
            'analysis_prompt': """
            Transaction ID: TXN_TEST
            User ID: USER_TEST
            Amount: $1500.00
            Type: Online
            Account Balance: $1000.00
            Daily Transaction Count: 5
            7-Day Average Amount: $200.00
            Failed Transactions (7d): 2
            Device Type: Mobile
            Merchant Category: Electronics
            Card Age: 15 days
            Authentication Method: Biometric
            IP Address Flag: No
            Risk Score: 0.85
            Is Weekend: No

            This transaction involves an online transaction for $1500.00 at an electronics merchant.
            The user has $1000.00 in their account and has made 5 transactions today with an average of $200.00 over the past 7 days.
            """,
            'amount': 1500.0,
            'is_high_value': True
        }

        # Test fast analysis
        print("\n1. Testing fast analysis...")
        fast_result = detector.analyze_transaction_fast(sample_transaction['analysis_prompt'])
        print(f"Fast analysis result: {fast_result.fraud_probability:.3f} ({fast_result.risk_level})")

        # Test deep analysis
        print("\n2. Testing deep analysis...")
        deep_result = detector.analyze_transaction_deep(sample_transaction['analysis_prompt'])
        print(f"Deep analysis result: {deep_result.fraud_probability:.3f} ({deep_result.risk_level})")

        # Test self-consistency (for high-value transaction)
        print("\n3. Testing self-consistency analysis...")
        consistency_result = detector.analyze_with_self_consistency(sample_transaction['analysis_prompt'])
        print(f"Self-consistency result: {consistency_result.final_probability:.3f} (agreement: {consistency_result.agreement_score:.3f})")

        # Test main analysis function
        print("\n4. Testing main analysis function...")
        main_result = detector.analyze_transaction(sample_transaction)
        print(f"Main analysis: {main_result['fraud_probability']:.3f} ({main_result['risk_level']}) - Mode: {main_result['analysis_mode']}")

        print("\n✅ DeepSeek Fraud Detector test completed successfully!")

    except Exception as e:
        print(f"❌ Error testing DeepSeek detector: {e}")
        import traceback
        traceback.print_exc()
