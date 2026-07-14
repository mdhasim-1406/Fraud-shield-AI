"""
Ollama Local Fraud Reasoning Engine for Fraud Detection System
Component 3: Local AI-powered fraud detection using Ollama (replaces DeepSeek/OpenRouter)

Uses Ollama's OpenAI-compatible API endpoint for fully local inference.
No external API keys required. No data leaves your machine.
"""
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FraudAnalysisResult:
    """Structured fraud analysis result from local LLM"""
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


class OllamaFraudDetector:
    """
    Local fraud detection using Ollama-hosted LLM.
    Fully offline — no data sent to external APIs.
    Uses OpenAI-compatible endpoint provided by Ollama.
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434/v1",
                 fast_model: str = "gemma2:2b",
                 reasoning_model: str = "gemma2:2b",
                 timeout: int = 60,
                 max_retries: int = 2):
        """
        Initialize local fraud detector.

        Args:
            base_url: Ollama OpenAI-compatible endpoint
            fast_model: Model for quick analysis (default: gemma2:2b)
            reasoning_model: Model for deep reasoning (default: gemma2:2b)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url
        self.fast_model = fast_model
        self.reasoning_model = reasoning_model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize OpenAI-compatible client pointing to Ollama
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama ignores the key but requires the field
            timeout=timeout,
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
  "reasoning_steps": ["Step 1: ...", "Step 2: ...", ...],
  "red_flags": ["flag1", "flag2", ...],
  "confidence": <float 0-1>,
  "recommendation": "<APPROVE|REVIEW|BLOCK>"
}

IMPORTANT RULES:
1. Output ONLY valid JSON — no markdown, no code fences, no extra text
2. fraud_probability must be a float between 0 and 1
3. reasoning_steps must be a list of 3-5 clear reasoning steps
4. red_flags should list specific suspicious indicators (empty list if none)
5. confidence reflects how certain you are in your assessment
6. recommendation must be APPROVE (low risk), REVIEW (medium/uncertain), or BLOCK (high risk)
7. Keep your analysis factual and data-driven
8. Consider all transaction features holistically"""

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response, handling various formats.

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed dictionary or None if parsing fails
        """
        try:
            # Try direct JSON parse first
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON block within the response
        try:
            # Remove markdown code fences if present
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # Find the first { after ```
                start = cleaned.find("{")
                if start != -1:
                    cleaned = cleaned[start:]
                # Remove trailing ```
                end = cleaned.rfind("}")
                if end != -1:
                    cleaned = cleaned[:end + 1]
                return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        logger.error(f"Failed to parse LLM response as JSON: {response_text[:200]}")
        return None

    def _call_llm(self, messages: List[Dict[str, str]], model: str = None) -> Optional[str]:
        """
        Call Ollama LLM with retry logic.

        Args:
            messages: Chat messages
            model: Model override (defaults to fast_model)

        Returns:
            Response text or None on failure
        """
        model = model or self.fast_model
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            try:
                logger.debug(f"LLM call attempt {attempt} with model {model}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_tokens=1024,
                )
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt} failed: {e}")
                if attempt < self.max_retries + 1:
                    time.sleep(1 * attempt)  # Simple backoff

        logger.error(f"All LLM call attempts failed: {last_error}")
        return None

    def analyze_with_llm(self, prompt: str, model: str = None) -> FraudAnalysisResult:
        """
        Analyze a transaction using the local LLM.

        Args:
            prompt: Transaction analysis prompt
            model: Model to use (defaults to fast_model)

        Returns:
            FraudAnalysisResult with structured analysis
        """
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": f"Here are examples of fraud analysis:\n\n{self.few_shot_examples}\n\nNow analyze this transaction:\n\n{prompt}\n\nRespond with JSON only."}
        ]

        response = self._call_llm(messages, model)

        if not response:
            return FraudAnalysisResult(
                fraud_probability=0.5,
                risk_level='MEDIUM',
                reasoning_steps=['LLM call failed — using default fallback'],
                red_flags=['Analysis unavailable'],
                confidence=0.0,
                recommendation='REVIEW'
            )

        parsed = self._parse_json_response(response)

        if not parsed:
            return FraudAnalysisResult(
                fraud_probability=0.5,
                risk_level='MEDIUM',
                reasoning_steps=['Response parsing failed — using default fallback'],
                red_flags=['Parse error'],
                confidence=0.0,
                recommendation='REVIEW'
            )

        return FraudAnalysisResult(
            fraud_probability=parsed.get('fraud_probability', 0.5),
            risk_level=parsed.get('risk_level', 'MEDIUM'),
            reasoning_steps=parsed.get('reasoning_steps', ['No reasoning provided']),
            red_flags=parsed.get('red_flags', []),
            confidence=parsed.get('confidence', 0.0),
            recommendation=parsed.get('recommendation', 'REVIEW')
        )

    def analyze_with_self_consistency(self, prompt: str, num_samples: int = 3) -> SelfConsistencyResult:
        """
        Run multiple analyses and combine results for higher confidence.
        Useful for high-value transactions where accuracy is critical.

        Args:
            prompt: Transaction analysis prompt
            num_samples: Number of analysis runs (default: 3)

        Returns:
            SelfConsistencyResult with aggregated analysis
        """
        individual_results = []

        logger.info(f"Running self-consistency analysis with {num_samples} samples")

        for i in range(num_samples):
            try:
                # Use slightly different prompts for diversity
                if i == 0:
                    result = self.analyze_with_llm(prompt, self.reasoning_model)
                elif i == 1:
                    result = self.analyze_with_llm(
                        prompt + "\n\nFocus particularly on behavioral patterns and anomalies.",
                        self.reasoning_model
                    )
                else:
                    result = self.analyze_with_llm(
                        prompt + "\n\nFocus particularly on amount-based and location-based indicators.",
                        self.reasoning_model
                    )

                individual_results.append(result)
                logger.debug(f"Self-consistency sample {i+1}: probability={result.fraud_probability:.3f}")

            except Exception as e:
                logger.error(f"Error in self-consistency sample {i+1}: {e}")
                continue

        if not individual_results:
            return SelfConsistencyResult(
                final_probability=0.5,
                agreement_score=0.0,
                individual_results=[],
                most_common_recommendation='REVIEW'
            )

        # Aggregate results
        probabilities = [r.fraud_probability for r in individual_results]
        final_probability = float(np.mean(probabilities))

        recommendations = [r.recommendation for r in individual_results]
        from collections import Counter
        rec_counter = Counter(recommendations)
        most_common_recommendation = rec_counter.most_common(1)[0][0]

        # Calculate agreement (lower std = higher agreement)
        std_prob = float(np.std(probabilities)) if len(probabilities) > 1 else 0.0
        agreement_score = max(0.0, 1.0 - std_prob * 2)

        logger.info(f"Self-consistency completed. Agreement: {agreement_score:.3f}")
        return SelfConsistencyResult(
            final_probability=final_probability,
            agreement_score=agreement_score,
            individual_results=individual_results,
            most_common_recommendation=most_common_recommendation
        )

    def _calculate_consensus_risk_level(self, results: List[FraudAnalysisResult]) -> str:
        """Calculate consensus risk level from multiple results"""
        risk_levels = [r.risk_level for r in results]
        risk_hierarchy = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        risk_scores = [risk_hierarchy.get(level, 2) for level in risk_levels]
        max_risk_score = max(risk_scores)
        return {1: 'LOW', 2: 'MEDIUM', 3: 'HIGH', 4: 'CRITICAL'}.get(max_risk_score, 'MEDIUM')

    def _combine_reasoning_steps(self, results: List[FraudAnalysisResult]) -> List[str]:
        """Combine reasoning steps from multiple analyses"""
        all_steps = []
        for result in results:
            all_steps.extend(result.reasoning_steps)

        seen = set()
        unique_steps = []
        for step in all_steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)

        return unique_steps[:5]

    def _combine_red_flags(self, results: List[FraudAnalysisResult]) -> List[str]:
        """Combine red flags from multiple analyses"""
        all_flags = []
        for result in results:
            all_flags.extend(result.red_flags)

        flag_counts = {}
        for flag in all_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

        min_analyses = max(2, len(results) // 2)
        consensus_flags = [flag for flag, count in flag_counts.items() if count >= min_analyses]
        return consensus_flags

    def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis function that chooses appropriate analysis mode.

        Args:
            transaction_data: Complete transaction data dictionary

        Returns:
            Dict containing analysis results and metadata
        """
        try:
            transaction_id = transaction_data.get('transaction_id', 'Unknown')
            prompt = transaction_data.get('analysis_prompt', '')
            amount = transaction_data.get('amount', 0)
            is_high_value = transaction_data.get('is_high_value', False)

            logger.info(f"Analyzing transaction {transaction_id} (Amount: ${amount})")

            if is_high_value and amount > 1000:
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
                    'analysis_mode': 'self_consistency'
                }
            else:
                logger.info("Using single-pass analysis")
                result = self.analyze_with_llm(prompt)

                return {
                    'transaction_id': transaction_id,
                    'fraud_probability': result.fraud_probability,
                    'risk_level': result.risk_level,
                    'reasoning_steps': result.reasoning_steps,
                    'red_flags': result.red_flags,
                    'confidence': result.confidence,
                    'recommendation': result.recommendation,
                    'analysis_mode': 'fast'
                }

        except Exception as e:
            logger.error(f"Error in fraud analysis: {e}")
            return {
                'transaction_id': transaction_data.get('transaction_id', 'Unknown'),
                'fraud_probability': 0.5,
                'risk_level': 'MEDIUM',
                'reasoning_steps': [f'Analysis error: {str(e)}'],
                'red_flags': ['Analysis error'],
                'confidence': 0.0,
                'recommendation': 'REVIEW',
                'analysis_mode': 'error'
            }


# Convenience function for easy fraud analysis
def analyze_fraud_locally(transaction_data: Dict[str, Any],
                           base_url: str = "http://localhost:11434/v1",
                           model: str = "gemma2:2b") -> Dict[str, Any]:
    """
    Convenience function for local fraud analysis.

    Args:
        transaction_data: Transaction data dictionary
        base_url: Ollama endpoint URL
        model: Model name to use

    Returns:
        Dict containing fraud analysis results
    """
    detector = OllamaFraudDetector(base_url=base_url, fast_model=model, reasoning_model=model)
    return detector.analyze_transaction(transaction_data)


if __name__ == "__main__":
    # Test the local fraud detector
    try:
        import numpy as np
        from collections import Counter

        print("Testing Ollama Fraud Detector...")
        print(f"Make sure Ollama is running with: ollama serve")
        print(f"Pull the model: ollama pull gemma2:2b")

        detector = OllamaFraudDetector()

        # Test prompt
        test_prompt = """
Transaction: User USER_123 making a $1,500.00 purchase at Electronics Store using Visa card. Account balance: $2,000.00. Daily transactions: 1, 7-day average: $250.00, Failed attempts: 0. Authentication: Biometric from Mobile device. Location: New York. Card age: 365 days. Weekend: No.
"""

        print("\n1. Testing single analysis...")
        result = detector.analyze_with_llm(test_prompt)
        print(f"Fraud Probability: {result.fraud_probability:.3f}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Red Flags: {result.red_flags}")

        print("\n✅ Ollama Fraud Detector test completed!")

    except Exception as e:
        print(f"Error testing Ollama detector: {e}")
        import traceback
        traceback.print_exc()
