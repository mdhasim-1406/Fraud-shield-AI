"""
FastAPI Backend for Fraud Detection System
Component 5: Production-grade REST API for transaction processing and monitoring
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import uvicorn
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd

# Import our modules
from data_loader import TransactionDataLoader, load_transaction_data
from finbert_analyzer import FinBERTAnalyzer, analyze_financial_sentiment
from deepseek_detector import DeepSeekFraudDetector, analyze_fraud_with_deepseek
from rag_system import FraudKnowledgeBase, retrieve_similar_frauds
from config import config
from simulation_manager import run_simulation_loop

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models (will be initialized on startup)
data_loader: Optional[TransactionDataLoader] = None
finbert_analyzer: Optional[FinBERTAnalyzer] = None
deepseek_detector: Optional[DeepSeekFraudDetector] = None
fraud_knowledge_base: Optional[FraudKnowledgeBase] = None

# Global transaction store for live UI updates
from collections import deque
recent_transactions = deque(maxlen=100)  # Store last 100 transactions

# Global variable for background simulator task management
simulator_task: Optional[asyncio.Task] = None

# Pydantic Models
class Transaction(BaseModel):
    """Transaction data model"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_type: str = Field(..., description="Type of transaction")
    timestamp: str = Field(..., description="Transaction timestamp")
    account_balance: float = Field(..., ge=0, description="Account balance")
    device_type: str = Field(..., description="Device used for transaction")
    location: str = Field(..., description="Transaction location")
    merchant_category: str = Field(..., description="Merchant category")
    ip_address_flag: int = Field(..., ge=0, le=1, description="IP address flag")
    previous_fraudulent_activity: int = Field(..., ge=0, le=1, description="Previous fraud flag")
    daily_transaction_count: int = Field(..., ge=0, description="Daily transaction count")
    avg_transaction_amount_7d: float = Field(..., ge=0, description="7-day average amount")
    failed_transaction_count_7d: int = Field(..., ge=0, description="Failed transactions in 7d")
    card_type: str = Field(..., description="Card type")
    card_age: int = Field(..., ge=0, description="Card age in days")
    transaction_distance: float = Field(..., ge=0, description="Transaction distance")
    authentication_method: str = Field(..., description="Authentication method")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag")
    fraud_label: Optional[int] = Field(None, ge=0, le=1, description="Fraud label (for testing)")

class FraudAnalysisResponse(BaseModel):
    """Fraud analysis response model"""
    transaction_id: str
    fraud_probability: float
    risk_level: str
    finbert_sentiment: Dict[str, Any]
    deepseek_reasoning: Dict[str, Any]
    similar_frauds: List[Dict[str, Any]]
    final_verdict: str
    processing_time: float
    analysis_timestamp: str

class BatchAnalysisRequest(BaseModel):
    """Batch analysis request model"""
    transactions: List[Transaction]
    max_concurrent: int = Field(10, ge=1, le=50, description="Max concurrent analyses")

class BatchAnalysisResponse(BaseModel):
    """Batch analysis response model"""
    batch_id: str
    total_transactions: int
    processed_transactions: int
    results: List[FraudAnalysisResponse]
    processing_time: float
    summary_stats: Dict[str, Any]

class SystemStats(BaseModel):
    """System statistics model"""
    total_transactions_processed: int
    fraud_detection_rate: float
    average_processing_time: float
    system_uptime: float
    model_status: Dict[str, str]

class FeedbackRequest(BaseModel):
    """Feedback request model"""
    transaction_id: str
    actual_fraud_label: int
    analyst_notes: Optional[str] = None
    feedback_timestamp: str

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown with robust background task management."""
    global simulator_task, data_loader, finbert_analyzer, deepseek_detector, fraud_knowledge_base
    logger.info("Starting Fraud Detection System...")

    # Startup
    try:
        # Initialize data loader
        data_loader = TransactionDataLoader(config.settings.DATASET_PATH)
        df = data_loader.load_transaction_data()
        logger.info(f"Loaded {len(df)} transactions from dataset")

        # Initialize FinBERT analyzer
        finbert_analyzer = FinBERTAnalyzer()
        logger.info("FinBERT analyzer initialized")

        # Initialize DeepSeek detector (requires API key)
        if config.settings.OPENROUTER_API_KEY:
            deepseek_detector = DeepSeekFraudDetector(
                api_key=config.settings.OPENROUTER_API_KEY,
                base_url=config.settings.OPENROUTER_BASE_URL,
                timeout=config.settings.API_TIMEOUT,
                max_retries=config.settings.MAX_RETRIES
            )
            logger.info("DeepSeek detector initialized")
        else:
            logger.warning("DeepSeek API key not found - fraud detection will be limited")

        # Initialize RAG knowledge base
        fraud_knowledge_base = FraudKnowledgeBase()
        fraud_cases_added = fraud_knowledge_base.build_fraud_knowledge_base(df)
        logger.info(f"Built fraud knowledge base with {fraud_cases_added} cases")

        # Start background simulation task with proper lifecycle management
        logger.info("Starting background transaction simulator as a managed task...")
        simulator_task = asyncio.create_task(run_simulation_loop())
        logger.info("✅ Background simulator task started successfully")

        yield

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise

    # Shutdown - ROBUST TASK MANAGEMENT
    logger.info("🛑 Shutting down Fraud Detection System...")
    if simulator_task:
        logger.info("🛑 Cancelling background simulator task...")
        simulator_task.cancel()
        try:
            # Wait for the task to acknowledge cancellation (with timeout)
            await asyncio.wait_for(simulator_task, timeout=10.0)
            logger.info("✅ Simulator task successfully cancelled")
        except asyncio.CancelledError:
            logger.info("✅ Simulator task cancellation acknowledged")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Simulator task did not cancel within timeout - forcing termination")
        except Exception as e:
            logger.error(f"❌ Error during simulator task cancellation: {e}")

# Create FastAPI application
app = FastAPI(
    title="Fraud Detection System API",
    description="Enterprise-grade AI-powered fraud detection system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to check if models are initialized
def get_models():
    """Dependency to ensure models are available"""
    if not data_loader or not finbert_analyzer:
        raise HTTPException(status_code=503, detail="Models not initialized")
    return {
        "data_loader": data_loader,
        "finbert_analyzer": finbert_analyzer,
        "deepseek_detector": deepseek_detector,
        "fraud_knowledge_base": fraud_knowledge_base
    }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection System API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_loaded": {
            "data_loader": data_loader is not None,
            "finbert_analyzer": finbert_analyzer is not None,
            "deepseek_detector": deepseek_detector is not None,
            "fraud_knowledge_base": fraud_knowledge_base is not None
        }
    }

@app.post("/api/v1/analyze", response_model=FraudAnalysisResponse)
async def analyze_transaction(transaction: Transaction, models=Depends(get_models)):
    """
    Analyze a single transaction for fraud

    This endpoint processes a single transaction through the complete fraud detection pipeline:
    1. FinBERT sentiment analysis
    2. RAG similarity search
    3. DeepSeek reasoning
    4. Weighted scoring and final verdict
    """
    start_time = time.time()

    try:
        logger.info(f"Analyzing transaction {transaction.transaction_id}")

        # Convert Pydantic model to dict for processing
        transaction_dict = transaction.dict()

        # Step 1: FinBERT sentiment analysis
        finbert_result = models["finbert_analyzer"].analyze_transaction_sentiment(transaction_dict)

        # Step 2: Generate transaction embedding for RAG
        transaction_text = models["finbert_analyzer"].build_financial_context(transaction_dict)
        # Use fallback embedding approach for compatibility
        import hashlib
        hash_obj = hashlib.md5(transaction_text.encode())
        hash_bytes = hash_obj.digest()
        transaction_embedding = np.zeros(384)  # Standard embedding dimension
        for i in range(min(len(hash_bytes), 384)):
            transaction_embedding[i] = hash_bytes[i] / 255.0

        # Step 3: Retrieve similar fraud cases
        try:
            similar_cases = models["fraud_knowledge_base"].retrieve_similar_frauds(
                transaction_embedding,
                k=config.settings.TOP_K_SIMILAR
            )
        except Exception as rag_error:
            logger.warning(f"RAG system error: {rag_error}")
            similar_cases = []  # Use empty list as fallback

        # Step 4: Prepare DeepSeek analysis data
        # Create prompt data similar to data_loader format
        transaction_prompt_data = {
            'transaction_id': transaction.transaction_id,
            'analysis_prompt': transaction_text,
            'amount': transaction.transaction_amount,
            'is_high_value': transaction.transaction_amount > config.settings.HIGH_VALUE_THRESHOLD
        }

        # Step 5: DeepSeek fraud analysis
        if models["deepseek_detector"]:
            deepseek_result = models["deepseek_detector"].analyze_transaction(transaction_prompt_data)
        else:
            # Fallback when DeepSeek is not available
            deepseek_result = {
                "fraud_probability": 0.5,
                "risk_level": "MEDIUM",
                "reasoning_steps": ["DeepSeek detector not available"],
                "red_flags": ["API unavailable"],
                "confidence": 0.0,
                "recommendation": "REVIEW",
                "analysis_mode": "fallback"
            }

        # Step 6: Combine results with weighted scoring
        final_probability = (
            finbert_result.get('overall_risk_score', 0.5) * 0.3 +
            deepseek_result.get('fraud_probability', 0.5) * 0.7
        )

        # Determine final verdict
        if final_probability >= 0.8:
            final_verdict = "BLOCK"
        elif final_probability >= 0.6:
            final_verdict = "REVIEW"
        else:
            final_verdict = "APPROVE"

        # Determine risk level
        if final_probability >= 0.8:
            risk_level = "CRITICAL"
        elif final_probability >= 0.6:
            risk_level = "HIGH"
        elif final_probability >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        processing_time = time.time() - start_time

        # Prepare response
        response = FraudAnalysisResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(final_probability, 4),
            risk_level=risk_level,
            finbert_sentiment=finbert_result,
            deepseek_reasoning=deepseek_result,
            similar_frauds=[
                {
                    "transaction_id": case.case.transaction_id,
                    "similarity_score": case.similarity_score,
                    "fraud_indicators": case.case.fraud_indicators,
                    "amount": case.case.amount,
                    "merchant_category": case.case.merchant_category
                }
                for case in similar_cases
            ],
            final_verdict=final_verdict,
            processing_time=round(processing_time, 3),
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        # Store transaction in recent transactions for live UI updates
        transaction_record = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'transaction_id': transaction.transaction_id,
            'amount': transaction.transaction_amount,
            'merchant_category': transaction.merchant_category,
            'risk_score': final_probability,
            'status': 'Analyzed',
            'verdict': final_verdict,
            'analysis_result': response.dict()
        }
        recent_transactions.append(transaction_record)

        logger.info(f"Analysis completed for {transaction.transaction_id} in {processing_time:.3f}s")
        return response

    except Exception as e:
        logger.error(f"Error analyzing transaction {transaction.transaction_id}: {e}")
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/api/v1/batch")
async def analyze_batch(request: BatchAnalysisRequest, models=Depends(get_models)):
    """
    Analyze multiple transactions in batch

    This endpoint processes multiple transactions concurrently for improved throughput.
    """
    start_time = time.time()
    batch_id = f"batch_{int(time.time())}"

    try:
        logger.info(f"Processing batch {batch_id} with {len(request.transactions)} transactions")

        # Limit concurrent processing
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def analyze_single_transaction(trans: Transaction):
            async with semaphore:
                # Create new event loop for this analysis
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, analyze_transaction_sync, trans)

        def analyze_transaction_sync(trans: Transaction) -> FraudAnalysisResponse:
            """Synchronous wrapper for transaction analysis"""
            # This is a simplified version - in production, you'd want proper async handling
            from fastapi.responses import JSONResponse
            import json

            # For now, we'll use a synchronous approach
            # In production, consider using asyncio.to_thread() or proper async clients

            # Convert to dict and analyze
            trans_dict = trans.dict()
            finbert_result = models["finbert_analyzer"].analyze_transaction_sentiment(trans_dict)

            # Generate embedding and get similar cases
            trans_text = models["finbert_analyzer"].build_financial_context(trans_dict)
            # Use fallback embedding approach for compatibility
            import hashlib
            hash_obj = hashlib.md5(trans_text.encode())
            hash_bytes = hash_obj.digest()
            trans_embedding = np.zeros(384)  # Standard embedding dimension
            for i in range(min(len(hash_bytes), 384)):
                trans_embedding[i] = hash_bytes[i] / 255.0
            try:
                similar_cases = models["fraud_knowledge_base"].retrieve_similar_frauds(trans_embedding, 3)
            except Exception as rag_error:
                logger.warning(f"RAG system error in batch: {rag_error}")
                similar_cases = []  # Use empty list as fallback

            # DeepSeek analysis
            prompt_data = {
                'transaction_id': trans.transaction_id,
                'analysis_prompt': trans_text,
                'amount': trans.transaction_amount,
                'is_high_value': trans.transaction_amount > 1000
            }
            if models["deepseek_detector"]:
                deepseek_result = models["deepseek_detector"].analyze_transaction(prompt_data)
            else:
                # Fallback when DeepSeek is not available
                deepseek_result = {
                    "fraud_probability": 0.5,
                    "risk_level": "MEDIUM",
                    "reasoning_steps": ["DeepSeek detector not available"],
                    "red_flags": ["API unavailable"],
                    "confidence": 0.0,
                    "recommendation": "REVIEW",
                    "analysis_mode": "fallback"
                }

            # Combine results
            final_prob = finbert_result.get('overall_risk_score', 0.5) * 0.3 + deepseek_result.get('fraud_probability', 0.5) * 0.7

            if final_prob >= 0.8:
                verdict = "BLOCK"
            elif final_prob >= 0.6:
                verdict = "REVIEW"
            else:
                verdict = "APPROVE"

            return FraudAnalysisResponse(
                transaction_id=trans.transaction_id,
                fraud_probability=round(final_prob, 4),
                risk_level="HIGH" if final_prob >= 0.6 else "MEDIUM" if final_prob >= 0.4 else "LOW",
                finbert_sentiment=finbert_result,
                deepseek_reasoning=deepseek_result,
                similar_frauds=[
                    {
                        "transaction_id": case.case.transaction_id,
                        "similarity_score": case.similarity_score,
                        "fraud_indicators": case.case.fraud_indicators,
                        "amount": case.case.amount,
                        "merchant_category": case.case.merchant_category
                    }
                    for case in similar_cases
                ],
                final_verdict=verdict,
                processing_time=0.0,  # Will be calculated at batch level
                analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        # Process all transactions
        results = []
        for trans in request.transactions:
            try:
                result = await analyze_single_transaction(trans)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing transaction {trans.transaction_id}: {e}")
                # Add error result
                error_result = FraudAnalysisResponse(
                    transaction_id=trans.transaction_id,
                    fraud_probability=0.5,
                    risk_level="MEDIUM",
                    finbert_sentiment={"error": str(e)},
                    deepseek_reasoning={"error": str(e)},
                    similar_frauds=[],
                    final_verdict="REVIEW",
                    processing_time=0.0,
                    analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                results.append(error_result)

        # Calculate summary statistics
        processing_time = time.time() - start_time
        fraud_count = sum(1 for r in results if r.fraud_probability >= 0.6)
        avg_probability = sum(r.fraud_probability for r in results) / len(results)

        summary_stats = {
            "total_analyzed": len(results),
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / len(results),
            "average_fraud_probability": round(avg_probability, 4),
            "processing_rate": len(results) / processing_time
        }

        response = BatchAnalysisResponse(
            batch_id=batch_id,
            total_transactions=len(request.transactions),
            processed_transactions=len(results),
            results=results,
            processing_time=round(processing_time, 3),
            summary_stats=summary_stats
        )

        logger.info(f"Batch {batch_id} completed: {len(results)}/{len(request.transactions)} transactions in {processing_time:.3f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/api/v1/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and performance metrics"""
    try:
        # This would typically come from a metrics database
        # For now, return placeholder data
        stats = SystemStats(
            total_transactions_processed=0,  # Would be tracked in production
            fraud_detection_rate=0.0,  # Would be calculated from historical data
            average_processing_time=0.0,  # Would be calculated from recent analyses
            system_uptime=time.time(),  # Would track actual uptime
            model_status={
                "data_loader": "loaded" if data_loader else "not_loaded",
                "finbert_analyzer": "loaded" if finbert_analyzer else "not_loaded",
                "deepseek_detector": "loaded" if deepseek_detector else "not_loaded",
                "fraud_knowledge_base": "loaded" if fraud_knowledge_base else "not_loaded"
            }
        )

        return stats

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")

@app.get("/api/v1/transaction/{transaction_id}")
async def get_transaction_analysis(transaction_id: str, models=Depends(get_models)):
    """Retrieve analysis for a specific transaction by ID"""
    try:
        # Get transaction from dataset
        transaction_data = models["data_loader"].get_transaction_by_id(transaction_id)

        if not transaction_data:
            raise HTTPException(status_code=404, detail="Transaction not found")

        # If we have DeepSeek detector, run analysis
        if models["deepseek_detector"]:
            try:
                analysis_result = models["deepseek_detector"].analyze_transaction(transaction_data)
                return {
                    "transaction": transaction_data,
                    "analysis": analysis_result,
                    "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.warning(f"DeepSeek analysis failed: {e}")
                return {
                    "transaction": transaction_data,
                    "analysis": {"error": f"DeepSeek analysis failed: {str(e)}"},
                    "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
        else:
            return {
                "transaction": transaction_data,
                "analysis": {"error": "DeepSeek detector not available"},
                "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transaction {transaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve transaction")

@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit analyst feedback for continuous learning"""
    try:
        logger.info(f"Received feedback for transaction {feedback.transaction_id}")

        # In a production system, this would:
        # 1. Store feedback in database
        # 2. Update model weights if using fine-tuning
        # 3. Improve RAG knowledge base
        # 4. Track analyst accuracy

        # For now, just log the feedback
        feedback_data = {
            "transaction_id": feedback.transaction_id,
            "actual_fraud_label": feedback.actual_fraud_label,
            "analyst_notes": feedback.analyst_notes,
            "timestamp": feedback.feedback_timestamp
        }

        logger.info(f"Feedback logged: {feedback_data}")

        return {
            "message": "Feedback submitted successfully",
            "feedback_id": f"feedback_{int(time.time())}",
            "status": "logged"
        }

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@app.get("/api/v1/knowledge-base/stats")
async def get_knowledge_base_stats(models=Depends(get_models)):
    """Get RAG knowledge base statistics"""
    try:
        if not models["fraud_knowledge_base"]:
            raise HTTPException(status_code=503, detail="Knowledge base not available")

        stats = models["fraud_knowledge_base"].get_knowledge_base_stats()
        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve knowledge base statistics")

@app.get("/api/v1/recent-transactions")
async def get_recent_transactions():
    """Get recent transactions for live UI updates"""
    try:
        # Convert deque to list and return as JSON
        transactions_list = list(recent_transactions)

        # Sort by timestamp (newest first)
        transactions_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return {
            "transactions": transactions_list,
            "total_count": len(transactions_list),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Error getting recent transactions: {e}")
        return {
            "transactions": [],
            "total_count": 0,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host=config.settings.HOST,
        port=config.settings.PORT,
        reload=config.settings.DEBUG,
        log_level=config.settings.LOG_LEVEL.lower()
    )
