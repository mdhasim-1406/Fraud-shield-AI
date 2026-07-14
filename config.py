"""
Configuration management for Fraud Detection System — Local Only
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings — all local, no external API dependencies"""

    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma2:2b")

    # API Settings
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "60"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/fraud_detection.log")

    # Data Configuration
    DATASET_PATH: str = "synthetic_fraud_dataset.csv"
    LOGS_DIR: str = "logs"

    # Model Parameters
    FINBERT_MODEL: str = "ProsusAI/finbert"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "models/fraud_index.faiss"

    # Thresholds
    HIGH_VALUE_THRESHOLD: float = 1000.0
    SELF_CONSISTENCY_THRESHOLD: float = 0.8
    MIN_CONFIDENCE_THRESHOLD: float = 0.6

    # RAG Configuration
    TOP_K_SIMILAR: int = 3
    EMBEDDING_DIMENSION: int = 384


class Config:
    """Global configuration instance"""
    settings = Settings()

    @classmethod
    def get_settings(cls) -> Settings:
        """Get application settings"""
        return cls.settings

    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration.
        No API keys required — purely local operation.
        """
        return True


# Global configuration instance
config = Config()
