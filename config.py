"""
Configuration management for Fraud Detection System
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings with environment variable support"""

    # API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Model Configuration
    FAST_MODEL: str = os.getenv("FAST_MODEL", "deepseek/deepseek-chat-v3.1:free")
    REASONING_MODEL: str = os.getenv("REASONING_MODEL", "deepseek/deepseek-chat-v3.1:free")

    # API Settings
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
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
        """Validate that all required configuration is present"""
        required_vars = [
            "OPENROUTER_API_KEY"
        ]

        missing_vars = []
        for var in required_vars:
            if not getattr(cls.settings, var):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        return True

# Global configuration instance
config = Config()
