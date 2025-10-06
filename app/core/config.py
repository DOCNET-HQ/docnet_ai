"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application Info
    PROJECT_NAME: str = "Medical AI Models Microservice"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]  # Update with your Django app URL in production
    
    # Model Settings
    MODEL_REGISTRY_PATH: str = "models/model_registry.json"
    MODELS_BASE_PATH: str = "models"
    MAX_IMAGE_SIZE_MB: int = 10
    SUPPORTED_IMAGE_FORMATS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    # AWS Settings
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = ""  # Optional: for loading models from S3
    
    # Security
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: str = ""  # Set in production via environment variable
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    
    # Prediction Settings
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    ENABLE_GRADCAM: bool = True
    GRADCAM_LAYER_NAME: str = "auto"  # or specify layer name
    
    # Performance
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 300  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
