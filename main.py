"""
FastAPI Medical AI Models Microservice
Main application entry point
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from mangum import Mangum
import logging
from typing import Optional
import io

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.routes import health, prediction
from app.core.model_registry import ModelRegistry
from app.core.exceptions import ModelNotFoundError, PredictionError

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Medical AI Models Prediction Microservice",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(prediction.router, prefix="/api/v1/predict", tags=["Prediction"])

# Initialize model registry on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models on application startup"""
    try:
        logger.info("Starting Medical AI Microservice...")
        model_registry = ModelRegistry()
        model_registry.load_models()
        logger.info(f"Successfully loaded {len(model_registry.get_available_models())} models")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down Medical AI Microservice...")

# Global exception handler
@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": str(exc),
            "error_type": "MODEL_NOT_FOUND"
        }
    )

@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "error_type": "PREDICTION_ERROR"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred",
            "error_type": "INTERNAL_ERROR"
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs" if settings.ENVIRONMENT != "production" else "disabled in production"
    }

# AWS Lambda handler
handler = Mangum(app, lifespan="off")
