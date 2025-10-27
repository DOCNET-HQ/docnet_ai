"""
Health check routes
"""
from fastapi import APIRouter
from datetime import datetime
from app.core.config import settings
from app.core.model_registry import ModelRegistry

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check - verifies service is ready to accept requests"""
    try:
        model_registry = ModelRegistry()
        available_models = model_registry.get_available_models()

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "models_registered": len(available_models),
            "models_loaded": sum(1 for m in available_models if m["loaded"])
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/live")
async def liveness_check():
    """Liveness check - verifies service is running"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
