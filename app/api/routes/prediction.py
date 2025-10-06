"""
Prediction routes
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import logging

from app.schemas.prediction import (
    JSONPredictionRequest,
    JSONPredictionResponse,
    ImagePredictionResponse,
    ModelsListResponse,
    ModelInfo
)
from app.services.predictor import PredictionService
from app.core.model_registry import ModelRegistry
from app.core.exceptions import ModelNotFoundError, PredictionError, InvalidInputError

logger = logging.getLogger(__name__)
router = APIRouter()


def get_predictor_service() -> PredictionService:
    """Dependency to get predictor service"""
    return PredictionService()


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List all available models"""
    try:
        model_registry = ModelRegistry()
        models = model_registry.get_available_models()
        
        return ModelsListResponse(
            success=True,
            models=[ModelInfo(**model) for model in models],
            total=len(models)
        )
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/json", response_model=JSONPredictionResponse)
async def predict_with_json(
    request: JSONPredictionRequest,
    predictor: PredictionService = Depends(get_predictor_service)
):
    """Make prediction with JSON input"""
    try:
        result = predictor.predict_json(
            model_name=request.model_name,
            input_data=request.input_data,
            return_probabilities=request.return_probabilities
        )
        
        return JSONPredictionResponse(
            success=True,
            result=result,
            probabilities=result.get("probabilities"),
            metadata=result.get("metadata")
        )
    
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in JSON prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/image", response_model=ImagePredictionResponse)
async def predict_with_image(
    model_name: str = Form(..., description="Name of the model to use"),
    image: UploadFile = File(..., description="Image file to analyze"),
    return_gradcam: bool = Form(False, description="Return GradCAM visualization"),
    confidence_threshold: Optional[float] = Form(None, description="Minimum confidence threshold"),
    predictor: PredictionService = Depends(get_predictor_service)
):
    """Make prediction with image input"""
    try:
        # Read image file
        image_bytes = await image.read()
        
        # Make prediction
        result = predictor.predict_image(
            model_name=model_name,
            image_bytes=image_bytes,
            return_gradcam=return_gradcam,
            confidence_threshold=confidence_threshold
        )
        
        return ImagePredictionResponse(
            success=True,
            result=result,
            probabilities=result.get("probabilities"),
            gradcam_image=result.get("gradcam_image"),
            metadata=result.get("metadata")
        )
    
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in image prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    try:
        model_registry = ModelRegistry()
        config = model_registry.get_model_config(model_name)
        
        return {
            "success": True,
            "model": {
                "name": model_name,
                "description": config.get("description", ""),
                "version": config.get("version", "1.0"),
                "input_type": config.get("input_type", "unknown"),
                "framework": config.get("framework", "unknown"),
                "classes": config.get("classes", []),
                "input_shape": config.get("input_shape", []),
                "supports_gradcam": config.get("supports_gradcam", True)
            }
        }
    
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))