"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime


class JSONPredictionRequest(BaseModel):
    """Request schema for JSON-based predictions"""
    model_name: str = Field(..., description="Name of the model to use")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    return_probabilities: bool = Field(default=True, description="Return probability scores")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "malaria_classifier",
                "input_data": {
                    "temperature": 38.5,
                    "symptom_days": 3,
                    "region": "tropical"
                },
                "return_probabilities": True
            }
        }


class ImagePredictionRequest(BaseModel):
    """Request schema for image-based predictions"""
    model_name: str = Field(..., description="Name of the model to use")
    return_gradcam: bool = Field(default=False, description="Return GradCAM visualization")
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    
    @validator('confidence_threshold')
    def validate_threshold(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v


class PredictionResult(BaseModel):
    """Base prediction result"""
    prediction: Any = Field(..., description="Model prediction")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    model_name: str = Field(..., description="Name of model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class JSONPredictionResponse(BaseModel):
    """Response schema for JSON-based predictions"""
    success: bool = True
    result: PredictionResult
    probabilities: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": {
                    "prediction": "positive",
                    "confidence": 0.92,
                    "model_name": "malaria_classifier",
                    "timestamp": "2025-10-06T10:30:00"
                },
                "probabilities": {
                    "positive": 0.92,
                    "negative": 0.08
                },
                "metadata": {
                    "processing_time_ms": 145
                }
            }
        }


class ImagePredictionResponse(BaseModel):
    """Response schema for image-based predictions"""
    success: bool = True
    result: PredictionResult
    probabilities: Optional[Dict[str, float]] = None
    gradcam_image: Optional[str] = Field(None, description="Base64 encoded GradCAM image")
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": {
                    "prediction": "glioma",
                    "confidence": 0.87,
                    "model_name": "brain_tumor_classifier",
                    "timestamp": "2025-10-06T10:30:00"
                },
                "probabilities": {
                    "glioma": 0.87,
                    "meningioma": 0.08,
                    "pituitary": 0.05
                },
                "gradcam_image": "base64_encoded_image_data",
                "metadata": {
                    "processing_time_ms": 523,
                    "image_size": [224, 224]
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information schema"""
    name: str
    description: str
    input_type: str
    version: str
    loaded: bool


class ModelsListResponse(BaseModel):
    """Response schema for listing models"""
    success: bool = True
    models: List[ModelInfo]
    total: int


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
