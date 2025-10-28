"""
Prediction service for handling model predictions
"""
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from app.core.model_registry import ModelRegistry
from app.core.exceptions import PredictionError, InvalidInputError
from app.services.image_processor import ImageProcessor
from app.core.config import settings

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions with registered models"""

    def __init__(self):
        self.model_registry = ModelRegistry()
        self.image_processor = ImageProcessor()

    def predict_json(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Make prediction with JSON input"""
        start_time = time.time()

        try:
            # Get model and config
            model = self.model_registry.get_model(model_name)
            config = self.model_registry.get_model_config(model_name)

            # Validate input type
            if config.get("input_type") != "json":
                raise InvalidInputError(
                    f"Model '{model_name}' does not accept JSON input. "
                    f"Expected input type: {config.get('input_type')}"
                )

            # Prepare input based on model requirements
            input_array = self._prepare_json_input(input_data, config)

            # Make prediction
            prediction = model.predict(input_array)

            # Process results
            result = self._process_prediction_results(
                prediction,
                config,
                return_probabilities
            )

            # Add metadata
            processing_time = (time.time() - start_time) * 1000
            result["metadata"] = {
                "processing_time_ms": round(processing_time, 2),
                "model_version": config.get("version", "1.0")
            }

            logger.info(
                f"Prediction completed for {model_name}",
                extra={"model_name": model_name, "duration_ms": processing_time}
            )

            return result

        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {str(e)}")
            raise PredictionError(str(e), model_name)

    def predict_image(
        self,
        model_name: str,
        image_bytes: bytes,
        return_gradcam: bool = False,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make prediction with image input"""
        start_time = time.time()

        try:
            # Get model and config
            model = self.model_registry.get_model(model_name)
            config = self.model_registry.get_model_config(model_name)

            # Validate input type
            if config.get("input_type") != "image":
                raise InvalidInputError(
                    f"Model '{model_name}' does not accept image input. "
                    f"Expected input type: {config.get('input_type')}"
                )

            # Load and preprocess image
            original_img = self.image_processor.load_image(image_bytes)

            target_size = tuple(config.get("input_shape", [224, 224])[:2])
            img_array = self.image_processor.preprocess_image(
                original_img,
                target_size=target_size,
                normalize=config.get("normalize", True)
            )

            # Make prediction
            prediction = model.predict(img_array)

            # Process results
            result = self._process_prediction_results(
                prediction,
                config,
                return_probabilities=True
            )

            # Apply confidence threshold if specified
            threshold = confidence_threshold or config.get(
                "confidence_threshold",
                settings.DEFAULT_CONFIDENCE_THRESHOLD
            )

            if result["confidence"] < threshold:
                result["warning"] = f"Confidence below threshold ({threshold})"

            # Generate GradCAM if requested and supported
            gradcam_image = None
            if return_gradcam and settings.ENABLE_GRADCAM:
                if config.get("supports_gradcam", True):
                    try:
                        pred_index = np.argmax(prediction[0])
                        heatmap = self.image_processor.generate_gradcam(
                            model,
                            img_array,
                            layer_name=config.get("gradcam_layer", "auto"),
                            pred_index=pred_index
                        )
                        gradcam_image = self.image_processor.overlay_gradcam(
                            original_img,
                            heatmap
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate GradCAM: {str(e)}")
                        result["gradcam_warning"] = "GradCAM generation failed"

            # Add metadata
            processing_time = (time.time() - start_time) * 1000
            result["metadata"] = {
                "processing_time_ms": round(processing_time, 2),
                "model_version": config.get("version", "1.0"),
                "input_size": list(target_size),
                "original_image_size": list(original_img.size)
            }

            if gradcam_image:
                result["gradcam_image"] = gradcam_image

            logger.info(
                f"Image prediction completed for {model_name}",
                extra={"model_name": model_name, "duration_ms": processing_time}
            )

            if model_name:
                result["model_name"] = model_name

            return result

        except Exception as e:
            logger.error(f"Image prediction error for {model_name}: {str(e)}")
            raise PredictionError(str(e), model_name)

    def _prepare_json_input(
        self,
        input_data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Prepare JSON input for model"""
        try:
            features = config.get("features", [])

            if not features:
                # If no features specified, use input as-is
                values = list(input_data.values())
            else:
                # Extract features in order
                values = []
                for feature in features:
                    if feature not in input_data:
                        raise InvalidInputError(f"Missing required feature: {feature}")
                    values.append(input_data[feature])

            # Convert to numpy array
            input_array = np.array([values], dtype=np.float32)

            return input_array

        except Exception as e:
            raise InvalidInputError(f"Error preparing input: {str(e)}")

    def _process_prediction_results(
        self,
        prediction: np.ndarray,
        config: Dict[str, Any],
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Process model prediction results"""
        try:
            classes = config.get("classes", [])

            # Handle different output formats
            if len(prediction.shape) == 2 and prediction.shape[1] > 1:
                # Multi-class classification
                probabilities = prediction[0]
                pred_index = np.argmax(probabilities)
                confidence = float(probabilities[pred_index])

                if classes:
                    prediction_label = classes[pred_index]
                else:
                    prediction_label = int(pred_index)

                result = {
                    "prediction": prediction_label,
                    "confidence": confidence
                }

                if return_probabilities and classes:
                    result["probabilities"] = {
                        class_name: float(prob)
                        for class_name, prob in zip(classes, probabilities)
                    }

            elif len(prediction.shape) == 2 and prediction.shape[1] == 1:
                # Binary classification or regression
                value = float(prediction[0][0])

                if config.get("task_type") == "classification":
                    # Binary classification
                    prediction_label = classes[1] if value > 0.5 else classes[0] if classes else int(value > 0.5)
                    confidence = value if value > 0.5 else 1 - value

                    result = {
                        "prediction": prediction_label,
                        "confidence": confidence
                    }

                    if return_probabilities and classes:
                        result["probabilities"] = {
                            classes[0]: 1 - value,
                            classes[1]: value
                        }
                else:
                    # Regression
                    result = {
                        "prediction": value,
                        "confidence": 1.0  # Regression doesn't have confidence
                    }
            else:
                # Generic output
                result = {
                    "prediction": prediction.tolist(),
                    "confidence": 1.0
                }

            return result

        except Exception as e:
            raise PredictionError(f"Error processing prediction results: {str(e)}")
