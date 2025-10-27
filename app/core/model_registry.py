"""
Model Registry - Manages loading and accessing AI models
"""
import json
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import torch

from app.core.config import settings
from app.core.exceptions import ModelNotFoundError, ModelLoadError, ModelRegistryError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton class to manage model loading and access"""

    _instance = None
    _models: Dict[str, Any] = {}
    _model_configs: Dict[str, Dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_models(self):
        """Load all models defined in the registry"""
        registry_path = Path(settings.MODEL_REGISTRY_PATH)

        if not registry_path.exists():
            raise ModelRegistryError(f"Model registry file not found at {registry_path}")

        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)

            models_config = registry_data.get("models", [])

            for model_config in models_config:
                try:
                    model_name = model_config["name"]
                    self._model_configs[model_name] = model_config

                    # Lazy loading - models will be loaded on first use
                    if model_config.get("preload", False):
                        self._load_model(model_name)

                    logger.info(f"Registered model: {model_name}")

                except Exception as e:
                    logger.error(f"Error registering model {model_config.get('name', 'unknown')}: {str(e)}")
                    continue

            logger.info(f"Model registry loaded with {len(self._model_configs)} models")

        except json.JSONDecodeError as e:
            raise ModelRegistryError(f"Invalid JSON in model registry: {str(e)}")
        except Exception as e:
            raise ModelRegistryError(f"Error loading model registry: {str(e)}")

    def _load_tensorflow_model(self, model_path: Path, model_name: str) -> Any:
        """Load TensorFlow/Keras model"""

        try:
            logger.info(f"Loading TensorFlow model: {model_name}")

            # TF 2.16+ handles both Keras 2 and Keras 3 model formats
            model = tf.keras.models.load_model(str(model_path), compile=False)

            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            logger.info(f"✓ Model loaded: Input {model.input_shape}, Output {model.output_shape}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError(model_name, str(e))

    def _load_model(self, model_name: str) -> Any:
        """Load a specific model"""
        if model_name in self._models:
            return self._models[model_name]

        if model_name not in self._model_configs:
            raise ModelNotFoundError(model_name)

        config = self._model_configs[model_name]
        model_path = Path(settings.MODELS_BASE_PATH) / config["model_path"]

        if not model_path.exists():
            raise ModelLoadError(model_name, f"Model file not found at {model_path}")

        try:
            framework = config.get("framework", "tensorflow").lower()

            if framework == "tensorflow" or framework == "keras":
                model = self._load_tensorflow_model(model_path, model_name)

            elif framework == "pytorch":
                logger.info(f"Loading PyTorch model: {model_name}")
                model = torch.load(str(model_path))
                model.eval()

            elif framework == "sklearn":
                logger.info(f"Loading scikit-learn model: {model_name}")
                import joblib
                model = joblib.load(str(model_path))

            else:
                raise ModelLoadError(model_name, f"Unsupported framework: {framework}")

            self._models[model_name] = model
            logger.info(f"✓ Successfully loaded and cached model: {model_name}")

            return model

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model {model_name}: {str(e)}", exc_info=True)
            raise ModelLoadError(model_name, str(e))

    def get_model(self, model_name: str) -> Any:
        """Get a model by name, loading it if necessary"""
        try:
            if model_name not in self._models:
                logger.info(f"Model '{model_name}' not in cache, loading...")
                self._load_model(model_name)
            return self._models[model_name]
        except Exception as e:
            logger.error(f"Error getting model '{model_name}': {str(e)}")
            raise

    def get_model_config(self, model_name: str) -> Dict:
        """Get model configuration"""
        if model_name not in self._model_configs:
            raise ModelNotFoundError(model_name)
        return self._model_configs[model_name]

    def get_available_models(self) -> List[Dict]:
        """Get list of all available models"""
        return [
            {
                "name": name,
                "description": config.get("description", ""),
                "input_type": config.get("input_type", "unknown"),
                "version": config.get("version", "1.0"),
                "loaded": name in self._models,
                "framework": config.get("framework", "tensorflow")
            }
            for name, config in self._model_configs.items()
        ]

    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Unloaded model: {model_name}")

            # Clear TensorFlow session if it's a TF model
            if model_name in self._model_configs:
                framework = self._model_configs[model_name].get("framework", "tensorflow").lower()
                if framework in ["tensorflow", "keras"]:
                    try:
                        tf.keras.backend.clear_session()
                        logger.info("Cleared TensorFlow session")
                    except Exception as e:
                        logger.warning(f"Could not clear TF session: {str(e)}")

    def reload_model(self, model_name: str):
        """Reload a model (useful after model updates)"""
        logger.info(f"Reloading model: {model_name}")
        self.unload_model(model_name)
        return self._load_model(model_name)
