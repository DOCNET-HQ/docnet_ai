"""
Model Registry - Manages loading and accessing AI models
"""
import json
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import tensorflow as tf
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
                model = tf.keras.models.load_model(str(model_path))
            elif framework == "pytorch":
                model = torch.load(str(model_path))
                model.eval()
            elif framework == "sklearn":
                import joblib
                model = joblib.load(str(model_path))
            else:
                raise ModelLoadError(model_name, f"Unsupported framework: {framework}")
            
            self._models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            
            return model
        
        except Exception as e:
            raise ModelLoadError(model_name, str(e))
    
    def get_model(self, model_name: str) -> Any:
        """Get a model by name, loading it if necessary"""
        if model_name not in self._models:
            self._load_model(model_name)
        return self._models[model_name]
    
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
                "loaded": name in self._models
            }
            for name, config in self._model_configs.items()
        ]
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Unloaded model: {model_name}")
