"""
Custom exceptions for the application
"""


class BaseAPIException(Exception):
    """Base exception class for API errors"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelNotFoundError(BaseAPIException):
    """Raised when a requested model is not found"""
    def __init__(self, model_name: str):
        message = f"Model '{model_name}' not found in registry"
        super().__init__(message, "MODEL_NOT_FOUND")
        self.model_name = model_name


class ModelLoadError(BaseAPIException):
    """Raised when a model fails to load"""
    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR")
        self.model_name = model_name


class PredictionError(BaseAPIException):
    """Raised when prediction fails"""
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message, "PREDICTION_ERROR")
        self.model_name = model_name


class InvalidInputError(BaseAPIException):
    """Raised when input validation fails"""
    def __init__(self, message: str):
        super().__init__(message, "INVALID_INPUT")


class ImageProcessingError(BaseAPIException):
    """Raised when image processing fails"""
    def __init__(self, message: str):
        super().__init__(message, "IMAGE_PROCESSING_ERROR")


class ModelRegistryError(BaseAPIException):
    """Raised when model registry operations fail"""
    def __init__(self, message: str):
        super().__init__(message, "MODEL_REGISTRY_ERROR")
