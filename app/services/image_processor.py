"""
Image processing service for handling image inputs
"""
import io
import base64
import logging
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional
import tensorflow as tf

from app.core.config import settings
from app.core.exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Service for processing images for model input"""
    
    @staticmethod
    def validate_image(image_bytes: bytes) -> bool:
        """Validate image format and size"""
        try:
            # Check size
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > settings.MAX_IMAGE_SIZE_MB:
                raise ImageProcessingError(
                    f"Image size {size_mb:.2f}MB exceeds maximum {settings.MAX_IMAGE_SIZE_MB}MB"
                )
            
            # Try to open image
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            
            return True
        
        except Exception as e:
            raise ImageProcessingError(f"Invalid image file: {str(e)}")
    
    @staticmethod
    def load_image(image_bytes: bytes) -> Image.Image:
        """Load image from bytes"""
        try:
            ImageProcessor.validate_image(image_bytes)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
        
        except Exception as e:
            raise ImageProcessingError(f"Error loading image: {str(e)}")
    
    @staticmethod
    def preprocess_image(
        img: Image.Image,
        target_size: Tuple[int, int],
        normalize: bool = True,
        preprocessing_func: Optional[callable] = None
    ) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Resize image
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to array
            img_array = np.array(img_resized)
            
            # Normalize if requested
            if normalize:
                img_array = img_array.astype(np.float32) / 255.0
            
            # Apply custom preprocessing if provided
            if preprocessing_func:
                img_array = preprocessing_func(img_array)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        
        except Exception as e:
            raise ImageProcessingError(f"Error preprocessing image: {str(e)}")
    
    @staticmethod
    def generate_gradcam(
        model: tf.keras.Model,
        img_array: np.ndarray,
        layer_name: Optional[str] = None,
        pred_index: Optional[int] = None
    ) -> np.ndarray:
        """Generate GradCAM heatmap"""
        try:
            # Find last convolutional layer if not specified
            if layer_name is None or layer_name == "auto":
                for layer in reversed(model.layers):
                    if len(layer.output_shape) == 4:  # Conv layer
                        layer_name = layer.name
                        break
            
            if layer_name is None:
                raise ImageProcessingError("No convolutional layer found for GradCAM")
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(layer_name).output, model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
            
            # Compute gradients of class output with respect to feature map
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Global average pooling
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            return heatmap
        
        except Exception as e:
            logger.error(f"Error generating GradCAM: {str(e)}")
            raise ImageProcessingError(f"Error generating GradCAM: {str(e)}")
    
    @staticmethod
    def overlay_gradcam(
        original_img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> str:
        """Overlay GradCAM heatmap on original image and return base64 string"""
        try:
            # Resize heatmap to match image
            img_array = np.array(original_img)
            heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            
            # Convert heatmap to RGB
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_resized),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap
            overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
            
            # Convert to PIL Image
            result_img = Image.fromarray(overlayed)
            
            # Convert to base64
            buffered = io.BytesIO()
            result_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return img_base64
        
        except Exception as e:
            raise ImageProcessingError(f"Error overlaying GradCAM: {str(e)}")
        