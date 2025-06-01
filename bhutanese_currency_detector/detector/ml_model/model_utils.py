# /detector/ml_model/model_utils.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import base64
import os
from django.conf import settings
from PIL import Image, ImageEnhance
import json

# Load the Keras model once when the module is imported
MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector', 'ml_model', 'keras_model.h5')
model = None

# Currency information mapping based on labels.txt
CURRENCY_INFO = {
    0: {
        'name': 'Nu.1',
        'description': 'Bhutanese Ngultrum 1 note - The smallest denomination of Bhutanese currency.',
        'audio': 'Nu. 1.mp3'
    },
    1: {
        'name': 'Nu.5',
        'description': 'Bhutanese Ngultrum 5 note - Features traditional Bhutanese artwork and symbols.',
        'audio': 'Nu. 5.mp3'
    },
    2: {
        'name': 'Nu.10',
        'description': 'Bhutanese Ngultrum 10 note - Commonly used denomination in daily transactions.',
        'audio': 'Nu.10.mp3'
    },
    3: {
        'name': 'Nu.20',
        'description': 'Bhutanese Ngultrum 20 note - Features the portrait of the King of Bhutan.',
        'audio': 'Nu.20.mp3'
    },
    4: {
        'name': 'Nu.50',
        'description': 'Bhutanese Ngultrum 50 note - Medium denomination note with security features.',
        'audio': 'Nu. 50.mp3'
    },
    5: {
        'name': 'Nu.100',
        'description': 'Bhutanese Ngultrum 100 note - High value note with advanced security features.',
        'audio': 'Nu. 100.mp3'
    },
    6: {
        'name': 'Nu.500',
        'description': 'Bhutanese Ngultrum 500 note - One of the highest denomination notes.',
        'audio': 'Nu. 500.mp3'
    },
    7: {
        'name': 'Nu.1000',
        'description': 'Bhutanese Ngultrum 1000 note - The highest denomination of Bhutanese currency.',
        'audio': 'Nu. 1000.mp3'
    },
    8: {
        'name': 'Unknown',
        'description': 'Could not identify the currency. Please ensure the image is clear and shows a Bhutanese note.',
        'audio': 'unknown.mp3'
    }
}

def fix_model_config(config):
    """
    Fix model configuration to be compatible with current TensorFlow version
    """
    if isinstance(config, dict):
        # Remove 'groups' parameter from DepthwiseConv2D layers
        if config.get('class_name') == 'DepthwiseConv2D':
            if 'config' in config and isinstance(config['config'], dict):
                config['config'].pop('groups', None)
        
        # Recursively fix nested configurations
        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = fix_model_config(value)
            elif isinstance(value, list):
                config[key] = [fix_model_config(item) if isinstance(item, dict) else item for item in value]
    
    return config

def load_model_with_compatibility():
    """
    Load model with TensorFlow version compatibility fixes
    """
    try:
        print(f"TensorFlow version: {tf.__version__}")
        
        # Try loading directly first
        try:
            model = keras.models.load_model(MODEL_PATH)
            return model
        except Exception as direct_load_error:
            print(f"Direct loading failed: {direct_load_error}")
            
            # Try custom object scope for compatibility
            try:
                # Define custom DepthwiseConv2D that ignores 'groups' parameter
                class CompatibleDepthwiseConv2D(keras.layers.DepthwiseConv2D):
                    def __init__(self, *args, **kwargs):
                        # Remove incompatible parameters
                        kwargs.pop('groups', None)
                        super().__init__(*args, **kwargs)
                
                custom_objects = {
                    'DepthwiseConv2D': CompatibleDepthwiseConv2D
                }
                
                model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
                return model
                
            except Exception as custom_load_error:
                print(f"Custom object loading failed: {custom_load_error}")
                
                # Last resort: try to rebuild the model
                try:
                    # Load the model architecture and weights separately
                    with open(MODEL_PATH.replace('.h5', '_architecture.json'), 'r') as f:
                        model_json = json.load(f)
                    
                    # Fix the configuration
                    fixed_config = fix_model_config(model_json)
                    
                    # Recreate model from fixed config
                    model = keras.models.model_from_json(json.dumps(fixed_config))
                    model.load_weights(MODEL_PATH.replace('.h5', '_weights.h5'))
                    
                    return model
                    
                except Exception as rebuild_error:
                    print(f"Model rebuild failed: {rebuild_error}")
                    raise direct_load_error  # Raise the original error
                    
    except Exception as e:
        print(f"All model loading attempts failed: {str(e)}")
        raise e

def load_model():
    """Load the Keras model with compatibility handling"""
    global model
    if model is None:
        try:
            if os.path.exists(MODEL_PATH):
                model = load_model_with_compatibility()
                print(f"Model loaded successfully from {MODEL_PATH}")
                print(f"Model input shape: {model.input_shape}")
                print(f"Model output shape: {model.output_shape}")
                
                # Test the model with sample input
                sample_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
                sample_output = model.predict(sample_input, verbose=0)
                print(f"Sample prediction shape: {sample_output.shape}")
                print(f"Sample prediction range: {np.min(sample_output):.4f} to {np.max(sample_output):.4f}")
                
            else:
                print(f"Model file not found at {MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    return model

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model prediction
    
    Args:
        image: PIL Image object
        target_size: tuple of (width, height) for resizing
    
    Returns:
        numpy array ready for model prediction
    """
    try:
        print(f"Original image size: {image.size}, mode: {image.mode}")
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image with high quality resampling
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        print(f"Image array shape after resize: {img_array.shape}")
        
        # Normalize pixel values to [0, 1] (standard for most models)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array_final = np.expand_dims(img_array, axis=0)
        
        print(f"Final preprocessed shape: {img_array_final.shape}")
        print(f"Final preprocessed range: {np.min(img_array_final):.4f} to {np.max(img_array_final):.4f}")
        
        return img_array_final
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise e

def detect_currency_from_image(image):
    """
    Detect currency from PIL Image object with enhanced error handling
    
    Args:
        image: PIL Image object
    
    Returns:
        dict: Contains currency name, description, and audio file
    """
    try:
        # Load model if not already loaded
        model = load_model()
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        print("Making prediction...")
        predictions = model.predict(processed_image, verbose=0)
        
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions: {predictions[0]}")
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Print all class probabilities for debugging
        for i, prob in enumerate(predictions[0]):
            currency_name = CURRENCY_INFO.get(i, {}).get('name', f'Class_{i}')
            print(f"{currency_name}: {prob:.4f}")
        
        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
        
        # Set a minimum confidence threshold
        MIN_CONFIDENCE = 0.1  # Lower threshold for testing
        
        # If confidence is too low, return unknown
        if confidence < MIN_CONFIDENCE:
            print(f"Confidence {confidence:.4f} below threshold {MIN_CONFIDENCE}, returning unknown")
            result = CURRENCY_INFO[8].copy()  # Unknown
            result['confidence'] = confidence
            result['predicted_class'] = int(predicted_class)
        else:
            # Get currency information
            currency_info = CURRENCY_INFO.get(predicted_class, CURRENCY_INFO[8])
            result = currency_info.copy()
            result['confidence'] = confidence
            result['predicted_class'] = int(predicted_class)
        
        # Add debugging information
        result['all_predictions'] = predictions[0].tolist()
        result['model_working'] = True
        
        return result
        
    except Exception as e:
        print(f"Error in currency detection: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return unknown currency info with error details
        error_result = CURRENCY_INFO[8].copy()
        error_result['confidence'] = 0.0
        error_result['predicted_class'] = -1
        error_result['error'] = str(e)
        error_result['model_working'] = False
        
        return error_result

def process_base64_image(base64_string):
    """
    Convert base64 image string to PIL Image object with validation
    
    Args:
        base64_string: base64 encoded image string
    
    Returns:
        PIL Image object
    """
    try:
        print(f"Processing base64 image, length: {len(base64_string)}")
        
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        print(f"Decoded image data size: {len(image_data)} bytes")
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        print(f"Loaded PIL image: {image.size}, mode: {image.mode}")
        
        return image
        
    except Exception as e:
        print(f"Error processing base64 image: {str(e)}")
        raise e

def debug_model_predictions(image, num_augmentations=3):
    """
    Debug function to test model with multiple image variations
    
    Args:
        image: PIL Image object
        num_augmentations: number of slight variations to test
    
    Returns:
        dict: debugging information
    """
    try:
        model = load_model()
        results = []
        
        # Original image
        processed = preprocess_image(image)
        pred = model.predict(processed, verbose=0)
        results.append({
            'variation': 'original',
            'prediction': pred[0].tolist(),
            'predicted_class': int(np.argmax(pred[0])),
            'confidence': float(np.max(pred[0]))
        })
        
        # Try with different brightness levels
        for i in range(num_augmentations):
            try:
                # Adjust brightness
                enhancer = ImageEnhance.Brightness(image)
                factor = 0.8 + (i * 0.2)  # 0.8, 1.0, 1.2
                varied_image = enhancer.enhance(factor)
                
                processed = preprocess_image(varied_image)
                pred = model.predict(processed, verbose=0)
                results.append({
                    'variation': f'brightness_{factor:.1f}',
                    'prediction': pred[0].tolist(),
                    'predicted_class': int(np.argmax(pred[0])),
                    'confidence': float(np.max(pred[0]))
                })
            except Exception as aug_error:
                print(f"Error in augmentation {i}: {aug_error}")
                continue
        
        return {'debug_results': results}
        
    except Exception as e:
        return {'error': str(e)}

def get_model_info():
    """
    Get comprehensive information about the loaded model
    
    Returns:
        dict: Model information
    """
    try:
        model = load_model()
        
        info = {
            'model_type': 'Keras/TensorFlow',
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': model.count_params(),
            'model_path': MODEL_PATH,
            'model_exists': os.path.exists(MODEL_PATH),
            'tensorflow_version': tf.__version__,
            'num_classes': len(CURRENCY_INFO) - 1  # Excluding 'Unknown'
        }
        
        # Test with dummy input
        try:
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            dummy_output = model.predict(dummy_input, verbose=0)
            info['dummy_prediction_shape'] = str(dummy_output.shape)
            info['dummy_prediction_range'] = f"{np.min(dummy_output):.4f} to {np.max(dummy_output):.4f}"
            info['model_working'] = True
        except Exception as e:
            info['model_working'] = False
            info['model_error'] = str(e)
        
        return info
        
    except Exception as e:
        return {
            'error': f"Could not get model info: {str(e)}",
            'model_exists': os.path.exists(MODEL_PATH) if MODEL_PATH else False,
            'tensorflow_version': tf.__version__,
            'model_working': False
        }

def check_tensorflow_compatibility():
    """
    Check TensorFlow version and provide upgrade recommendations
    
    Returns:
        dict: Compatibility information
    """
    current_version = tf.__version__
    version_parts = [int(x) for x in current_version.split('.')]
    
    compatibility_info = {
        'current_version': current_version,
        'compatible': True,
        'recommendations': []
    }
    
    # Check for known compatibility issues
    if version_parts[0] < 2:
        compatibility_info['compatible'] = False
        compatibility_info['recommendations'].append('Upgrade to TensorFlow 2.x')
    
    if version_parts[0] == 2 and version_parts[1] < 4:
        compatibility_info['recommendations'].append('Consider upgrading to TensorFlow 2.4+ for better model compatibility')
    
    return compatibility_info