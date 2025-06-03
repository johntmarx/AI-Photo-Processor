"""
AI Analysis service using Ollama with structured outputs.

This module provides intelligent photo analysis capabilities using the Ollama AI service
with Gemma3 model. It analyzes photos to identify subjects, composition, and provides
recommendations for optimal cropping, rotation, and color adjustments.

Key features:
- Subject detection and localization with bounding boxes
- Composition analysis (rule of thirds, symmetry, etc.)
- Intelligent crop recommendations
- Rotation correction for tilted horizons
- Color analysis for enhancement suggestions
- Structured output using Pydantic models for reliability
"""
import os
import logging
from typing import Optional
import ollama
from schemas import PhotoAnalysis
import json

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """
    AI photo analyzer using Ollama for intelligent image analysis.
    
    This class interfaces with Ollama to analyze photos and provide structured
    recommendations for processing. It handles model availability, connection
    management, and response parsing with error correction.
    """
    
    def __init__(self, ollama_host: str = "http://ollama:11434", model: str = "gemma3:12b"):
        """
        Initialize the AI analyzer with Ollama connection details.
        
        Args:
            ollama_host: URL of the Ollama service endpoint
            model: Name of the model to use (default: gemma3:12b for vision capabilities)
        """
        self.ollama_host = ollama_host
        self.model = model
        self.client = ollama.Client(host=ollama_host)
        
    def analyze_photo(self, image_path: str, original_filename: str) -> Optional[PhotoAnalysis]:
        """
        Analyze a photo using Ollama AI to identify subjects and recommend processing.
        
        This method sends an image to the Ollama AI service for analysis. The AI will:
        - Identify the main subject and its location in the image
        - Recommend optimal cropping to improve composition
        - Suggest rotation angles to level horizons or fix orientation
        - Provide confidence scores for its recommendations
        
        Args:
            image_path: Path to the preprocessed image (896x896 with letterboxing)
            original_filename: Original filename for context (helps AI understand content)
            
        Returns:
            PhotoAnalysis object with structured recommendations, or None if analysis fails
            
        Note:
            The image is preprocessed to 896x896 pixels with black letterboxing to maintain
            aspect ratio. All coordinates returned are in percentages (0-100) to be
            resolution-independent.
        """
        try:
            # Construct the prompt that will guide the AI's analysis
            # This prompt is carefully crafted to ensure the AI returns coordinates
            # in the correct format (percentages, not pixels) and understands the
            # letterboxing preprocessing that has been applied
            prompt = f"""
Analyze this photo to identify the main subject and determine optimal framing.

IMPORTANT: 
- You're seeing a 896x896 pixel image (with black letterboxing if needed)
- ALL coordinates must be given as PERCENTAGES (0-100), not pixels!
- The black borders are letterboxing - the actual photo is within them

Your task:
1. Identify the primary subject and its exact location
2. Determine if the image needs rotation to be level/properly oriented
3. Define the optimal crop to best showcase the subject

COORDINATE SYSTEM - CRITICAL:
- All values must be percentages from 0 to 100
- x=0 is left edge, x=100 is right edge
- y=0 is top edge, y=100 is bottom edge
- width and height are also percentages of the full image
- Example: A centered 50% crop would be: x=25, y=25, width=50, height=50

For rotation_degrees:
- Positive values rotate clockwise, negative counter-clockwise
- Usually between -10 and +10 degrees for straightening
- 0 if no rotation needed

For crop_box:
- Define the area that best frames the subject
- Consider composition rules (rule of thirds, leading space for motion)
- Include environmental context when it adds to the story
- For full image: x=0, y=0, width=100, height=100

Original filename: {original_filename}
"""

            # Call Ollama with structured output
            # The format parameter ensures the AI response matches our PhotoAnalysis schema
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]  # Include the preprocessed image
                    }
                ],
                format=PhotoAnalysis.model_json_schema(),  # Force structured output matching our schema
                options={
                    'temperature': 0.1,  # Low temperature for consistent, deterministic results
                    'top_p': 0.9        # High nucleus sampling for focused responses
                }
            )
            
            # Parse the structured response from JSON string to Python dict
            # The AI is instructed to return JSON matching our PhotoAnalysis schema
            analysis_data = json.loads(response.message.content)
            
            # Log the raw AI response for debugging
            # This helps diagnose issues when the AI returns unexpected formats
            logger.debug(f"Raw AI response: {json.dumps(analysis_data, indent=2)}")
            
            # Error correction function for bounding boxes
            # Sometimes the AI returns pixel coordinates instead of percentages,
            # or suggests crops that are too small or exceed image boundaries
            def fix_bounding_box(box_data):
                """
                Correct common AI mistakes in bounding box coordinates.
                
                This function handles several types of errors:
                1. Pixel coordinates instead of percentages (values > 100)
                2. Crops that are too small to be useful
                3. Crops that extend beyond image boundaries
                
                Args:
                    box_data: Dict with x, y, width, height keys
                    
                Returns:
                    Corrected box_data dict
                """
                # Check if AI returned pixel values instead of percentages
                # Our preprocessed images are 896x896, so we convert accordingly
                if box_data.get('width', 0) > 100:
                    box_data['width'] = min(100, box_data['width'] / 8.96)  # 896 pixels = 100%
                if box_data.get('height', 0) > 100:
                    box_data['height'] = min(100, box_data['height'] / 8.96)
                if box_data.get('x', 0) > 100:
                    box_data['x'] = min(100, box_data['x'] / 8.96)
                if box_data.get('y', 0) > 100:
                    box_data['y'] = min(100, box_data['y'] / 8.96)
                    
                # Ensure minimum crop size for practical use
                # We allow crops as small as 20% to support tight subject framing
                min_width = 20.0  # Minimum 20% of image width
                min_height = 20.0  # Minimum 20% of image height
                
                if box_data.get('width', 0) < min_width:
                    logger.warning(f"Crop width too small ({box_data.get('width', 0)}%), setting to {min_width}%")
                    box_data['width'] = min_width
                    
                if box_data.get('height', 0) < min_height:
                    logger.warning(f"Crop height too small ({box_data.get('height', 0)}%), setting to {min_height}%")
                    box_data['height'] = min_height
                
                # Log the aspect ratio for debugging but don't enforce it
                # This allows the AI to suggest crops that match the subject's shape
                current_width = box_data.get('width', 0)
                current_height = box_data.get('height', 0)
                
                if current_height > 0:
                    current_aspect = current_width / current_height
                    logger.info(f"AI suggested crop aspect ratio: {current_aspect:.2f}:1 ({current_width:.1f}% x {current_height:.1f}%)")
                    
                # Ensure crop stays within image boundaries
                # If the crop would extend past the edge, shift it back inside
                if box_data.get('x', 0) + box_data.get('width', 0) > 100:
                    box_data['x'] = max(0, 100 - box_data['width'])
                if box_data.get('y', 0) + box_data.get('height', 0) > 100:
                    box_data['y'] = max(0, 100 - box_data['height'])
                    
                return box_data
            
            # Error correction function for confidence values
            def fix_confidence(value):
                """
                Convert confidence values to proper 0-1 range.
                
                Some AI models return confidence as percentages (0-100) instead
                of decimals (0-1). This function ensures consistency.
                
                Args:
                    value: Confidence value from AI
                    
                Returns:
                    Normalized confidence value between 0 and 1
                """
                if isinstance(value, (int, float)) and value > 1:
                    return value / 100.0  # Convert percentage to decimal
                return value
            
            # Apply error correction to confidence scores
            if 'subject_confidence' in analysis_data:
                analysis_data['subject_confidence'] = fix_confidence(analysis_data['subject_confidence'])
            
            # Apply error correction to all bounding boxes in the response
            # The AI may have multiple bounding boxes that need correction
            if 'primary_subject_box' in analysis_data:
                analysis_data['primary_subject_box'] = fix_bounding_box(analysis_data['primary_subject_box'])
            if 'recommended_crop' in analysis_data and 'crop_box' in analysis_data['recommended_crop']:
                analysis_data['recommended_crop']['crop_box'] = fix_bounding_box(analysis_data['recommended_crop']['crop_box'])
            
            # Validate the corrected data against our Pydantic schema
            # This ensures all required fields are present and properly typed
            analysis = PhotoAnalysis.model_validate(analysis_data)
            
            # Log analysis results for monitoring and debugging
            logger.info(f"AI analysis completed for {original_filename}")
            logger.info(f"Subject: {analysis.primary_subject}")
            logger.info(f"Subject box: x={analysis.primary_subject_box.x:.1f}%, y={analysis.primary_subject_box.y:.1f}%, w={analysis.primary_subject_box.width:.1f}%, h={analysis.primary_subject_box.height:.1f}%")
            logger.info(f"Crop recommendation: x={analysis.recommended_crop.crop_box.x:.1f}%, y={analysis.recommended_crop.crop_box.y:.1f}%, w={analysis.recommended_crop.crop_box.width:.1f}%, h={analysis.recommended_crop.crop_box.height:.1f}%")
            
            return analysis
            
        except json.JSONDecodeError as e:
            # Handle cases where the AI returns invalid JSON
            # This can happen if the model is not properly configured for structured output
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Raw response: {response.message.content if 'response' in locals() else 'No response'}")
            return None
            
        except Exception as e:
            # Catch-all for other errors (network issues, model errors, etc.)
            logger.error(f"Error during AI analysis: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama service and verify model availability.
        
        This method checks:
        1. If we can connect to the Ollama service
        2. If our required model is available
        
        Returns:
            True if connected and model is available, False otherwise
            
        Note:
            The Ollama API may return model lists in different formats depending
            on the version. This method handles multiple response formats for
            compatibility.
        """
        try:
            # Request list of available models from Ollama
            models_response = self.client.list()
            logger.debug(f"Raw Ollama response: {models_response}")
            
            # Handle different response formats from various Ollama versions
            # Newer versions return an object with 'models' attribute
            # Older versions may return a dict with 'models' key
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                logger.error(f"Unexpected Ollama response format: {type(models_response)}")
                return False
            
            # Extract model names from the list
            # Different Ollama versions use different attribute names for the model identifier
            available_models = []
            for model in models_list:
                # Try various attribute names that Ollama might use
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict) and 'model' in model:
                    available_models.append(model['model'])
                elif isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
                else:
                    # Log unexpected formats to help with future compatibility
                    logger.warning(f"Unexpected model format: {model}")
            
            # Check if our required model is available
            if self.model in available_models:
                logger.info(f"Successfully connected to Ollama. Model {self.model} is available.")
                return True
            else:
                logger.warning(f"Model {self.model} not found. Available models: {available_models}")
                return False
                
        except Exception as e:
            # Connection failed - likely Ollama service is not running
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def ensure_model_available(self) -> bool:
        """
        Ensure the required model is downloaded and available in Ollama.
        
        This method will:
        1. Check if the model is already available
        2. If not, attempt to pull (download) it from the Ollama registry
        
        Returns:
            True if model is available or successfully pulled, False otherwise
            
        Note:
            Pulling a model can take significant time (several GB download).
            The gemma3:12b model is approximately 8GB and provides good
            vision capabilities for photo analysis.
        """
        try:
            logger.info(f"Ensuring model {self.model} is available...")
            
            # First, check current model availability
            models_response = self.client.list()
            
            # Handle different response formats (same logic as test_connection)
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                logger.error(f"Unexpected Ollama response format: {type(models_response)}")
                return False
            
            # Extract model names to check availability
            available_models = []
            for model in models_list:
                # Support multiple attribute naming conventions
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict) and 'model' in model:
                    available_models.append(model['model'])
                elif isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
            
            # If model not available, pull it from Ollama registry
            if self.model not in available_models:
                logger.info(f"Model {self.model} not found locally. Pulling from registry...")
                logger.info("This may take several minutes for large models...")
                
                # Pull the model - this downloads it from Ollama's servers
                self.client.pull(self.model)
                logger.info(f"Model {self.model} pulled successfully")
            else:
                logger.info(f"Model {self.model} is already available")
            
            return True
            
        except Exception as e:
            # Pull might fail due to network issues or invalid model name
            logger.error(f"Failed to ensure model availability: {e}")
            return False