"""
AI Analysis service using Ollama with structured outputs
"""
import os
import logging
from typing import Optional
import ollama
from schemas import PhotoAnalysis
import json

logger = logging.getLogger(__name__)

class AIAnalyzer:
    def __init__(self, ollama_host: str = "http://ollama:11434", model: str = "gemma3:12b"):
        self.ollama_host = ollama_host
        self.model = model
        self.client = ollama.Client(host=ollama_host)
        
    def analyze_photo(self, image_path: str, original_filename: str) -> Optional[PhotoAnalysis]:
        """
        Analyze photo using Ollama with structured output
        """
        try:
            # Simplified prompt focusing on subject identification and optimal framing
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
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                ],
                format=PhotoAnalysis.model_json_schema(),
                options={
                    'temperature': 0.1,  # Low temperature for consistent results
                    'top_p': 0.9
                }
            )
            
            # Parse the structured response
            analysis_data = json.loads(response.message.content)
            
            # Log the raw AI response for debugging
            logger.debug(f"Raw AI response: {json.dumps(analysis_data, indent=2)}")
            
            # Fix common AI mistakes - ensure reasonable crop sizes
            def fix_bounding_box(box_data):
                # If values are > 100, assume they're pixel values for 896x896 image
                if box_data.get('width', 0) > 100:
                    box_data['width'] = min(100, box_data['width'] / 8.96)  # 896 pixels = 100%
                if box_data.get('height', 0) > 100:
                    box_data['height'] = min(100, box_data['height'] / 8.96)
                if box_data.get('x', 0) > 100:
                    box_data['x'] = min(100, box_data['x'] / 8.96)
                if box_data.get('y', 0) > 100:
                    box_data['y'] = min(100, box_data['y'] / 8.96)
                    
                # Allow very gentle crops or "enhance only" crops that preserve most of the image
                min_width = 20.0  # Allow minimal crops
                min_height = 20.0  # Allow minimal crops
                
                if box_data.get('width', 0) < min_width:
                    logger.warning(f"Crop width too small ({box_data.get('width', 0)}%), setting to {min_width}%")
                    box_data['width'] = min_width
                    
                if box_data.get('height', 0) < min_height:
                    logger.warning(f"Crop height too small ({box_data.get('height', 0)}%), setting to {min_height}%")
                    box_data['height'] = min_height
                
                # Just log the aspect ratio without enforcing it
                current_width = box_data.get('width', 0)
                current_height = box_data.get('height', 0)
                
                if current_height > 0:
                    current_aspect = current_width / current_height
                    logger.info(f"AI suggested crop aspect ratio: {current_aspect:.2f}:1 ({current_width:.1f}% x {current_height:.1f}%)")
                    
                # Ensure crop doesn't exceed image bounds
                if box_data.get('x', 0) + box_data.get('width', 0) > 100:
                    box_data['x'] = max(0, 100 - box_data['width'])
                if box_data.get('y', 0) + box_data.get('height', 0) > 100:
                    box_data['y'] = max(0, 100 - box_data['height'])
                    
                return box_data
            
            # Fix confidence values if they're percentages instead of decimals
            def fix_confidence(value):
                if isinstance(value, (int, float)) and value > 1:
                    return value / 100.0
                return value
            
            if 'subject_confidence' in analysis_data:
                analysis_data['subject_confidence'] = fix_confidence(analysis_data['subject_confidence'])
            
            # Fix bounding boxes if needed
            if 'primary_subject_box' in analysis_data:
                analysis_data['primary_subject_box'] = fix_bounding_box(analysis_data['primary_subject_box'])
            if 'recommended_crop' in analysis_data and 'crop_box' in analysis_data['recommended_crop']:
                analysis_data['recommended_crop']['crop_box'] = fix_bounding_box(analysis_data['recommended_crop']['crop_box'])
            
            analysis = PhotoAnalysis.model_validate(analysis_data)
            
            logger.info(f"AI analysis completed for {original_filename}")
            logger.info(f"Subject: {analysis.primary_subject}")
            logger.info(f"Subject box: x={analysis.primary_subject_box.x:.1f}%, y={analysis.primary_subject_box.y:.1f}%, w={analysis.primary_subject_box.width:.1f}%, h={analysis.primary_subject_box.height:.1f}%")
            logger.info(f"Crop recommendation: x={analysis.recommended_crop.crop_box.x:.1f}%, y={analysis.recommended_crop.crop_box.y:.1f}%, w={analysis.recommended_crop.crop_box.width:.1f}%, h={analysis.recommended_crop.crop_box.height:.1f}%")
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Raw response: {response.message.content if 'response' in locals() else 'No response'}")
            return None
            
        except Exception as e:
            logger.error(f"Error during AI analysis: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama service
        """
        try:
            models_response = self.client.list()
            logger.debug(f"Raw Ollama response: {models_response}")
            
            # Handle both dict and object response formats
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                logger.error(f"Unexpected Ollama response format: {type(models_response)}")
                return False
            
            # Extract model names - handle both dict and object formats
            available_models = []
            for model in models_list:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict) and 'model' in model:
                    available_models.append(model['model'])
                elif isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
                else:
                    logger.warning(f"Unexpected model format: {model}")
            
            if self.model in available_models:
                logger.info(f"Successfully connected to Ollama. Model {self.model} is available.")
                return True
            else:
                logger.warning(f"Model {self.model} not found. Available models: {available_models}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def ensure_model_available(self) -> bool:
        """
        Ensure the required model is pulled and available
        """
        try:
            # Try to pull the model if not available
            logger.info(f"Ensuring model {self.model} is available...")
            
            # Check if model exists
            models_response = self.client.list()
            
            # Handle both dict and object response formats
            if hasattr(models_response, 'models'):
                models_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                models_list = models_response['models']
            else:
                logger.error(f"Unexpected Ollama response format: {type(models_response)}")
                return False
            
            # Extract model names - handle both dict and object formats
            available_models = []
            for model in models_list:
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif hasattr(model, 'name'):
                    available_models.append(model.name)
                elif isinstance(model, dict) and 'model' in model:
                    available_models.append(model['model'])
                elif isinstance(model, dict) and 'name' in model:
                    available_models.append(model['name'])
            
            if self.model not in available_models:
                logger.info(f"Pulling model {self.model}...")
                self.client.pull(self.model)
                logger.info(f"Model {self.model} pulled successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            return False