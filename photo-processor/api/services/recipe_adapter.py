"""
Recipe Adapter to bridge between different recipe formats
Handles conversion between ProcessingRecipe and service recipe formats
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from recipe_storage import ProcessingRecipe, ProcessingOperation

logger = logging.getLogger(__name__)


class RecipeAdapter:
    """Adapts between ProcessingRecipe format and service JSON format"""
    
    @staticmethod
    def service_to_processing_recipe(service_recipe: Dict[str, Any]) -> ProcessingRecipe:
        """Convert service recipe format to ProcessingRecipe"""
        recipe = ProcessingRecipe(
            id=service_recipe.get('id', ''),
            original_hash='',  # Will be set when applied to a photo
            original_filename='',  # Will be set when applied to a photo
        )
        
        # Convert operations
        operations = service_recipe.get('operations', service_recipe.get('steps', []))
        for op in operations:
            if isinstance(op, dict) and op.get('enabled', True):
                recipe.add_operation(
                    op_type=op.get('operation', op.get('type', 'unknown')),
                    parameters=op.get('parameters', {}),
                    source='recipe'
                )
        
        # Store metadata
        recipe.ai_metadata = {
            'style_preset': service_recipe.get('style_preset', 'natural'),
            'processing_config': service_recipe.get('processing_config', {})
        }
        
        recipe.user_overrides = {
            'name': service_recipe.get('name', 'Unnamed Recipe'),
            'description': service_recipe.get('description', ''),
            'is_builtin': service_recipe.get('is_builtin', False),
            'is_default': service_recipe.get('is_default', False)
        }
        
        return recipe
    
    @staticmethod
    def processing_to_service_recipe(
        processing_recipe: ProcessingRecipe,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert ProcessingRecipe to service recipe format"""
        operations = []
        for op in processing_recipe.operations:
            operations.append({
                'operation': op.type,
                'parameters': op.parameters,
                'enabled': True
            })
        
        service_recipe = {
            'id': processing_recipe.id,
            'name': name or processing_recipe.user_overrides.get('name', 'Converted Recipe'),
            'description': description or processing_recipe.user_overrides.get('description', ''),
            'operations': operations,
            'steps': operations,  # For frontend compatibility
            'style_preset': processing_recipe.ai_metadata.get('style_preset', 'natural'),
            'processing_config': processing_recipe.ai_metadata.get('processing_config', {}),
            'is_builtin': processing_recipe.user_overrides.get('is_builtin', False),
            'is_default': processing_recipe.user_overrides.get('is_default', False),
            'is_preset': processing_recipe.user_overrides.get('is_builtin', False),
            'created_at': processing_recipe.created_at.isoformat(),
            'updated_at': processing_recipe.created_at.isoformat(),
            'usage_count': 0
        }
        
        return service_recipe
    
    @staticmethod
    def load_service_recipe(recipe_path: Path) -> Optional[Dict[str, Any]]:
        """Load a service format recipe from file"""
        try:
            if recipe_path.exists():
                with open(recipe_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load recipe from {recipe_path}: {e}")
        return None
    
    @staticmethod  
    def save_service_recipe(recipe_path: Path, recipe_data: Dict[str, Any]) -> bool:
        """Save a service format recipe to file"""
        try:
            recipe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(recipe_path, 'w') as f:
                json.dump(recipe_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save recipe to {recipe_path}: {e}")
            return False
    
    @staticmethod
    def apply_recipe_to_photo(
        recipe: ProcessingRecipe,
        photo_hash: str,
        photo_filename: str
    ) -> ProcessingRecipe:
        """Create a photo-specific recipe instance"""
        # Clone the recipe
        photo_recipe = ProcessingRecipe(
            original_hash=photo_hash,
            original_filename=photo_filename,
            operations=recipe.operations.copy(),
            ai_metadata=recipe.ai_metadata.copy(),
            user_overrides=recipe.user_overrides.copy(),
            version=recipe.version
        )
        
        return photo_recipe