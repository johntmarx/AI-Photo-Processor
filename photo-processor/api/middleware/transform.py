"""
Data transformation middleware for API responses
Converts backend formats to frontend expectations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import re

def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase"""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case"""
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def transform_dict_keys(obj: Any, transformer=snake_to_camel) -> Any:
    """Recursively transform dictionary keys"""
    if isinstance(obj, dict):
        return {
            transformer(k) if isinstance(k, str) else k: transform_dict_keys(v, transformer) 
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [transform_dict_keys(item, transformer) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def backend_to_frontend(data: Any) -> Any:
    """Transform backend response to frontend format"""
    return transform_dict_keys(data, snake_to_camel)

def frontend_to_backend(data: Any) -> Any:
    """Transform frontend request to backend format"""
    return transform_dict_keys(data, camel_to_snake)

def transform_photo(photo: Dict[str, Any]) -> Dict[str, Any]:
    """Transform photo object specifically"""
    # First, ensure critical fields are present and correctly typed
    if 'file_size' in photo and photo['file_size'] is None:
        photo['file_size'] = 0
    
    # Transform to camelCase - frontend now expects camelCase
    transformed = backend_to_frontend(photo)
    
    # Convert file paths to API URLs if needed
    for path_field in ['originalPath', 'processedPath', 'thumbnailPath', 'webPath']:
        if path_field in transformed and transformed[path_field]:
            # Convert absolute path to relative API URL
            path = transformed[path_field]
            if path.startswith('/app/data/'):
                # Convert /app/data/processed/file.jpg to /api/files/processed/file.jpg
                transformed[path_field] = path.replace('/app/data/', '/api/files/')
    
    return transformed

def transform_queue_status(status: Dict[str, Any]) -> Dict[str, Any]:
    """Transform queue status specifically"""
    # First do general transformation
    transformed = backend_to_frontend(status)
    
    # Add computed fields
    if 'pending' in transformed and isinstance(transformed['pending'], list):
        if 'processing' in transformed and isinstance(transformed['processing'], list):
            if 'completed' in transformed and isinstance(transformed['completed'], list):
                transformed['total'] = (
                    len(transformed['pending']) + 
                    len(transformed['processing']) + 
                    len(transformed['completed'])
                )
    
    # Add current item if processing
    if 'processing' in transformed and transformed['processing']:
        transformed['currentItem'] = transformed['processing'][0]
    
    return transformed

def transform_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Transform recipe object specifically"""
    transformed = backend_to_frontend(recipe)
    
    # Convert operations to steps for frontend compatibility
    if 'operations' in transformed:
        transformed['steps'] = transformed.pop('operations')
    
    return transformed

def transform_recipe_list(recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform list of recipes"""
    return [transform_recipe(recipe) for recipe in recipes]

def transform_pagination_response(response: Dict[str, Any], item_key: str = 'items') -> Dict[str, Any]:
    """Transform paginated response"""
    transformed = backend_to_frontend(response)
    
    # Handle different item keys (items vs photos)
    if item_key != 'items' and item_key in response:
        transformed[item_key] = backend_to_frontend(response[item_key])
    
    return transformed