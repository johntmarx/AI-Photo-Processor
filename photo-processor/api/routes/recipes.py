"""
Recipe management routes
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List, Dict, Any

from services.recipe_service_v2 import recipe_service
from middleware.transform import backend_to_frontend, transform_recipe, transform_recipe_list, transform_pagination_response

router = APIRouter()
# recipe_service is imported as singleton from recipe_service_v2

@router.get("")
async def list_recipes(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at", description="Sort field"),
    order: str = Query("desc", description="Sort order")
):
    """List all available recipes"""
    try:
        recipes = await recipe_service.list_recipes(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            order=order
        )
        # Check if it's a paginated response or just a list
        if isinstance(recipes, dict) and 'items' in recipes:
            # Transform each recipe in the items
            recipes['items'] = transform_recipe_list(recipes['items'])
            return transform_pagination_response(recipes)
        elif isinstance(recipes, list):
            # Just a list of recipes
            return transform_recipe_list(recipes)
        else:
            # It's already a dict, transform it
            return backend_to_frontend(recipes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/presets")
async def get_recipe_presets():
    """Get built-in recipe presets"""
    try:
        presets = await recipe_service.get_presets()
        return {"presets": presets}
    except Exception as e:
        # Log the error for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting recipe presets: {str(e)}")
        # Return empty presets for now
        return {"presets": []}

@router.get("/{recipe_id}")
async def get_recipe(recipe_id: str):
    """Get detailed information about a specific recipe"""
    recipe = await recipe_service.get_recipe(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    # Transform single recipe
    return transform_recipe(recipe)

@router.post("")
async def create_recipe(recipe_data: Dict[str, Any]):
    """Create a new recipe"""
    try:
        # Log the incoming data for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Creating recipe with data: {recipe_data}")
        
        # Extract fields from the recipe data
        recipe = await recipe_service.create_recipe(
            name=recipe_data.get('name'),
            description=recipe_data.get('description'),
            operations=recipe_data.get('steps', recipe_data.get('operations', [])),
            style_preset=recipe_data.get('style_preset', 'natural'),
            processing_config=recipe_data.get('processing_config'),
            is_default=recipe_data.get('is_default', False)
        )
        return recipe
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        logger.error(f"Error creating recipe: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{recipe_id}")
async def update_recipe(recipe_id: str, recipe_data: Dict[str, Any]):
    """Update an existing recipe"""
    try:
        # Log the incoming data for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Updating recipe {recipe_id} with data: {recipe_data}")
        
        recipe = await recipe_service.update_recipe(
            recipe_id=recipe_id,
            name=recipe_data.get('name'),
            description=recipe_data.get('description'),
            operations=recipe_data.get('steps', recipe_data.get('operations')),
            style_preset=recipe_data.get('style_preset'),
            processing_config=recipe_data.get('processing_config'),
            is_default=recipe_data.get('is_default')
        )
        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")
        return recipe
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        logger.error(f"Error updating recipe: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{recipe_id}")
async def delete_recipe(recipe_id: str):
    """Delete a recipe"""
    success = await recipe_service.delete_recipe(recipe_id)
    if not success:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return {"message": "Recipe deleted successfully"}

@router.post("/{recipe_id}/duplicate")
async def duplicate_recipe(
    recipe_id: str,
    new_name: str = Body(..., description="Name for the duplicated recipe")
):
    """Duplicate an existing recipe"""
    try:
        new_recipe = await recipe_service.duplicate_recipe(recipe_id, new_name)
        if not new_recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")
        return new_recipe
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{recipe_id}/apply")
async def apply_recipe_to_photos(
    recipe_id: str,
    photo_ids: List[str] = Body(..., description="Photos to apply recipe to"),
    priority: str = Body("normal", description="Processing priority")
):
    """Apply a recipe to multiple photos"""
    try:
        result = await recipe_service.apply_to_photos(
            recipe_id=recipe_id,
            photo_ids=photo_ids,
            priority=priority
        )
        if not result:
            raise HTTPException(status_code=404, detail="Recipe not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{recipe_id}/preview")
async def preview_recipe(
    recipe_id: str,
    photo_id: str = Query(..., description="Photo to preview recipe on")
):
    """Preview a recipe on a photo without processing"""
    try:
        preview = await recipe_service.preview_recipe(recipe_id, photo_id)
        if not preview:
            raise HTTPException(status_code=404, detail="Recipe or photo not found")
        return preview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))