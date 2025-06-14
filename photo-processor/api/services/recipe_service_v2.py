"""
Enhanced Recipe service with SQLite integration
Connects the API layer to SQLite database for recipe storage.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
import uuid
import sys
import logging

# Add parent directory to path to import recipe storage
sys.path.append(str(Path(__file__).parent.parent.parent))

from .sqlite_database import SQLiteDatabase
from .recipe_adapter import RecipeAdapter

logger = logging.getLogger(__name__)


class EnhancedRecipeService:
    """Enhanced recipe service with SQLite integration"""
    
    def __init__(self):
        self.data_path = Path("/app/data")
        self.recipes_path = self.data_path / "recipes"
        self.recipes_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        self.db = SQLiteDatabase(self.data_path)
        logger.info("Recipe service initialized with SQLite database")
    
    async def list_recipes(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> Dict[str, Any]:
        """List recipes with pagination from SQLite database"""
        try:
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Query recipes from database
            async with self.db.get_connection() as conn:
                # Count total recipes
                cursor = await conn.execute("SELECT COUNT(*) FROM recipes WHERE is_active = 1")
                total = (await cursor.fetchone())[0]
                
                # Get paginated recipes
                query = f"""
                    SELECT id, name, description, created_at, updated_at, 
                           steps, settings, usage_count, last_used
                    FROM recipes 
                    WHERE is_active = 1
                    ORDER BY {sort_by} {order.upper()}
                    LIMIT ? OFFSET ?
                """
                cursor = await conn.execute(query, (page_size, offset))
                rows = await cursor.fetchall()
                
                # Convert rows to recipe dicts
                all_recipes = []
                for row in rows:
                    recipe = {
                        'id': row[0],
                        'name': row[1],
                        'description': row[2] or '',
                        'created_at': row[3],
                        'updated_at': row[4],
                        'steps': json.loads(row[5]) if row[5] else [],
                        'operations': json.loads(row[5]) if row[5] else [],  # Compatibility
                        'settings': json.loads(row[6]) if row[6] else {},
                        'usage_count': row[7],
                        'last_used': row[8]
                    }
                    all_recipes.append(recipe)
            
            return {
                "recipes": all_recipes,
                "total": total,
                "page": page,
                "pageSize": page_size,
                "hasNext": (page * page_size) < total,
                "hasPrev": page > 1
            }
            
        except Exception as e:
            logger.error(f"Error listing recipes: {e}")
            return {
                "recipes": [],
                "total": 0,
                "page": page,
                "pageSize": page_size,
                "error": str(e)
            }
    
    async def get_recipe(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get recipe by ID from SQLite database"""
        try:
            async with self.db.get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT id, name, description, created_at, updated_at, 
                           steps, settings, usage_count, last_used
                    FROM recipes 
                    WHERE id = ? AND is_active = 1
                """, (recipe_id,))
                row = await cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'description': row[2] or '',
                        'created_at': row[3],
                        'updated_at': row[4],
                        'steps': json.loads(row[5]) if row[5] else [],
                        'operations': json.loads(row[5]) if row[5] else [],  # Compatibility
                        'settings': json.loads(row[6]) if row[6] else {},
                        'usage_count': row[7],
                        'last_used': row[8]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting recipe {recipe_id}: {e}")
            return None
    
    async def create_recipe(
        self,
        name: str,
        description: Optional[str],
        operations: List[Dict[str, Any]],
        style_preset: str = "natural",
        processing_config: Optional[Dict[str, Any]] = None,
        is_default: bool = False
    ) -> Dict[str, Any]:
        """Create a new recipe in SQLite database"""
        recipe_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Merge processing_config into settings
        settings = processing_config or {}
        settings['style_preset'] = style_preset
        
        try:
            async with self.db.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO recipes (id, name, description, steps, settings, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    recipe_id,
                    name,
                    description or "",
                    json.dumps(operations),
                    json.dumps(settings),
                    now,
                    now
                ))
                await conn.commit()
            
            return {
                "id": recipe_id,
                "name": name,
                "description": description or "",
                "steps": operations,
                "operations": operations,  # Compatibility
                "settings": settings,
                "created_at": now,
                "updated_at": now,
                "usage_count": 0,
                "last_used": None
            }
        except Exception as e:
            logger.error(f"Failed to create recipe: {e}")
            raise Exception(f"Failed to create recipe: {str(e)}")
    
    async def update_recipe(
        self,
        recipe_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        operations: Optional[List[Dict[str, Any]]] = None,
        style_preset: Optional[str] = None,
        processing_config: Optional[Dict[str, Any]] = None,
        is_default: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Update an existing recipe"""
        try:
            # First check if recipe exists
            existing = await self.get_recipe(recipe_id)
            if not existing:
                return None
            
            # Build update query dynamically
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            
            if operations is not None:
                updates.append("steps = ?")
                params.append(json.dumps(operations))
            
            if style_preset is not None or processing_config is not None:
                # Merge settings
                settings = existing.get('settings', {})
                if processing_config:
                    settings.update(processing_config)
                if style_preset:
                    settings['style_preset'] = style_preset
                updates.append("settings = ?")
                params.append(json.dumps(settings))
            
            if updates:
                updates.append("updated_at = ?")
                params.append(datetime.now().isoformat())
                
                # Add recipe_id to params
                params.append(recipe_id)
                
                async with self.db.get_connection() as conn:
                    query = f"UPDATE recipes SET {', '.join(updates)} WHERE id = ?"
                    await conn.execute(query, params)
                    await conn.commit()
            
            # Return updated recipe
            return await self.get_recipe(recipe_id)
            
        except Exception as e:
            logger.error(f"Failed to update recipe {recipe_id}: {e}")
            return None
    
    async def delete_recipe(self, recipe_id: str) -> bool:
        """Delete a recipe (soft delete by setting is_active = 0)"""
        try:
            async with self.db.get_connection() as conn:
                # Soft delete
                await conn.execute(
                    "UPDATE recipes SET is_active = 0, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), recipe_id)
                )
                await conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete recipe {recipe_id}: {e}")
            return False
    
    async def use_recipe(self, recipe_id: str) -> bool:
        """Update usage statistics when a recipe is used"""
        try:
            async with self.db.get_connection() as conn:
                await conn.execute("""
                    UPDATE recipes 
                    SET usage_count = usage_count + 1, 
                        last_used = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    recipe_id
                ))
                await conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update recipe usage for {recipe_id}: {e}")
            return False
    
    async def get_presets(self) -> List[Dict[str, Any]]:
        """Get recipe presets - returns empty list as we don't use presets"""
        return []
    
    # Adapter methods for compatibility with processing service
    async def get_recipe_for_processing(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get recipe formatted for processing service"""
        recipe = await self.get_recipe(recipe_id)
        if recipe:
            return RecipeAdapter.to_processing_format(recipe)
        return None
    
    def validate_recipe(self, recipe_data: Dict[str, Any]) -> bool:
        """Validate recipe structure"""
        required_fields = ['name', 'operations']
        return all(field in recipe_data for field in required_fields)


# Create singleton instance
recipe_service = EnhancedRecipeService()