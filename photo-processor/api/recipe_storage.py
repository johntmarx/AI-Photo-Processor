"""
Recipe Storage System for Photo Processing

This module handles storing and retrieving processing recipes that allow
for reproducible photo processing and maintaining original files.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessingOperation:
    """Individual processing operation"""
    type: str  # crop, rotate, enhance, etc.
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = 'ai'  # 'ai', 'user', 'preset'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingOperation':
        return cls(
            type=data['type'],
            parameters=data['parameters'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data.get('source', 'ai')
        )


@dataclass
class ProcessingRecipe:
    """Complete recipe for reproducing processing"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_hash: str = ""
    original_filename: str = ""
    operations: List[ProcessingOperation] = field(default_factory=list)
    ai_metadata: Dict[str, Any] = field(default_factory=dict)
    user_overrides: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        """Convert recipe to JSON string"""
        data = {
            'id': self.id,
            'original_hash': self.original_hash,
            'original_filename': self.original_filename,
            'operations': [op.to_dict() for op in self.operations],
            'ai_metadata': self.ai_metadata,
            'user_overrides': self.user_overrides,
            'version': self.version,
            'created_at': self.created_at.isoformat()
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProcessingRecipe':
        """Create recipe from JSON string"""
        data = json.loads(json_str)
        return cls(
            id=data['id'],
            original_hash=data['original_hash'],
            original_filename=data['original_filename'],
            operations=[ProcessingOperation.from_dict(op) for op in data['operations']],
            ai_metadata=data['ai_metadata'],
            user_overrides=data['user_overrides'],
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at'])
        )
    
    def add_operation(self, op_type: str, parameters: Dict[str, Any], source: str = 'ai'):
        """Add a processing operation to the recipe"""
        operation = ProcessingOperation(
            type=op_type,
            parameters=parameters,
            source=source
        )
        self.operations.append(operation)
        
    def get_description(self) -> str:
        """Generate human-readable description of the recipe"""
        desc_lines = [f"Processing Recipe v{self.version}"]
        desc_lines.append(f"Original: {self.original_filename}")
        desc_lines.append(f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M')}")
        desc_lines.append("\nOperations:")
        
        for i, op in enumerate(self.operations, 1):
            desc_lines.append(f"{i}. {op.type}: {self._describe_operation(op)}")
        
        return "\n".join(desc_lines)
    
    def _describe_operation(self, op: ProcessingOperation) -> str:
        """Generate human-readable description of an operation"""
        if op.type == 'crop':
            p = op.parameters
            return f"Crop to ({p.get('x1', 0):.1f}, {p.get('y1', 0):.1f}, {p.get('x2', 1):.1f}, {p.get('y2', 1):.1f})"
        elif op.type == 'rotate':
            return f"Rotate {op.parameters.get('angle', 0)}°"
        elif op.type == 'enhance':
            params = []
            for key, value in op.parameters.items():
                if value != 0:
                    params.append(f"{key}={value:+.2f}")
            return ", ".join(params) if params else "Auto enhance"
        else:
            return str(op.parameters)


class RecipeStorage:
    """Manages recipe storage and retrieval"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "recipe_index.json"
        self.index = self._load_index()
        
        # Create index file if it doesn't exist
        if not self.index_file.exists():
            self._save_index()
        
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load recipe index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load recipe index: {e}")
        return {}
    
    def _save_index(self):
        """Save recipe index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save recipe index: {e}")
    
    def save_recipe(self, recipe: ProcessingRecipe) -> bool:
        """Save a recipe to storage"""
        try:
            # Save recipe file
            recipe_file = self.storage_path / f"{recipe.id}.json"
            with open(recipe_file, 'w') as f:
                f.write(recipe.to_json())
            
            # Update index
            self.index[recipe.id] = {
                'original_hash': recipe.original_hash,
                'original_filename': recipe.original_filename,
                'created_at': recipe.created_at.isoformat(),
                'version': recipe.version,
                'operations_count': len(recipe.operations)
            }
            self._save_index()
            
            logger.info(f"Saved recipe {recipe.id} for {recipe.original_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save recipe: {e}")
            return False
    
    def load_recipe(self, recipe_id: str) -> Optional[ProcessingRecipe]:
        """Load a recipe by ID"""
        try:
            recipe_file = self.storage_path / f"{recipe_id}.json"
            if recipe_file.exists():
                with open(recipe_file, 'r') as f:
                    return ProcessingRecipe.from_json(f.read())
        except Exception as e:
            logger.error(f"Failed to load recipe {recipe_id}: {e}")
        return None
    
    def find_recipe_by_hash(self, file_hash: str) -> Optional[ProcessingRecipe]:
        """Find the most recent recipe for a file hash"""
        matching_recipes = [
            (rid, info) for rid, info in self.index.items()
            if info['original_hash'] == file_hash
        ]
        
        if not matching_recipes:
            return None
        
        # Sort by creation date and version, get most recent
        matching_recipes.sort(
            key=lambda x: (x[1]['created_at'], x[1]['version']),
            reverse=True
        )
        
        recipe_id = matching_recipes[0][0]
        return self.load_recipe(recipe_id)
    
    def list_recipes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent recipes"""
        recipes = []
        for recipe_id, info in self.index.items():
            recipes.append({
                'id': recipe_id,
                **info
            })
        
        # Sort by creation date
        recipes.sort(key=lambda x: x['created_at'], reverse=True)
        return recipes[:limit]