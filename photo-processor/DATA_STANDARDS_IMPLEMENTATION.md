# Data Standards Implementation Guide

## IMMEDIATE FIXES NEEDED

Based on the DATA_STANDARDS.md document, here are the critical fixes needed RIGHT NOW:

### 1. Fix QueueStatus Type Mismatch (CRITICAL - Breaks Processing Page)

**Problem**: Frontend expects numbers but API returns arrays

**Backend Fix** (`/api/routes/processing.py`):
```python
@router.get("/queue")
async def get_queue_status():
    status = await processing_service.get_queue_status()
    # Transform to match frontend expectations
    return {
        "pending": status.pending,  # Keep as array
        "processing": status.processing,  # Keep as array
        "completed": status.completed,  # Keep as array
        "isPaused": status.is_paused,  # Convert to camelCase
        "stats": {
            "isPaused": status.stats.is_paused,
            "currentPhoto": status.stats.current_photo,
            "queueLength": status.stats.queue_length,
            "processingRate": status.stats.processing_rate,
            "averageTime": status.stats.average_time,
            "errorsToday": status.stats.errors_today
        }
    }
```

**Frontend Fix** (Already partially done in Processing.tsx)

### 2. Fix Recipe Operations vs Steps

**Problem**: Backend uses "operations", frontend uses "steps"

**API Transformation** (`/api/routes/recipes.py`):
```python
def transform_recipe_for_frontend(recipe):
    return {
        "id": recipe.id,
        "name": recipe.name,
        "description": recipe.description,
        "steps": recipe.operations,  # Transform field name
        "createdAt": recipe.created_at.isoformat(),
        "updatedAt": recipe.updated_at.isoformat(),
        "isPreset": recipe.is_preset,
        "usageCount": recipe.usage_count,
        "category": recipe.category
    }
```

### 3. Add Transformation Middleware

Create `/api/middleware/transform.py`:
```python
from datetime import datetime
from typing import Any, Dict
import re

def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase"""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def transform_dict_keys(obj: Any) -> Any:
    """Recursively transform dictionary keys from snake_case to camelCase"""
    if isinstance(obj, dict):
        return {
            snake_to_camel(k): transform_dict_keys(v) 
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [transform_dict_keys(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def backend_to_frontend(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform backend response to frontend format"""
    return transform_dict_keys(data)
```

### 4. Fix Photo Model Fields

**Backend Service** (`/api/services/photo_service_v2.py`):
Ensure all photo responses include the new fields:
- `thumbnail_path`
- `web_path`
- `recipe_used`

### 5. Fix WebSocket Event Consistency

**WebSocket Manager** (`/api/services/websocket_manager.py`):
Ensure ALL events follow the standard format:
```python
async def send_event(self, event_type: str, data: Any):
    event = {
        "type": event_type,
        "data": transform_dict_keys(data),  # Transform to camelCase
        "timestamp": datetime.utcnow().isoformat()
    }
    await self.broadcast(event)
```

## TESTING CHECKLIST

After implementing fixes:

1. **Processing Page**:
   - [ ] Queue counts display correctly
   - [ ] Current item shows processing details
   - [ ] No "Objects are not valid as React child" errors

2. **Recipe Editor**:
   - [ ] Can create new recipes
   - [ ] Steps/operations save correctly
   - [ ] Recipe list displays properly

3. **Photo Grid**:
   - [ ] Photos display with correct fields
   - [ ] Batch upload works
   - [ ] Delete functionality works

4. **WebSocket Events**:
   - [ ] Real-time updates work
   - [ ] No console errors about data structure

## GRADUAL MIGRATION STRATEGY

1. **Phase 1** (Immediate):
   - Add transformation layer in API routes
   - Fix critical type mismatches
   - No breaking changes

2. **Phase 2** (Next Week):
   - Update backend models to match standards
   - Add proper serialization
   - Update tests

3. **Phase 3** (Next Month):
   - Remove transformation layer
   - Direct model-to-frontend mapping
   - Full compliance with standards

## ENFORCEMENT

Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: check-data-standards
      name: Check Data Standards Compliance
      entry: python scripts/check_data_standards.py
      language: python
      files: \.(py|ts|tsx)$
```

Create `scripts/check_data_standards.py` to validate compliance.