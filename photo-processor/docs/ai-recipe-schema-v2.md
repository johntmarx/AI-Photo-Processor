# AI-Enhanced Recipe Schema v2

## Overview

The recipe schema has been extended to support AI model configuration, conditional logic, and complex workflows. Recipes are JSON documents that define a series of processing stages and steps.

## Complete Schema Definition

```typescript
interface Recipe {
  id: string;
  name: string;
  description: string;
  version: string;
  created_at: string;
  updated_at: string;
  
  // Global settings for the recipe
  global_settings: {
    gpu_memory_limit?: string;  // "6GB"
    max_processing_time?: number;  // seconds
    cache_models?: string[];  // ["rt-detr", "sam2"]
    fallback_mode?: "skip" | "basic" | "error";
  };
  
  // Variables that can be referenced throughout the recipe
  variables: {
    [key: string]: {
      type: "string" | "number" | "boolean" | "array";
      default?: any;
      from?: "metadata" | "user_input" | "previous_step";
      description?: string;
    };
  };
  
  // Processing stages executed in order
  stages: Stage[];
}

interface Stage {
  name: string;
  description?: string;
  enabled: boolean;
  
  // Execution control
  parallel?: boolean;  // Execute steps in parallel
  continue_on_error?: boolean;
  timeout_seconds?: number;
  depends_on?: string[];  // Other stage names
  
  // Conditional execution
  condition?: Condition;
  
  // Steps within this stage
  steps: Step[];
}

interface Step {
  name: string;
  model: "qwen2.5-vl" | "rt-detr" | "sam2" | "siglip" | "nima" | "traditional";
  action: string;  // Model-specific action
  enabled: boolean;
  
  // Model parameters
  parameters: {
    [key: string]: any;
  };
  
  // Input/Output handling
  input_from?: string;  // Previous step name
  output_to?: string;   // Variable name for results
  
  // Conditional execution
  condition?: Condition;
  
  // Iteration support
  iterate?: {
    over: string;  // Variable containing array
    as: string;    // Loop variable name
    max_iterations?: number;
  };
}

interface Condition {
  type: "simple" | "complex";
  
  // Simple condition
  variable?: string;
  operator?: "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains" | "exists";
  value?: any;
  
  // Complex condition
  and?: Condition[];
  or?: Condition[];
  not?: Condition;
}
```

## Model-Specific Parameters

### Qwen2.5-VL (Vision-Language Model)
```json
{
  "model": "qwen2.5-vl",
  "action": "analyze_scene",
  "parameters": {
    "prompt_template": "Analyze this ${photo_type} image...",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 0.9,
    "response_format": "structured",  // or "text"
    "extract_fields": ["subjects", "lighting", "suggestions"],
    "include_context": ["detected_objects", "quality_scores"]
  }
}
```

### RT-DETR (Object Detection)
```json
{
  "model": "rt-detr",
  "action": "detect_objects",
  "parameters": {
    "classes": ["person", "face", "car", "pet"],  // null for all classes
    "confidence_threshold": 0.6,
    "nms_threshold": 0.5,
    "max_detections": 20,
    "return_embeddings": false
  }
}
```

### SAM2 (Segmentation)
```json
{
  "model": "sam2",
  "action": "segment_objects",
  "parameters": {
    "prompt_type": "boxes",  // "points", "boxes", "masks", "text"
    "prompt_data": "${detected_objects.bboxes}",
    "quality": "high",  // "low", "medium", "high"
    "multimask_output": false,
    "refine_edges": true,
    "min_mask_area": 100  // pixels
  }
}
```

### SigLIP (Vision-Language Embeddings)
```json
{
  "model": "siglip",
  "action": "compute_similarity",
  "parameters": {
    "reference_image": "${current_image}",
    "compare_with": "${image_batch}",
    "text_queries": ["beautiful sunset", "professional photo"],
    "return_top_k": 5,
    "similarity_metric": "cosine"
  }
}
```

### NIMA (Quality Assessment)
```json
{
  "model": "nima",
  "action": "assess_quality",
  "parameters": {
    "aspects": ["technical", "aesthetic"],
    "return_percentile": true,
    "detailed_scores": true
  }
}
```

## Complete Recipe Examples

### 1. Portrait Photography Enhancement
```json
{
  "name": "Portrait Pro Enhancement",
  "version": "2.0",
  "description": "Optimized workflow for portrait photography with face enhancement",
  
  "variables": {
    "min_face_size": {
      "type": "number",
      "default": 50,
      "description": "Minimum face size in pixels"
    },
    "style_preference": {
      "type": "string",
      "default": "natural",
      "from": "user_input"
    }
  },
  
  "stages": [
    {
      "name": "quality_check",
      "steps": [
        {
          "name": "assess_quality",
          "model": "nima",
          "action": "assess_quality",
          "parameters": {
            "aspects": ["technical", "aesthetic"]
          },
          "output_to": "quality_scores"
        }
      ],
      "condition": {
        "variable": "quality_scores.technical",
        "operator": ">",
        "value": 5.0
      }
    },
    
    {
      "name": "face_detection",
      "steps": [
        {
          "name": "detect_faces",
          "model": "rt-detr",
          "action": "detect_objects",
          "parameters": {
            "classes": ["face"],
            "confidence_threshold": 0.8
          },
          "output_to": "detected_faces"
        }
      ]
    },
    
    {
      "name": "face_enhancement",
      "condition": {
        "variable": "detected_faces.count",
        "operator": ">",
        "value": 0
      },
      "steps": [
        {
          "name": "segment_faces",
          "model": "sam2",
          "action": "segment_objects",
          "parameters": {
            "prompt_type": "boxes",
            "prompt_data": "${detected_faces.bboxes}",
            "quality": "high"
          },
          "output_to": "face_masks"
        },
        
        {
          "name": "analyze_portrait",
          "model": "qwen2.5-vl",
          "action": "analyze_scene",
          "parameters": {
            "prompt_template": "Analyze this portrait. Consider the ${detected_faces.count} faces detected. Suggest improvements for ${style_preference} style. Focus on: lighting, skin tones, composition.",
            "extract_fields": ["lighting_issues", "skin_tone_adjustments", "composition_suggestions"]
          },
          "output_to": "portrait_analysis"
        }
      ]
    }
  ]
}
```

### 2. Event Photography Burst Selection
```json
{
  "name": "Burst Shot Selection",
  "version": "1.0",
  "description": "Select best photos from rapid sequences",
  
  "global_settings": {
    "cache_models": ["siglip", "nima"],
    "max_processing_time": 60
  },
  
  "stages": [
    {
      "name": "group_similar",
      "parallel": true,
      "steps": [
        {
          "name": "find_similar",
          "model": "siglip",
          "action": "find_similar",
          "parameters": {
            "time_window_seconds": 10,
            "similarity_threshold": 0.9,
            "group_by": "visual_similarity"
          },
          "output_to": "similar_groups"
        }
      ]
    },
    
    {
      "name": "select_best",
      "steps": [
        {
          "name": "iterate_groups",
          "model": "nima",
          "action": "assess_quality",
          "iterate": {
            "over": "similar_groups",
            "as": "group"
          },
          "parameters": {
            "aspects": ["technical", "aesthetic"]
          },
          "output_to": "group_qualities"
        },
        
        {
          "name": "ai_selection",
          "model": "qwen2.5-vl",
          "action": "compare_images",
          "parameters": {
            "prompt_template": "Compare these ${group.count} similar photos. Select the best one considering: sharpness, expressions, timing, composition.",
            "images": "${group.images}",
            "return_ranking": true
          },
          "output_to": "ai_rankings"
        }
      ]
    }
  ]
}
```

### 3. Conditional Wildlife Photography
```json
{
  "name": "Wildlife Auto-Enhancement",
  "version": "1.5",
  
  "stages": [
    {
      "name": "detect_wildlife",
      "steps": [
        {
          "name": "detect_animals",
          "model": "rt-detr",
          "action": "detect_objects",
          "parameters": {
            "classes": ["bird", "animal", "wildlife"],
            "confidence_threshold": 0.5
          },
          "output_to": "wildlife_detections"
        }
      ]
    },
    
    {
      "name": "wildlife_processing",
      "condition": {
        "variable": "wildlife_detections.count",
        "operator": ">",
        "value": 0
      },
      "steps": [
        {
          "name": "isolate_subject",
          "model": "sam2",
          "action": "segment_primary",
          "parameters": {
            "subject": "${wildlife_detections.largest}",
            "quality": "high",
            "expand_mask": 10
          }
        },
        
        {
          "name": "analyze_shot",
          "model": "qwen2.5-vl",
          "action": "wildlife_analysis",
          "parameters": {
            "prompt_template": "Analyze this wildlife photo of ${wildlife_detections.classes}. Evaluate: focus quality, subject positioning, background distraction.",
            "specialized_model": true
          }
        }
      ]
    },
    
    {
      "name": "no_wildlife_fallback",
      "condition": {
        "variable": "wildlife_detections.count",
        "operator": "==",
        "value": 0
      },
      "steps": [
        {
          "name": "landscape_mode",
          "model": "nima",
          "action": "assess_landscape",
          "parameters": {
            "aspects": ["composition", "colors", "sharpness"]
          }
        }
      ]
    }
  ]
}
```

## Advanced Features

### 1. Dynamic Variables
```json
{
  "variables": {
    "time_of_day": {
      "type": "string",
      "from": "metadata",
      "default": "unknown"
    },
    "detected_count": {
      "type": "number",
      "from": "previous_step",
      "source": "detect_objects.count"
    }
  }
}
```

### 2. Complex Conditions
```json
{
  "condition": {
    "type": "complex",
    "and": [
      {
        "variable": "quality_scores.technical",
        "operator": ">",
        "value": 7
      },
      {
        "or": [
          {
            "variable": "detected_faces.count",
            "operator": ">",
            "value": 0
          },
          {
            "variable": "scene_type",
            "operator": "==",
            "value": "portrait"
          }
        ]
      }
    ]
  }
}
```

### 3. Iterative Enhancement
```json
{
  "name": "iterative_quality_improvement",
  "steps": [
    {
      "name": "enhance_until_good",
      "model": "traditional",
      "action": "auto_enhance",
      "parameters": {
        "adjustments": ["exposure", "contrast", "saturation"]
      },
      "iterate": {
        "max_iterations": 3,
        "until_condition": {
          "variable": "quality_scores.technical",
          "operator": ">=",
          "value": 7.5
        }
      }
    }
  ]
}
```

## Recipe Validation Rules

1. **Model Availability**: Specified models must be installed
2. **Parameter Types**: Parameters must match model expectations
3. **Variable References**: All ${var} references must be defined
4. **Circular Dependencies**: Stages cannot have circular dependencies
5. **Output Naming**: Output variable names must be unique
6. **Condition Logic**: Conditions must be evaluable

## Migration from v1 Recipes

Old v1 recipes can be automatically migrated to v2 format:
- Traditional processing steps remain unchanged
- New AI stages can be inserted at any point
- Existing variables are preserved
- Backward compatibility maintained

---

This schema provides the flexibility to create simple one-step recipes or complex multi-stage workflows with conditional logic and AI model orchestration.