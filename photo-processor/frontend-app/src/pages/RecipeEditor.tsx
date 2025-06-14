import { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd';
import { Plus, Trash2, GripVertical, Save, AlertCircle, Check } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { recipesApi } from '../services/api';

// Types for recipe and processing steps
interface ProcessingStep {
  id: string;
  type: ProcessingStepType;
  enabled: boolean;
  parameters: Record<string, any>;
}

type ProcessingStepType = 
  | 'crop'
  | 'rotate'
  | 'adjust_exposure'
  | 'adjust_contrast'
  | 'adjust_saturation'
  | 'adjust_highlights'
  | 'adjust_shadows'
  | 'adjust_whites'
  | 'adjust_blacks'
  | 'adjust_vibrance'
  | 'adjust_clarity'
  | 'apply_lut'
  | 'denoise'
  | 'sharpen'
  | 'lens_correction'
  | 'perspective_correction'
  | 'color_grading';

interface StylePreset {
  id: string;
  name: string;
  description: string;
}

interface Recipe {
  id?: string;
  name: string;
  description: string;
  isDraft: boolean;
  processingSteps: ProcessingStep[];
  settings: {
    stylePreset: string;
    qualityThreshold: number;
    culling: {
      enabled: boolean;
      minScore: number;
      groupSimilar: boolean;
      similarityThreshold: number;
    };
    export: {
      format: 'jpeg' | 'png' | 'webp';
      quality: number;
      maxDimension?: number;
      preserveMetadata: boolean;
    };
    aiModels: {
      enhancementModel: string;
      enhancementStrength: number;
      objectDetection: boolean;
      sceneAnalysis: boolean;
      faceEnhancement: boolean;
    };
  };
}

const STYLE_PRESETS: StylePreset[] = [
  { id: 'natural', name: 'Natural', description: 'Balanced, true-to-life colors' },
  { id: 'vivid', name: 'Vivid', description: 'Enhanced colors and contrast' },
  { id: 'monochrome', name: 'Monochrome', description: 'Black and white conversion' },
  { id: 'vintage', name: 'Vintage', description: 'Retro film-inspired look' },
  { id: 'cinematic', name: 'Cinematic', description: 'Movie-like color grading' },
  { id: 'portrait', name: 'Portrait', description: 'Optimized for people' },
  { id: 'landscape', name: 'Landscape', description: 'Enhanced for nature scenes' },
];

const PROCESSING_STEP_TEMPLATES: Record<ProcessingStepType, { name: string; defaultParams: Record<string, any> }> = {
  crop: { name: 'Crop', defaultParams: { aspectRatio: 'original', customRatio: null } },
  rotate: { name: 'Rotate', defaultParams: { angle: 0, autoStraighten: false } },
  adjust_exposure: { name: 'Adjust Exposure', defaultParams: { value: 0 } },
  adjust_contrast: { name: 'Adjust Contrast', defaultParams: { value: 0 } },
  adjust_saturation: { name: 'Adjust Saturation', defaultParams: { value: 0 } },
  adjust_highlights: { name: 'Adjust Highlights', defaultParams: { value: 0 } },
  adjust_shadows: { name: 'Adjust Shadows', defaultParams: { value: 0 } },
  adjust_whites: { name: 'Adjust Whites', defaultParams: { value: 0 } },
  adjust_blacks: { name: 'Adjust Blacks', defaultParams: { value: 0 } },
  adjust_vibrance: { name: 'Adjust Vibrance', defaultParams: { value: 0 } },
  adjust_clarity: { name: 'Adjust Clarity', defaultParams: { value: 0 } },
  apply_lut: { name: 'Apply LUT', defaultParams: { lutFile: '', intensity: 100 } },
  denoise: { name: 'Denoise', defaultParams: { strength: 50, preserveDetail: true } },
  sharpen: { name: 'Sharpen', defaultParams: { amount: 50, radius: 1, threshold: 0 } },
  lens_correction: { name: 'Lens Correction', defaultParams: { autoCorrect: true, vignetteAmount: 0, distortionAmount: 0 } },
  perspective_correction: { name: 'Perspective Correction', defaultParams: { auto: true, vertical: 0, horizontal: 0 } },
  color_grading: { name: 'Color Grading', defaultParams: { shadows: { r: 0, g: 0, b: 0 }, midtones: { r: 0, g: 0, b: 0 }, highlights: { r: 0, g: 0, b: 0 } } },
};

export default function RecipeEditor() {
  const navigate = useNavigate();
  const { id } = useParams();
  const [recipe, setRecipe] = useState<Recipe>({
    name: '',
    description: '',
    isDraft: true,
    processingSteps: [],
    settings: {
      stylePreset: 'natural',
      qualityThreshold: 80,
      culling: {
        enabled: false,
        minScore: 70,
        groupSimilar: true,
        similarityThreshold: 85,
      },
      export: {
        format: 'jpeg',
        quality: 90,
        maxDimension: undefined,
        preserveMetadata: true,
      },
      aiModels: {
        enhancementModel: 'standard',
        enhancementStrength: 50,
        objectDetection: false,
        sceneAnalysis: true,
        faceEnhancement: false,
      },
    },
  });

  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [showStepSelector, setShowStepSelector] = useState(false);

  // Load existing recipe if editing
  useEffect(() => {
    if (id) {
      loadRecipe(id);
    }
  }, [id]);

  // Auto-save draft every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (recipe.isDraft && recipe.name) {
        saveDraft();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [recipe]);

  const loadRecipe = async (recipeId: string) => {
    try {
      const response = await recipesApi.get(recipeId);
      const apiRecipe = response.data as any;
      
      // Convert API recipe format to internal format
      const loadedRecipe: Recipe = {
        id: apiRecipe.id,
        name: apiRecipe.name,
        description: apiRecipe.description,
        isDraft: apiRecipe.processing_config?.isDraft || false,
        processingSteps: apiRecipe.steps.map((step: any, index: number) => ({
          id: `step-${index}`,
          type: step.operation,
          enabled: step.enabled,
          parameters: step.parameters || {}
        })),
        settings: {
          stylePreset: apiRecipe.processing_config?.stylePreset || 'natural',
          qualityThreshold: apiRecipe.processing_config?.qualityThreshold || 70,
          culling: apiRecipe.processing_config?.culling || {
            enabled: false,
            minScore: 50,
            groupSimilar: true,
            similarityThreshold: 90
          },
          export: apiRecipe.processing_config?.export || {
            format: 'jpeg',
            quality: 95,
            maxDimension: null
          },
          aiModels: apiRecipe.processing_config?.aiModels || {
            enhancementModel: 'standard',
            enhancementStrength: 50,
            detectObjects: true,
            analyzeScene: true,
            faceEnhancement: true
          }
        }
      };
      
      setRecipe(loadedRecipe);
    } catch (error) {
      console.error('Failed to load recipe:', error);
    }
  };

  const saveDraft = async () => {
    try {
      // Use the same save function with draft flag
      await handleSave(true);
    } catch (error) {
      console.error('Failed to save draft:', error);
    }
  };

  const validateRecipe = (): string[] => {
    const errors: string[] = [];

    if (!recipe.name.trim()) {
      errors.push('Recipe name is required');
    }

    if (!recipe.description.trim()) {
      errors.push('Recipe description is required');
    }

    if (recipe.processingSteps.length === 0) {
      errors.push('At least one processing step is required');
    }

    // Validate processing steps
    recipe.processingSteps.forEach((step, index) => {
      if (step.type === 'apply_lut' && !step.parameters.lutFile) {
        errors.push(`Processing step ${index + 1} (Apply LUT): LUT file is required`);
      }
    });

    // Validate export settings
    if (recipe.settings.export.quality < 1 || recipe.settings.export.quality > 100) {
      errors.push('Export quality must be between 1 and 100');
    }

    // Validate culling settings
    if (recipe.settings.culling.enabled) {
      if (recipe.settings.culling.minScore < 0 || recipe.settings.culling.minScore > 100) {
        errors.push('Culling minimum score must be between 0 and 100');
      }
      if (recipe.settings.culling.similarityThreshold < 0 || recipe.settings.culling.similarityThreshold > 100) {
        errors.push('Culling similarity threshold must be between 0 and 100');
      }
    }

    return errors;
  };

  const handleSave = async (asDraft: boolean = false) => {
    if (!asDraft) {
      const errors = validateRecipe();
      if (errors.length > 0) {
        setValidationErrors(errors);
        return;
      }
    }

    setIsSaving(true);
    setValidationErrors([]);

    try {
      // Convert internal recipe format to API format
      const apiRecipe = {
        name: recipe.name,
        description: recipe.description,
        steps: recipe.processingSteps.map(step => ({
          operation: step.type,
          parameters: step.parameters,
          enabled: step.enabled
        })),
        is_preset: false,
        // Add custom fields for our settings
        processing_config: {
          isDraft: asDraft,
          stylePreset: recipe.settings.stylePreset,
          qualityThreshold: recipe.settings.qualityThreshold,
          culling: recipe.settings.culling,
          export: recipe.settings.export,
          aiModels: recipe.settings.aiModels
        }
      };
      
      const response = recipe.id 
        ? await recipesApi.update(recipe.id, apiRecipe)
        : await recipesApi.create(apiRecipe);
      
      if (!recipe.id) {
        setRecipe({ ...recipe, id: response.data.id });
      }
      
      setLastSaved(new Date());
      
      if (!asDraft) {
        navigate('/recipes');
      }
    } catch (error) {
      console.error('Failed to save recipe:', error);
      setValidationErrors(['Failed to save recipe. Please try again.']);
    } finally {
      setIsSaving(false);
    }
  };

  const addProcessingStep = (type: ProcessingStepType) => {
    const template = PROCESSING_STEP_TEMPLATES[type];
    const newStep: ProcessingStep = {
      id: `step-${Date.now()}`,
      type,
      enabled: true,
      parameters: { ...template.defaultParams },
    };

    setRecipe({
      ...recipe,
      processingSteps: [...recipe.processingSteps, newStep],
    });
    setShowStepSelector(false);
  };

  const updateProcessingStep = (stepId: string, updates: Partial<ProcessingStep>) => {
    setRecipe({
      ...recipe,
      processingSteps: recipe.processingSteps.map(step =>
        step.id === stepId ? { ...step, ...updates } : step
      ),
    });
  };

  const removeProcessingStep = (stepId: string) => {
    setRecipe({
      ...recipe,
      processingSteps: recipe.processingSteps.filter(step => step.id !== stepId),
    });
  };

  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return;

    const items = Array.from(recipe.processingSteps);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setRecipe({
      ...recipe,
      processingSteps: items,
    });
  };

  const renderStepParameters = (step: ProcessingStep) => {
    switch (step.type) {
      case 'crop':
        return (
          <div className="space-y-2">
            <select
              value={step.parameters.aspectRatio}
              onChange={(e) => updateProcessingStep(step.id, {
                parameters: { ...step.parameters, aspectRatio: e.target.value }
              })}
              className="w-full px-3 py-2 border rounded-md"
            >
              <option value="original">Original</option>
              <option value="1:1">Square (1:1)</option>
              <option value="4:3">4:3</option>
              <option value="16:9">16:9</option>
              <option value="custom">Custom</option>
            </select>
            {step.parameters.aspectRatio === 'custom' && (
              <Input
                type="text"
                placeholder="e.g., 21:9"
                value={step.parameters.customRatio || ''}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, customRatio: e.target.value }
                })}
              />
            )}
          </div>
        );

      case 'rotate':
        return (
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Input
                type="number"
                value={step.parameters.angle}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, angle: parseFloat(e.target.value) || 0 }
                })}
                min="-180"
                max="180"
                step="0.1"
                className="flex-1"
              />
              <span className="text-sm text-gray-600">degrees</span>
            </div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={step.parameters.autoStraighten}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, autoStraighten: e.target.checked }
                })}
              />
              <span className="text-sm">Auto-straighten</span>
            </label>
          </div>
        );

      case 'adjust_exposure':
      case 'adjust_contrast':
      case 'adjust_saturation':
      case 'adjust_highlights':
      case 'adjust_shadows':
      case 'adjust_whites':
      case 'adjust_blacks':
      case 'adjust_vibrance':
      case 'adjust_clarity':
        return (
          <div className="space-y-2">
            <input
              type="range"
              min="-100"
              max="100"
              value={step.parameters.value}
              onChange={(e) => updateProcessingStep(step.id, {
                parameters: { ...step.parameters, value: parseInt(e.target.value) }
              })}
              className="w-full"
            />
            <div className="text-center text-sm text-gray-600">{step.parameters.value}</div>
          </div>
        );

      case 'denoise':
        return (
          <div className="space-y-2">
            <div>
              <label className="text-sm font-medium">Strength</label>
              <input
                type="range"
                min="0"
                max="100"
                value={step.parameters.strength}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, strength: parseInt(e.target.value) }
                })}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600">{step.parameters.strength}%</div>
            </div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={step.parameters.preserveDetail}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, preserveDetail: e.target.checked }
                })}
              />
              <span className="text-sm">Preserve detail</span>
            </label>
          </div>
        );

      case 'sharpen':
        return (
          <div className="space-y-2">
            <div>
              <label className="text-sm font-medium">Amount</label>
              <input
                type="range"
                min="0"
                max="200"
                value={step.parameters.amount}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, amount: parseInt(e.target.value) }
                })}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600">{step.parameters.amount}%</div>
            </div>
            <div>
              <label className="text-sm font-medium">Radius</label>
              <Input
                type="number"
                value={step.parameters.radius}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, radius: parseFloat(e.target.value) || 1 }
                })}
                min="0.1"
                max="5"
                step="0.1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Threshold</label>
              <Input
                type="number"
                value={step.parameters.threshold}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, threshold: parseInt(e.target.value) || 0 }
                })}
                min="0"
                max="255"
              />
            </div>
          </div>
        );

      case 'apply_lut':
        return (
          <div className="space-y-2">
            <div>
              <label className="text-sm font-medium">LUT File</label>
              <Input
                type="text"
                placeholder="Select or enter LUT file path"
                value={step.parameters.lutFile}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, lutFile: e.target.value }
                })}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Intensity</label>
              <input
                type="range"
                min="0"
                max="100"
                value={step.parameters.intensity}
                onChange={(e) => updateProcessingStep(step.id, {
                  parameters: { ...step.parameters, intensity: parseInt(e.target.value) }
                })}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600">{step.parameters.intensity}%</div>
            </div>
          </div>
        );

      case 'color_grading':
        return (
          <div className="space-y-4">
            {(['shadows', 'midtones', 'highlights'] as const).map((tone) => (
              <div key={tone}>
                <h5 className="text-sm font-medium capitalize mb-2">{tone}</h5>
                <div className="grid grid-cols-3 gap-2">
                  {(['r', 'g', 'b'] as const).map((channel) => (
                    <div key={channel}>
                      <label className="text-xs text-gray-600 uppercase">{channel}</label>
                      <input
                        type="range"
                        min="-100"
                        max="100"
                        value={step.parameters[tone][channel]}
                        onChange={(e) => updateProcessingStep(step.id, {
                          parameters: {
                            ...step.parameters,
                            [tone]: {
                              ...step.parameters[tone],
                              [channel]: parseInt(e.target.value)
                            }
                          }
                        })}
                        className="w-full"
                      />
                      <div className="text-center text-xs text-gray-600">
                        {step.parameters[tone][channel]}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        );

      default:
        return <div className="text-sm text-gray-500">No parameters for this step</div>;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">
          {id ? 'Edit Recipe' : 'Create New Recipe'}
        </h1>
        <div className="flex items-center space-x-4">
          {recipe.isDraft && (
            <Badge variant="secondary">Draft</Badge>
          )}
          {lastSaved && (
            <span className="text-sm text-gray-600">
              Last saved: {lastSaved.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {validationErrors.length > 0 && (
        <Card className="mb-6 p-4 bg-red-50 border-red-200">
          <div className="flex items-start space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5" />
            <div>
              <h3 className="font-medium text-red-900">Please fix the following errors:</h3>
              <ul className="mt-2 space-y-1">
                {validationErrors.map((error, index) => (
                  <li key={index} className="text-sm text-red-700">• {error}</li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      )}

      <div className="space-y-6">
        {/* Basic Information */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Basic Information</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Recipe Name</label>
              <Input
                type="text"
                value={recipe.name}
                onChange={(e) => setRecipe({ ...recipe, name: e.target.value })}
                placeholder="e.g., Portrait Enhancement"
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Description</label>
              <textarea
                value={recipe.description}
                onChange={(e) => setRecipe({ ...recipe, description: e.target.value })}
                placeholder="Describe what this recipe does..."
                className="w-full px-3 py-2 border rounded-md resize-none"
                rows={3}
              />
            </div>
          </div>
        </Card>

        {/* Style Preset */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Style Preset</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {STYLE_PRESETS.map((preset) => (
              <button
                key={preset.id}
                onClick={() => setRecipe({
                  ...recipe,
                  settings: { ...recipe.settings, stylePreset: preset.id }
                })}
                className={`p-3 rounded-lg border-2 text-left transition-colors ${
                  recipe.settings.stylePreset === preset.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="font-medium">{preset.name}</div>
                <div className="text-sm text-gray-600">{preset.description}</div>
              </button>
            ))}
          </div>
        </Card>

        {/* Processing Steps */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Processing Steps</h2>
            <Button
              onClick={() => setShowStepSelector(!showStepSelector)}
              size="sm"
            >
              <Plus className="h-4 w-4 mr-1" />
              Add Step
            </Button>
          </div>

          {showStepSelector && (
            <div className="mb-4 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium mb-2">Select a processing step:</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {Object.entries(PROCESSING_STEP_TEMPLATES).map(([type, template]) => (
                  <Button
                    key={type}
                    variant="outline"
                    size="sm"
                    onClick={() => addProcessingStep(type as ProcessingStepType)}
                  >
                    {template.name}
                  </Button>
                ))}
              </div>
            </div>
          )}

          <DragDropContext onDragEnd={handleDragEnd}>
            <Droppable droppableId="steps">
              {(provided) => (
                <div {...provided.droppableProps} ref={provided.innerRef} className="space-y-3">
                  {recipe.processingSteps.map((step, index) => (
                    <Draggable key={step.id} draggableId={step.id} index={index}>
                      {(provided, snapshot) => (
                        <div
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          className={`p-4 bg-white border rounded-lg ${
                            snapshot.isDragging ? 'shadow-lg' : ''
                          }`}
                        >
                          <div className="flex items-start space-x-3">
                            <div
                              {...provided.dragHandleProps}
                              className="mt-1 cursor-move text-gray-400 hover:text-gray-600"
                            >
                              <GripVertical className="h-5 w-5" />
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center space-x-3">
                                  <h4 className="font-medium">
                                    {PROCESSING_STEP_TEMPLATES[step.type].name}
                                  </h4>
                                  <label className="flex items-center space-x-2">
                                    <input
                                      type="checkbox"
                                      checked={step.enabled}
                                      onChange={(e) => updateProcessingStep(step.id, {
                                        enabled: e.target.checked
                                      })}
                                    />
                                    <span className="text-sm">Enabled</span>
                                  </label>
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => removeProcessingStep(step.id)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                              {step.enabled && (
                                <div className="pl-4">
                                  {renderStepParameters(step)}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </Draggable>
                  ))}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>

          {recipe.processingSteps.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No processing steps added yet. Click "Add Step" to get started.
            </div>
          )}
        </Card>

        {/* Quality & Culling Settings */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Quality & Culling Settings</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Quality Threshold</label>
              <div className="flex items-center space-x-3">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={recipe.settings.qualityThreshold}
                  onChange={(e) => setRecipe({
                    ...recipe,
                    settings: {
                      ...recipe.settings,
                      qualityThreshold: parseInt(e.target.value)
                    }
                  })}
                  className="flex-1"
                />
                <span className="text-sm font-medium w-12 text-right">
                  {recipe.settings.qualityThreshold}%
                </span>
              </div>
            </div>

            <div className="border-t pt-4">
              <label className="flex items-center space-x-2 mb-3">
                <input
                  type="checkbox"
                  checked={recipe.settings.culling.enabled}
                  onChange={(e) => setRecipe({
                    ...recipe,
                    settings: {
                      ...recipe.settings,
                      culling: {
                        ...recipe.settings.culling,
                        enabled: e.target.checked
                      }
                    }
                  })}
                />
                <span className="font-medium">Enable Culling</span>
              </label>

              {recipe.settings.culling.enabled && (
                <div className="space-y-3 pl-6">
                  <div>
                    <label className="block text-sm font-medium mb-1">Minimum Score</label>
                    <div className="flex items-center space-x-3">
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={recipe.settings.culling.minScore}
                        onChange={(e) => setRecipe({
                          ...recipe,
                          settings: {
                            ...recipe.settings,
                            culling: {
                              ...recipe.settings.culling,
                              minScore: parseInt(e.target.value)
                            }
                          }
                        })}
                        className="flex-1"
                      />
                      <span className="text-sm font-medium w-12 text-right">
                        {recipe.settings.culling.minScore}%
                      </span>
                    </div>
                  </div>

                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={recipe.settings.culling.groupSimilar}
                      onChange={(e) => setRecipe({
                        ...recipe,
                        settings: {
                          ...recipe.settings,
                          culling: {
                            ...recipe.settings.culling,
                            groupSimilar: e.target.checked
                          }
                        }
                      })}
                    />
                    <span className="text-sm">Group similar photos</span>
                  </label>

                  {recipe.settings.culling.groupSimilar && (
                    <div>
                      <label className="block text-sm font-medium mb-1">Similarity Threshold</label>
                      <div className="flex items-center space-x-3">
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={recipe.settings.culling.similarityThreshold}
                          onChange={(e) => setRecipe({
                            ...recipe,
                            settings: {
                              ...recipe.settings,
                              culling: {
                                ...recipe.settings.culling,
                                similarityThreshold: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="flex-1"
                        />
                        <span className="text-sm font-medium w-12 text-right">
                          {recipe.settings.culling.similarityThreshold}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </Card>

        {/* Export Settings */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Export Settings</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Format</label>
              <select
                value={recipe.settings.export.format}
                onChange={(e) => setRecipe({
                  ...recipe,
                  settings: {
                    ...recipe.settings,
                    export: {
                      ...recipe.settings.export,
                      format: e.target.value as 'jpeg' | 'png' | 'webp'
                    }
                  }
                })}
                className="w-full px-3 py-2 border rounded-md"
              >
                <option value="jpeg">JPEG</option>
                <option value="png">PNG</option>
                <option value="webp">WebP</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Quality</label>
              <div className="flex items-center space-x-3">
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={recipe.settings.export.quality}
                  onChange={(e) => setRecipe({
                    ...recipe,
                    settings: {
                      ...recipe.settings,
                      export: {
                        ...recipe.settings.export,
                        quality: parseInt(e.target.value)
                      }
                    }
                  })}
                  className="flex-1"
                />
                <span className="text-sm font-medium w-12 text-right">
                  {recipe.settings.export.quality}%
                </span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Maximum Dimension (optional)
              </label>
              <Input
                type="number"
                value={recipe.settings.export.maxDimension || ''}
                onChange={(e) => setRecipe({
                  ...recipe,
                  settings: {
                    ...recipe.settings,
                    export: {
                      ...recipe.settings.export,
                      maxDimension: e.target.value ? parseInt(e.target.value) : undefined
                    }
                  }
                })}
                placeholder="e.g., 2048"
                min="1"
              />
              <p className="text-sm text-gray-600 mt-1">
                Leave empty to preserve original dimensions
              </p>
            </div>

            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={recipe.settings.export.preserveMetadata}
                onChange={(e) => setRecipe({
                  ...recipe,
                  settings: {
                    ...recipe.settings,
                    export: {
                      ...recipe.settings.export,
                      preserveMetadata: e.target.checked
                    }
                  }
                })}
              />
              <span className="text-sm">Preserve metadata</span>
            </label>
          </div>
        </Card>

        {/* AI Model Settings */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Advanced AI Model Parameters</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Enhancement Model</label>
              <select
                value={recipe.settings.aiModels.enhancementModel}
                onChange={(e) => setRecipe({
                  ...recipe,
                  settings: {
                    ...recipe.settings,
                    aiModels: {
                      ...recipe.settings.aiModels,
                      enhancementModel: e.target.value
                    }
                  }
                })}
                className="w-full px-3 py-2 border rounded-md"
              >
                <option value="standard">Standard</option>
                <option value="high-quality">High Quality</option>
                <option value="fast">Fast</option>
                <option value="none">None</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Enhancement Strength</label>
              <div className="flex items-center space-x-3">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={recipe.settings.aiModels.enhancementStrength}
                  onChange={(e) => setRecipe({
                    ...recipe,
                    settings: {
                      ...recipe.settings,
                      aiModels: {
                        ...recipe.settings.aiModels,
                        enhancementStrength: parseInt(e.target.value)
                      }
                    }
                  })}
                  className="flex-1"
                />
                <span className="text-sm font-medium w-12 text-right">
                  {recipe.settings.aiModels.enhancementStrength}%
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={recipe.settings.aiModels.objectDetection}
                  onChange={(e) => setRecipe({
                    ...recipe,
                    settings: {
                      ...recipe.settings,
                      aiModels: {
                        ...recipe.settings.aiModels,
                        objectDetection: e.target.checked
                      }
                    }
                  })}
                />
                <span className="text-sm">Enable object detection</span>
              </label>

              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={recipe.settings.aiModels.sceneAnalysis}
                  onChange={(e) => setRecipe({
                    ...recipe,
                    settings: {
                      ...recipe.settings,
                      aiModels: {
                        ...recipe.settings.aiModels,
                        sceneAnalysis: e.target.checked
                      }
                    }
                  })}
                />
                <span className="text-sm">Enable scene analysis</span>
              </label>

              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={recipe.settings.aiModels.faceEnhancement}
                  onChange={(e) => setRecipe({
                    ...recipe,
                    settings: {
                      ...recipe.settings,
                      aiModels: {
                        ...recipe.settings.aiModels,
                        faceEnhancement: e.target.checked
                      }
                    }
                  })}
                />
                <span className="text-sm">Enable face enhancement</span>
              </label>
            </div>
          </div>
        </Card>

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-6 border-t">
          <Button
            variant="outline"
            onClick={() => navigate('/recipes')}
          >
            Cancel
          </Button>
          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              onClick={() => handleSave(true)}
              disabled={isSaving}
            >
              <Save className="h-4 w-4 mr-1" />
              Save as Draft
            </Button>
            <Button
              onClick={() => handleSave(false)}
              disabled={isSaving}
            >
              {isSaving ? (
                <>Saving...</>
              ) : (
                <>
                  <Check className="h-4 w-4 mr-1" />
                  Save Recipe
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}