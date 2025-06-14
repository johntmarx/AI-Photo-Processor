import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/Dialog'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent } from '@/components/ui/Card'
import { Recipe, ProcessingStep, RecipePreset } from '@/types/api'
import { Plus, X, Settings, Save } from 'lucide-react'

interface RecipeDialogProps {
  isOpen: boolean
  onClose: () => void
  onSave: (recipe: Partial<Recipe>) => void
  recipe?: Recipe | null
  presets?: RecipePreset[]
}

const availableOperations = [
  { id: 'enhance', name: 'AI Enhancement', description: 'Enhance image quality using AI' },
  { id: 'denoise', name: 'Noise Reduction', description: 'Reduce image noise' },
  { id: 'sharpen', name: 'Sharpen', description: 'Increase image sharpness' },
  { id: 'color_correct', name: 'Color Correction', description: 'Automatic color correction' },
  { id: 'exposure', name: 'Exposure Adjustment', description: 'Adjust image exposure' },
  { id: 'contrast', name: 'Contrast Enhancement', description: 'Enhance image contrast' },
  { id: 'crop_subject', name: 'Smart Crop', description: 'AI-powered subject cropping' },
  { id: 'rotate', name: 'Auto Rotation', description: 'Correct image orientation' },
  { id: 'raw_develop', name: 'RAW Development', description: 'Process RAW files' },
]

export default function RecipeDialog({ isOpen, onClose, onSave, recipe, presets }: RecipeDialogProps) {
  const [name, setName] = useState(recipe?.name || '')
  const [description, setDescription] = useState(recipe?.description || '')
  const [steps, setSteps] = useState<ProcessingStep[]>(recipe?.steps || [])
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)

  const handleAddStep = (operation: string) => {
    const newStep: ProcessingStep = {
      operation,
      parameters: {},
      enabled: true
    }
    setSteps([...steps, newStep])
  }

  const handleRemoveStep = (index: number) => {
    setSteps(steps.filter((_, i) => i !== index))
  }

  const handleToggleStep = (index: number) => {
    const newSteps = [...steps]
    newSteps[index].enabled = !newSteps[index].enabled
    setSteps(newSteps)
  }

  const handleMoveStep = (index: number, direction: 'up' | 'down') => {
    const newSteps = [...steps]
    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex >= 0 && newIndex < steps.length) {
      [newSteps[index], newSteps[newIndex]] = [newSteps[newIndex], newSteps[index]]
      setSteps(newSteps)
    }
  }

  const handleUsePreset = (preset: RecipePreset) => {
    setName(preset.name)
    setDescription(preset.description)
    setSteps(preset.steps || preset.operations || [])
    setSelectedPreset(preset.id)
  }

  const handleSave = () => {
    if (!name.trim()) {
      alert('Please enter a recipe name')
      return
    }

    const recipeData: Partial<Recipe> = {
      name: name.trim(),
      description: description.trim(),
      steps,
      // Additional fields will be handled by the API
    }

    if (recipe?.id) {
      recipeData.id = recipe.id
    }

    onSave(recipeData)
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden">
        <DialogHeader className="border-b pb-4">
          <DialogTitle>{recipe ? 'Edit Recipe' : 'Create New Recipe'}</DialogTitle>
        </DialogHeader>

        <div className="flex gap-4 h-[60vh]">
          {/* Left panel - Recipe details */}
          <div className="flex-1 space-y-4 overflow-y-auto pr-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Recipe Name</label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Portrait Enhancement"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe what this recipe does..."
                className="w-full min-h-[80px] p-2 border rounded-md text-sm"
              />
            </div>

            {/* Presets */}
            {presets && presets.length > 0 && !recipe && (
              <div>
                <label className="text-sm font-medium mb-2 block">Start from Preset</label>
                <div className="grid gap-2">
                  {presets.map((preset) => (
                    <Card 
                      key={preset.id} 
                      className={`cursor-pointer transition-colors ${
                        selectedPreset === preset.id ? 'border-primary' : ''
                      }`}
                      onClick={() => handleUsePreset(preset)}
                    >
                      <CardContent className="p-3">
                        <div className="font-medium text-sm">{preset.name}</div>
                        <div className="text-xs text-muted-foreground">{preset.description}</div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}

            {/* Processing Steps */}
            <div>
              <label className="text-sm font-medium mb-2 block">Processing Steps</label>
              <div className="space-y-2">
                {steps.map((step, index) => (
                  <div 
                    key={index} 
                    className="flex items-center gap-2 p-3 border rounded-lg"
                  >
                    <span className="text-sm font-medium w-6">{index + 1}.</span>
                    <div className="flex-1">
                      <div className="font-medium text-sm">{step.operation}</div>
                      <div className="text-xs text-muted-foreground">
                        {availableOperations.find(op => op.id === step.operation)?.description}
                      </div>
                    </div>
                    <Badge variant={step.enabled ? "success" : "secondary"}>
                      {step.enabled ? "Enabled" : "Disabled"}
                    </Badge>
                    <div className="flex items-center gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleMoveStep(index, 'up')}
                        disabled={index === 0}
                      >
                        ↑
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleMoveStep(index, 'down')}
                        disabled={index === steps.length - 1}
                      >
                        ↓
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleToggleStep(index)}
                      >
                        <Settings className="h-3 w-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => handleRemoveStep(index)}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                ))}
                
                {steps.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    No processing steps added yet
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right panel - Available operations */}
          <div className="w-64 border-l pl-4">
            <h3 className="font-medium mb-3">Available Operations</h3>
            <div className="space-y-2">
              {availableOperations.map((op) => (
                <Card 
                  key={op.id}
                  className="cursor-pointer hover:border-primary transition-colors"
                  onClick={() => handleAddStep(op.id)}
                >
                  <CardContent className="p-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-sm">{op.name}</div>
                        <div className="text-xs text-muted-foreground">{op.description}</div>
                      </div>
                      <Plus className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 border-t pt-4">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={!name.trim() || steps.length === 0}>
            <Save className="h-4 w-4 mr-2" />
            {recipe ? 'Update Recipe' : 'Create Recipe'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}