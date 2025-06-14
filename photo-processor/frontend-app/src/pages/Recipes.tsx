import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { recipesApi } from '@/services/api'
import { Recipe, RecipePreset } from '@/types/api'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { formatRelativeTime } from '@/lib/utils'
import { 
  Plus, 
  Search, 
  BookOpen, 
  Edit, 
  Copy, 
  Trash2,
  Play,
  Star,
  Sparkles,
  Users
} from 'lucide-react'

export default function Recipes() {
  const navigate = useNavigate()
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const queryClient = useQueryClient()

  const { data: recipesData, isLoading } = useQuery(
    ['recipes'],
    () => recipesApi.list().then(res => res.data)
  )

  const { data: presetsData } = useQuery(
    ['recipes', 'presets'],
    () => recipesApi.getPresets().then(res => res.data)
  )

  const recipes: Recipe[] = Array.isArray(recipesData) ? recipesData : (recipesData?.recipes || [])
  const presets: RecipePreset[] = Array.isArray(presetsData) ? presetsData : (presetsData?.presets || [])


  const deleteMutation = useMutation(
    (recipeId: string) => recipesApi.delete(recipeId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['recipes'])
        toast.success('Recipe deleted successfully')
      },
      onError: () => toast.error('Failed to delete recipe')
    }
  )

  const duplicateMutation = useMutation(
    ({ recipeId, name }: { recipeId: string; name: string }) => 
      recipesApi.duplicate(recipeId, name),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['recipes'])
        toast.success('Recipe duplicated successfully')
      },
      onError: () => toast.error('Failed to duplicate recipe')
    }
  )

  const filteredRecipes = recipes.filter((recipe: Recipe) =>
    recipe.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    recipe.description.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleDuplicate = (recipe: Recipe) => {
    const name = prompt('Enter name for duplicated recipe:', `${recipe.name} (Copy)`)
    if (name) {
      duplicateMutation.mutate({ recipeId: recipe.id, name })
    }
  }

  const handleDelete = (recipe: Recipe) => {
    if (confirm(`Delete recipe "${recipe.name}"?`)) {
      deleteMutation.mutate(recipe.id)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Recipes</h1>
          <p className="text-muted-foreground">
            Manage processing recipes and presets
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => navigate('/recipes/builder')}>
            <Sparkles className="h-4 w-4 mr-2" />
            Recipe Builder
          </Button>
          <Button onClick={() => navigate('/recipes/new')}>
            <Plus className="h-4 w-4 mr-2" />
            New Recipe
          </Button>
        </div>
      </div>

      {/* Search */}
      <div className="flex items-center space-x-4">
        <div className="flex-1 max-w-sm">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Search recipes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
      </div>

      {/* Recipe Presets */}
      {presets && presets.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Recipe Presets</h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {presets.map((preset) => (
              <Card key={preset.id} className="cursor-pointer hover:shadow-md transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base flex items-center">
                      <Star className="h-4 w-4 mr-2 text-yellow-500" />
                      {preset.name}
                    </CardTitle>
                    <Badge variant="outline" className="text-xs">
                      Preset
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-3">
                    {preset.description}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      {(preset.steps || preset.operations || []).length} steps
                    </span>
                    <Button size="sm" variant="outline">
                      Use Preset
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* User Recipes */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Your Recipes</h2>
          <span className="text-sm text-muted-foreground">
            {filteredRecipes.length} recipe{filteredRecipes.length !== 1 ? 's' : ''}
          </span>
        </div>

        {isLoading ? (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-48 bg-muted animate-pulse rounded-lg" />
            ))}
          </div>
        ) : filteredRecipes.length === 0 ? (
          <div className="text-center py-12">
            <BookOpen className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No recipes found</h3>
            <p className="text-sm text-muted-foreground mb-4">
              {searchQuery ? 'Try adjusting your search' : 'Create your first recipe to get started'}
            </p>
            <Button onClick={() => navigate('/recipes/new')}>
              <Plus className="h-4 w-4 mr-2" />
              Create Recipe
            </Button>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredRecipes.map((recipe) => (
              <Card 
                key={recipe.id} 
                className="cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => setSelectedRecipe(recipe)}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{recipe.name}</CardTitle>
                    <div className="flex items-center space-x-1">
                      {recipe.is_preset && (
                        <Badge variant="outline" className="text-xs">
                          <Star className="h-3 w-3 mr-1" />
                          Preset
                        </Badge>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                    {recipe.description}
                  </p>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{recipe.steps.length} steps</span>
                      <span>{formatRelativeTime(new Date(recipe.updated_at))}</span>
                    </div>
                    
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <div className="flex items-center">
                        <Users className="h-3 w-3 mr-1" />
                        <span>Used {recipe.usage_count} times</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between mt-4">
                    <div className="flex items-center space-x-1">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation()
                          navigate(`/recipes/${recipe.id}/edit`)
                        }}
                      >
                        <Edit className="h-3 w-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDuplicate(recipe)
                        }}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDelete(recipe)
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                    <Button size="sm">
                      <Play className="h-3 w-3 mr-1" />
                      Apply
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Recipe Detail Modal would go here */}
      {selectedRecipe && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="bg-background rounded-lg shadow-lg max-w-2xl w-full max-h-[80vh] overflow-hidden">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">{selectedRecipe.name}</h2>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setSelectedRecipe(null)}
                >
                  ×
                </Button>
              </div>
            </div>
            <div className="p-6">
              <p className="text-muted-foreground mb-4">
                {selectedRecipe.description}
              </p>
              <div className="space-y-3">
                <h3 className="font-medium">Processing Steps:</h3>
                {selectedRecipe.steps.map((step, index) => (
                  <div key={index} className="flex items-center space-x-3 p-2 border rounded">
                    <span className="text-sm font-medium">{index + 1}.</span>
                    <span className="text-sm">{step.operation}</span>
                    <Badge variant={step.enabled ? "success" : "secondary"}>
                      {step.enabled ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}