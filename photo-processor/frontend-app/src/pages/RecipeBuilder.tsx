import { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'
import { Button } from '@/components/ui/Button'
import { Progress } from '@/components/ui/Progress'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Input } from '@/components/ui/Input'
import { 
  ChevronRight, 
  RotateCw, 
  Crop, 
  Sparkles,
  Save,
  X,
  Zap,
  Image as ImageIcon,
  Search,
  Filter,
  SlidersHorizontal,
  CheckCircle,
  XCircle,
  Clock,
  Star,
  Camera,
  Loader2
} from 'lucide-react'
import { photosApi } from '@/services/api'
import PhotoGrid from '@/components/photos/PhotoGrid'
import { useWebSocketStatus } from '@/providers/WebSocketProvider'

// Status filters for photo selection
const statusFilters = [
  { label: 'All', value: '', icon: null },
  { label: 'Completed', value: 'completed', icon: CheckCircle },
  { label: 'Processing', value: 'processing', icon: Zap },
  { label: 'Pending', value: 'pending', icon: Clock },
  { label: 'Failed', value: 'failed', icon: XCircle },
]

// Sort options for photo selection
const sortOptions = [
  { label: 'Newest First', value: 'created_at_desc' },
  { label: 'Oldest First', value: 'created_at_asc' },
  { label: 'Aesthetic Score (High)', value: 'aesthetic_desc' },
  { label: 'Aesthetic Score (Low)', value: 'aesthetic_asc' },
  { label: 'Technical Score (High)', value: 'technical_desc' },
  { label: 'Technical Score (Low)', value: 'technical_asc' },
  { label: 'Combined Score (High)', value: 'combined_desc' },
  { label: 'Combined Score (Low)', value: 'combined_asc' },
]

// Recipe builder API (to be added to api.ts)
const recipeBuilderApi = {
  start: (photoIds: string[], name: string, description: string) =>
    fetch('/api/recipe-builder/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ photo_ids: photoIds, name, description })
    }).then(res => res.json()),
    
  getCurrentState: (sessionId: string) =>
    fetch(`/api/recipe-builder/${sessionId}/current`).then(res => res.json()),
    
  analyzeRotation: (sessionId: string) =>
    fetch(`/api/recipe-builder/${sessionId}/rotate/analyze`, {
      method: 'POST'
    }).then(res => res.json()),
    
  applyRotation: (sessionId: string, angle: number, autoDetect: boolean) =>
    fetch(`/api/recipe-builder/${sessionId}/rotate/apply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ angle, auto_detect: autoDetect })
    }).then(res => res.json()),
    
  updateCropSettings: (sessionId: string, aspectRatio: string) =>
    fetch(`/api/recipe-builder/${sessionId}/crop/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ aspect_ratio: aspectRatio })
    }).then(res => res.json()),
    
  analyzeCropComposition: (sessionId: string, ollamaModel?: string) =>
    fetch(`/api/recipe-builder/${sessionId}/crop/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ollama_model: ollamaModel || 'qwen2.5-vl:7b' })
    }).then(res => res.json()),
    
  generateIntelligentCrop: (sessionId: string, userIntent: string, targetAspectRatio: string, ollamaModel?: string) =>
    fetch(`/api/recipe-builder/${sessionId}/crop/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        user_intent: userIntent, 
        target_aspect_ratio: targetAspectRatio,
        ollama_model: ollamaModel || 'qwen2.5-vl:7b'
      })
    }).then(res => res.json()),

  applyCrop: (sessionId: string, aspectRatio?: string, cropBox?: any, useIntelligentCrop?: boolean) =>
    fetch(`/api/recipe-builder/${sessionId}/crop/apply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        aspect_ratio: aspectRatio, 
        crop_box: cropBox,
        use_intelligent_crop: useIntelligentCrop || false
      })
    }).then(res => {
      if (!res.ok) {
        return res.json().then(err => Promise.reject(err))
      }
      return res.json()
    }),
    
  previewEnhancement: (sessionId: string, strength: number) =>
    fetch(`/api/recipe-builder/${sessionId}/enhance/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strength: strength / 100 }) // Convert percentage to 0-2 range
    }).then(res => res.json()),
    
  previewCustomEnhancement: (sessionId: string, settings: any) =>
    fetch(`/api/recipe-builder/${sessionId}/enhance/preview-custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    }).then(res => res.json()),
    
  applyEnhancement: (sessionId: string, strength: number) =>
    fetch(`/api/recipe-builder/${sessionId}/enhance/apply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strength: strength / 100 }) // Convert percentage to 0-2 range
    }).then(res => res.json()),
    
  applyCustomEnhancement: (sessionId: string, settings: any) =>
    fetch(`/api/recipe-builder/${sessionId}/enhance/apply-custom`, {
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    }).then(res => res.json()),
    
  saveRecipe: (sessionId: string) =>
    fetch(`/api/recipe-builder/${sessionId}/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ finalize: true })
    }).then(res => res.json()),
    
  getOllamaModels: () =>
    fetch('/api/recipe-builder/ollama/models')
      .then(res => res.json())
}

export default function RecipeBuilder() {
  const { isConnected } = useWebSocketStatus()
  
  const [selectedPhotos, setSelectedPhotos] = useState<string[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [currentStep, setCurrentStep] = useState<'select' | 'rotate' | 'crop' | 'enhance' | 'complete'>('select')
  const [recipeName, setRecipeName] = useState('My Recipe')
  const [recipeDescription, setRecipeDescription] = useState('')
  const [currentPhotoIndex, setCurrentPhotoIndex] = useState(0)
  const [rotationAngle, setRotationAngle] = useState(0)
  const [autoRotate, setAutoRotate] = useState(true)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [analysisPhase, setAnalysisPhase] = useState('idle')
  const [analysisMessage, setAnalysisMessage] = useState('')
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [enhancementStrength, setEnhancementStrength] = useState(100) // 0-200%
  const [enhancementPreview, setEnhancementPreview] = useState<{
    originalUrl?: string
    previewUrl?: string
  }>({})
  const [isGeneratingPreview, setIsGeneratingPreview] = useState(false)
  
  // Individual enhancement controls
  const [enhancementMode, setEnhancementMode] = useState<'intelligent' | 'custom'>('intelligent')
  const [customEnhancements, setCustomEnhancements] = useState({
    white_balance: true,
    white_balance_strength: 1.0,
    exposure: true,
    exposure_strength: 1.0,
    contrast: true,
    contrast_strength: 1.0,
    vibrance: true,
    vibrance_strength: 1.0,
    shadow_highlight: true,
    shadow_highlight_strength: 1.0,
  })

  // VLM Cropping states
  const [cropMode, setCropMode] = useState<'manual' | 'intelligent'>('intelligent')
  const [ollamaModel, setOllamaModel] = useState('')
  const [availableModels, setAvailableModels] = useState<any[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [compositionAnalysis, setCompositionAnalysis] = useState<any>(null)
  const [isAnalyzingComposition, setIsAnalyzingComposition] = useState(false)
  const [showVlmResults, setShowVlmResults] = useState(false)

  // Photo selection pagination and filtering
  const [page, setPage] = useState(1)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState('created_at_desc')
  const [statusFilter, setStatusFilter] = useState('')
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false)
  const [minAestheticScore, setMinAestheticScore] = useState<number | ''>('')
  const [maxAestheticScore, setMaxAestheticScore] = useState<number | ''>('')
  const [minTechnicalScore, setMinTechnicalScore] = useState<number | ''>('')
  const [maxTechnicalScore, setMaxTechnicalScore] = useState<number | ''>('')

  // Get photos for selection with pagination and filtering
  const { data: photosData, isLoading: photosLoading } = useQuery(
    ['photos-for-recipe', page, searchQuery, sortBy, statusFilter, minAestheticScore, maxAestheticScore, minTechnicalScore, maxTechnicalScore],
    () => photosApi.list({
      page,
      per_page: 20,
      status: statusFilter || undefined,
      search: searchQuery || undefined,
      sort: sortBy,
      min_aesthetic_score: minAestheticScore || undefined,
      max_aesthetic_score: maxAestheticScore || undefined,
      min_technical_score: minTechnicalScore || undefined,
      max_technical_score: maxTechnicalScore || undefined,
    }).then(res => res.data),
    { enabled: currentStep === 'select', keepPreviousData: true }
  )

  // Get current session state
  const { data: sessionState, refetch: refetchSession } = useQuery(
    ['recipe-session', sessionId],
    () => sessionId ? recipeBuilderApi.getCurrentState(sessionId) : null,
    { 
      enabled: !!sessionId,
      // Don't poll - we'll refetch on WebSocket events
      refetchInterval: false,
      staleTime: 30000, // Consider data stale after 30 seconds
    }
  )

  // Fetch available Ollama models when on crop step
  useEffect(() => {
    if (currentStep === 'crop' && cropMode === 'intelligent') {
      // Always fetch fresh models when entering crop step
      setLoadingModels(true)
      setAvailableModels([]) // Clear any old data
      setOllamaModel('') // Clear selection
      
      recipeBuilderApi.getOllamaModels()
        .then(data => {
          console.log('Ollama models response:', data)
          if (data.models && data.models.length > 0) {
            setAvailableModels(data.models)
          } else {
            toast.error(data.error || 'No vision models found in Ollama')
          }
        })
        .catch(error => {
          console.error('Failed to fetch Ollama models:', error)
          toast.error('Failed to fetch available models')
        })
        .finally(() => {
          setLoadingModels(false)
        })
    }
  }, [currentStep, cropMode])

  // Listen for WebSocket events that should trigger session state refetch
  useEffect(() => {
    if (!sessionId) return

    const handleRecipeBuilderEvent = (event: CustomEvent) => {
      const data = event.detail.data
      // Only refetch if the event is for our session
      if (data && data.session_id === sessionId) {
        refetchSession()
      }
    }

    const handleCompositionAnalysisComplete = (event: CustomEvent) => {
      const data = event.detail.data
      if (data && data.session_id === sessionId) {
        // Update composition analysis state with the new data
        setCompositionAnalysis(data.analysis)
        refetchSession()
      }
    }

    // Add event listeners for recipe builder events
    const events = [
      'recipe_builder_started',
      'rotation_analysis_complete',
      'rotation_analysis_failed',
      'crop_generation_complete',
      'recipe_builder_updated'
    ]

    events.forEach(eventType => {
      window.addEventListener(eventType, handleRecipeBuilderEvent as EventListener)
    })

    // Add specific handler for composition analysis
    window.addEventListener('composition_analysis_complete', handleCompositionAnalysisComplete as EventListener)

    // Cleanup
    return () => {
      events.forEach(eventType => {
        window.removeEventListener(eventType, handleRecipeBuilderEvent as EventListener)
      })
      window.removeEventListener('composition_analysis_complete', handleCompositionAnalysisComplete as EventListener)
    }
  }, [sessionId, refetchSession])

  // Refetch session state when step changes
  useEffect(() => {
    if (sessionId && currentStep === 'enhance') {
      // When entering enhance step, ensure we have the latest session state
      refetchSession()
    }
  }, [currentStep, sessionId, refetchSession])

  // Start recipe building
  const startSession = async () => {
    if (selectedPhotos.length === 0) {
      toast.error('Please select at least one photo')
      return
    }
    
    if (selectedPhotos.length > 10) {
      toast.error('Maximum 10 photos allowed for recipe building')
      return
    }

    const result = await recipeBuilderApi.start(
      selectedPhotos,
      recipeName,
      recipeDescription
    )
    
    setSessionId(result.session_id)
    setCurrentStep('rotate')
    toast.success('Recipe building session started')
  }

  // Rotation analysis parameters
  const [rotationMethod, setRotationMethod] = useState<'cv' | 'onealign'>('cv')
  const [cvMethod, setCvMethod] = useState('auto')
  const [minRotationAngle, setMinRotationAngle] = useState(-20)
  const [maxRotationAngle, setMaxRotationAngle] = useState(20)
  const [rotationStepSize, setRotationStepSize] = useState(0.5)
  const [analysisResults, setAnalysisResults] = useState<Record<string, any>>({})
  const [currentAnalyzingPhoto, setCurrentAnalyzingPhoto] = useState(0)
  const [showAnalysisResults, setShowAnalysisResults] = useState(false)

  // Analyze rotation for all selected photos
  const analyzeRotation = async () => {
    console.log('analyzeRotation called')
    
    if (!sessionId || !sessionState) {
      console.error('No session ID or session state available', { sessionId, sessionState })
      toast.error('No active session. Please start over.')
      return
    }
    
    console.log('Starting rotation analysis:', {
      sessionId,
      method: rotationMethod,
      cvMethod,
      sessionState
    })
    
    setAnalysisProgress(0)
    setCurrentAnalyzingPhoto(0)
    setAnalysisResults({})
    setShowAnalysisResults(true)
    
    // Set up event listeners for WebSocket events
    const handleProgressEvent = (event: CustomEvent) => {
      const data = event.detail.data
      if (data.session_id === sessionId) {
        setAnalysisProgress(data.progress)
        setAnalysisPhase(data.phase || 'processing')
        setAnalysisMessage(data.message || 'Processing...')
        
        // Update results for current photo
        if (data.photo_id) {
          setAnalysisResults(prev => ({
            ...prev,
            [data.photo_id]: {
              ...prev[data.photo_id],
              currentAngle: data.current_angle,
              currentScore: data.current_score,
              bestAngle: data.best_angle,
              bestScore: data.best_score,
              progress: data.progress,
              phase: data.phase,
              message: data.message
            }
          }))
        }
      }
    }

    const handleCompleteEvent = (event: CustomEvent) => {
      const data = event.detail.data
      if (data.session_id === sessionId) {
        setAnalysisProgress(100)
        setAnalysisPhase('complete')
        setAnalysisMessage('Analysis completed successfully')
        
        if (data.photo_id) {
          setAnalysisResults(prev => ({
            ...prev,
            [data.photo_id]: {
              ...prev[data.photo_id],
              optimal_angle: data.optimal_angle,
              optimal_score: data.optimal_score,
              display_image_url: data.display_image_url,
              progress: 100,
              phase: 'complete',
              message: 'Analysis completed successfully'
            }
          }))
        }
      }
    }

    const handleFailedEvent = (event: CustomEvent) => {
      const data = event.detail.data
      if (data.session_id === sessionId) {
        setAnalysisProgress(0)
        setAnalysisPhase('failed')
        setAnalysisMessage(data.message || 'Analysis failed')
        toast.error(`Analysis failed: ${data.error || 'Unknown error'}`)
      }
    }

    // Add event listeners
    window.addEventListener('rotation_analysis_progress', handleProgressEvent)
    window.addEventListener('rotation_analysis_complete', handleCompleteEvent)
    window.addEventListener('rotation_analysis_failed', handleFailedEvent)
    
    try {
      // Note: The API now handles one photo at a time based on the session's current photo
      // Call the API with custom parameters - use correct API port
      const requestBody = {
        method: rotationMethod,
        min_angle: minRotationAngle,
        max_angle: maxRotationAngle,
        angle_step: rotationStepSize,
        cv_method: cvMethod,
        recipe_params: rotationMethod === 'cv' ? {
          enable_face_detection: true,
          enable_perspective_correction: true,
          enable_skew_correction: true,
          enable_distortion_correction: false
        } : undefined
      }
      
      console.log('Sending rotation analysis request:', requestBody)
      
      const response = await fetch(`/api/recipe-builder/${sessionId}/rotate/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })
      
      console.log('Response status:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('API error response:', errorText)
        throw new Error(`API returned ${response.status}: ${response.statusText} - ${errorText}`)
      }
      
      const result = await response.json()
      console.log('Analysis result:', result)
      
      // Get the current photo from session state
      const currentPhotoId = sessionState?.current_photo_id
      if (currentPhotoId) {
        // Handle different response formats for CV vs OneAlign
        if (rotationMethod === 'cv') {
          setAnalysisResults(prev => ({
            ...prev,
            [currentPhotoId]: {
              optimal_angle: result.optimal_angle,
              optimal_score: result.confidence || 1.0,
              needs_rotation: result.needs_rotation,
              method_used: result.method_used,
              scene_type: result.scene_type,
              needs_perspective_correction: result.needs_perspective_correction,
              needs_skew_correction: result.needs_skew_correction,
              display_image_url: result.display_image_url,
              status: result.status,
              photoId: currentPhotoId,
              filename: `photo_${sessionState?.current_photo_index + 1}`
            }
          }))
        } else {
          setAnalysisResults(prev => ({
            ...prev,
            [currentPhotoId]: {
              ...result,
              photoId: currentPhotoId,
              filename: `photo_${sessionState?.current_photo_index + 1}`,
              display_image_url: result.display_image_url
            }
          }))
        }
      }
      
      toast.success(`Analysis complete using ${rotationMethod.toUpperCase()} method`)
    } catch (error) {
      console.error('Analysis failed:', error)
      toast.error(`Failed to analyze rotation: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      // Clean up event listeners
      window.removeEventListener('rotation_analysis_progress', handleProgressEvent)
      window.removeEventListener('rotation_analysis_complete', handleCompleteEvent)
      window.removeEventListener('rotation_analysis_failed', handleFailedEvent)
    }
  }

  // Apply rotation
  const applyRotation = async () => {
    if (!sessionId) return
    
    const result = await recipeBuilderApi.applyRotation(
      sessionId,
      rotationAngle,
      autoRotate
    )
    
    if (result.next_step) {
      setCurrentStep(result.next_step)
      // Refetch session state to get the updated temp_image_url
      await refetchSession()
      toast.success('Rotation applied')
    }
  }

  // Analyze composition with VLM
  const analyzeComposition = async () => {
    if (!sessionId) return
    
    setIsAnalyzingComposition(true)
    setCompositionAnalysis(null)
    
    try {
      const result = await recipeBuilderApi.analyzeCropComposition(sessionId, ollamaModel)
      setCompositionAnalysis(result)
      setShowVlmResults(true)
      toast.success('Composition analysis completed')
    } catch (error) {
      console.error('Composition analysis failed:', error)
      toast.error('Failed to analyze composition')
    } finally {
      setIsAnalyzingComposition(false)
    }
  }

  // No longer needed - crop generation happens automatically after analysis

  // Apply crop
  const applyCrop = async () => {
    if (!sessionId) return
    
    try {
      const useIntelligentCrop = cropMode === 'intelligent' && compositionAnalysis?.round2
      
      const cropBox = compositionAnalysis?.round2 ? {
        x: compositionAnalysis.round2.crop_coordinates.x1_px,
        y: compositionAnalysis.round2.crop_coordinates.y1_px,
        width: compositionAnalysis.round2.crop_coordinates.x2_px - compositionAnalysis.round2.crop_coordinates.x1_px,
        height: compositionAnalysis.round2.crop_coordinates.y2_px - compositionAnalysis.round2.crop_coordinates.y1_px
      } : null
      
      console.log('Applying crop with:', { sessionId, cropBox, useIntelligentCrop })
      
      const result = await recipeBuilderApi.applyCrop(
        sessionId, 
        null, // aspect ratio is now auto-selected by AI
        cropBox,
        useIntelligentCrop
      )
      
      console.log('Crop result:', result)
      
      if (result.next_step) {
        setCurrentStep(result.next_step)
        // Refetch session state to get the updated temp_image_url
        await refetchSession()
        toast.success('Crop applied')
      } else {
        console.error('No next_step in result:', result)
        toast.error('Failed to move to next step')
      }
    } catch (error) {
      console.error('Apply crop failed:', error)
      toast.error('Failed to apply crop')
    }
  }

  // Generate enhancement preview
  const generateEnhancementPreview = async () => {
    if (!sessionId || isGeneratingPreview) return
    
    setIsGeneratingPreview(true)
    try {
      let result
      if (enhancementMode === 'intelligent') {
        result = await recipeBuilderApi.previewEnhancement(sessionId, enhancementStrength)
      } else {
        // Use custom enhancement endpoint
        result = await recipeBuilderApi.previewCustomEnhancement(sessionId, {
          ...customEnhancements,
          overall_strength: enhancementStrength / 100
        })
      }
      
      if (result.success) {
        setEnhancementPreview({
          originalUrl: result.original_url,
          previewUrl: result.preview_url
        })
        toast.success('Preview generated')
      }
    } catch (error) {
      console.error('Failed to generate preview:', error)
      toast.error('Failed to generate preview')
    } finally {
      setIsGeneratingPreview(false)
    }
  }

  // Apply enhancement
  const applyEnhancement = async () => {
    if (!sessionId) return
    
    let result
    if (enhancementMode === 'intelligent') {
      result = await recipeBuilderApi.applyEnhancement(sessionId, enhancementStrength)
    } else {
      // Apply custom enhancement (using same endpoint but with custom parameters)
      result = await recipeBuilderApi.applyCustomEnhancement(sessionId, {
        ...customEnhancements,
        overall_strength: enhancementStrength / 100
      })
    }
    
    if (result.next_action === 'next_photo') {
      setCurrentPhotoIndex(result.next_photo_index)
      setCurrentStep('rotate')
      toast.success('Moving to next photo')
    } else if (result.next_action === 'complete') {
      setCurrentStep('complete')
      toast.success('All photos processed!')
    }
  }

  // Save recipe
  const saveRecipe = async () => {
    if (!sessionId) return
    
    const result = await recipeBuilderApi.saveRecipe(sessionId)
    toast.success(`Recipe saved: ${result.recipe.name}`)
    
    // Reset state
    setSessionId(null)
    setCurrentStep('select')
    setSelectedPhotos([])
  }

  // Render based on current step
  if (currentStep === 'select') {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Recipe Builder</h1>
          <p className="text-muted-foreground">
            Select up to 10 photos to create a custom processing recipe
          </p>
        </div>

        <div className="space-y-4">
          <Card className="p-4">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Recipe Name</label>
                <input
                  type="text"
                  value={recipeName}
                  onChange={(e) => setRecipeName(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md"
                  placeholder="My Custom Recipe"
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Description</label>
                <textarea
                  value={recipeDescription}
                  onChange={(e) => setRecipeDescription(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md"
                  rows={3}
                  placeholder="Describe what this recipe does..."
                />
              </div>
            </div>
          </Card>

          {/* Filters and Search */}
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="flex-1 max-w-sm">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                  <Input
                    placeholder="Search photos..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-muted-foreground" />
                <div className="flex space-x-1">
                  {statusFilters.map((filter) => {
                    const Icon = filter.icon
                    return (
                      <Button
                        key={filter.value}
                        variant={statusFilter === filter.value ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setStatusFilter(filter.value)}
                        className="text-xs"
                      >
                        {Icon && <Icon className="h-3 w-3 mr-1" />}
                        {filter.label}
                      </Button>
                    )
                  })}
                </div>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
                >
                  <SlidersHorizontal className="h-4 w-4 mr-1" />
                  Advanced
                </Button>
              </div>
            </div>

            {/* Advanced Filters */}
            {showAdvancedFilters && (
              <div className="p-4 border rounded-lg space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Sorting */}
                  <div>
                    <label className="text-sm font-medium mb-2 block">Sort By</label>
                    <select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value)}
                      className="w-full px-3 py-2 border rounded-md bg-background"
                    >
                      {sortOptions.map(option => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Aesthetic Score Range */}
                  <div>
                    <label className="text-sm font-medium mb-2 block flex items-center">
                      <Star className="h-3 w-3 mr-1" />
                      Aesthetic Score Range
                    </label>
                    <div className="flex items-center space-x-2">
                      <Input
                        type="number"
                        placeholder="Min"
                        value={minAestheticScore}
                        onChange={(e) => setMinAestheticScore(e.target.value ? Number(e.target.value) : '')}
                        min={0}
                        max={10}
                        step={0.1}
                        className="w-20"
                      />
                      <span className="text-muted-foreground">to</span>
                      <Input
                        type="number"
                        placeholder="Max"
                        value={maxAestheticScore}
                        onChange={(e) => setMaxAestheticScore(e.target.value ? Number(e.target.value) : '')}
                        min={0}
                        max={10}
                        step={0.1}
                        className="w-20"
                      />
                    </div>
                  </div>

                  {/* Technical Score Range */}
                  <div>
                    <label className="text-sm font-medium mb-2 block flex items-center">
                      <Camera className="h-3 w-3 mr-1" />
                      Technical Score Range
                    </label>
                    <div className="flex items-center space-x-2">
                      <Input
                        type="number"
                        placeholder="Min"
                        value={minTechnicalScore}
                        onChange={(e) => setMinTechnicalScore(e.target.value ? Number(e.target.value) : '')}
                        min={0}
                        max={10}
                        step={0.1}
                        className="w-20"
                      />
                      <span className="text-muted-foreground">to</span>
                      <Input
                        type="number"
                        placeholder="Max"
                        value={maxTechnicalScore}
                        onChange={(e) => setMaxTechnicalScore(e.target.value ? Number(e.target.value) : '')}
                        min={0}
                        max={10}
                        step={0.1}
                        className="w-20"
                      />
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setMinAestheticScore('')
                      setMaxAestheticScore('')
                      setMinTechnicalScore('')
                      setMaxTechnicalScore('')
                      setSortBy('created_at_desc')
                      setSearchQuery('')
                      setStatusFilter('')
                    }}
                  >
                    Clear Filters
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium">
                {selectedPhotos.length} of {photosData?.total || 0} photos selected
              </span>
              <Button
                onClick={startSession}
                disabled={selectedPhotos.length === 0}
              >
                Start Building <ChevronRight className="h-4 w-4 ml-2" />
              </Button>
            </div>

            {/* Photo Grid */}
            {photosLoading ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="aspect-square bg-muted animate-pulse rounded-lg" />
                ))}
              </div>
            ) : photosData ? (
              <PhotoGrid
                photos={photosData.photos || photosData.items || []}
                onPhotoSelect={(photo, selected) => {
                  if (selected) {
                    if (selectedPhotos.length < 10) {
                      setSelectedPhotos([...selectedPhotos, photo.id])
                    } else {
                      toast.error('Maximum 10 photos allowed')
                    }
                  } else {
                    setSelectedPhotos(selectedPhotos.filter(id => id !== photo.id))
                  }
                }}
                selectedPhotos={new Set(selectedPhotos)}
                showActions={false}
              />
            ) : (
              <div className="text-center py-12">
                <div className="text-muted-foreground">
                  <ImageIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-medium">No photos found</p>
                  <p className="text-sm">Try adjusting your filters or upload some photos</p>
                </div>
              </div>
            )}

            {/* Pagination */}
            {photosData && (photosData.total > 20 || photosData.hasNext || photosData.hasPrev || page > 1) && (
              <div className="flex justify-center space-x-2 mt-6">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!photosData.hasPrev && page === 1}
                  onClick={() => setPage(page - 1)}
                >
                  Previous
                </Button>
                
                <div className="flex items-center space-x-1">
                  <span className="px-4 py-2 text-sm text-muted-foreground">
                    Page {page} of {Math.ceil(photosData.total / (photosData.pageSize || 20))}
                  </span>
                </div>
                
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!photosData.hasNext && page >= Math.ceil(photosData.total / (photosData.pageSize || 20))}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (currentStep === 'rotate') {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Step 1: Rotation Analysis</h1>
          <p className="text-muted-foreground">
            Analyze optimal rotation for {selectedPhotos.length} selected photos
          </p>
        </div>

        {/* Analysis Controls */}
        <Card className="p-6">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <RotateCw className="h-5 w-5" />
              <span className="font-medium">Rotation Analysis Parameters</span>
            </div>

            {/* Method Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Analysis Method</label>
              <div className="flex space-x-2">
                <Button
                  variant={rotationMethod === 'cv' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setRotationMethod('cv')}
                >
                  <Zap className="h-4 w-4 mr-2" />
                  Computer Vision (Fast)
                </Button>
                <Button
                  variant={rotationMethod === 'onealign' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setRotationMethod('onealign')}
                >
                  <Sparkles className="h-4 w-4 mr-2" />
                  AI Aesthetic (Slow)
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                {rotationMethod === 'cv' 
                  ? 'Uses Hough transform, horizon detection, and face detection for fast rotation analysis'
                  : 'Uses AI to score multiple rotation angles for optimal aesthetic quality'}
              </p>
            </div>

            {/* CV Method Selection (only shown for CV) */}
            {rotationMethod === 'cv' && (
              <div className="space-y-2">
                <label className="text-sm font-medium">CV Detection Method</label>
                <select
                  value={cvMethod}
                  onChange={(e) => setCvMethod(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md bg-background"
                >
                  <option value="auto">Auto (Best for scene type)</option>
                  <option value="horizon">Horizon Detection (Landscapes)</option>
                  <option value="lines">Line Detection (Architecture)</option>
                  <option value="faces">Face Detection (Portraits)</option>
                  <option value="exif">EXIF Only</option>
                </select>
                <p className="text-xs text-muted-foreground">
                  Auto mode will automatically choose the best method based on image content
                </p>
              </div>
            )}

            {/* Parameters for OneAlign method */}
            {rotationMethod === 'onealign' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Min Angle (°)</label>
                  <Input
                    type="number"
                    value={minRotationAngle}
                    onChange={(e) => setMinRotationAngle(Number(e.target.value))}
                    min={-45}
                    max={0}
                    step={1}
                  />
                </div>
                
                <div>
                  <label className="text-sm font-medium mb-2 block">Max Angle (°)</label>
                  <Input
                    type="number"
                    value={maxRotationAngle}
                    onChange={(e) => setMaxRotationAngle(Number(e.target.value))}
                    min={0}
                    max={45}
                    step={1}
                  />
                </div>
                
                <div>
                  <label className="text-sm font-medium mb-2 block">Step Size (°)</label>
                  <Input
                    type="number"
                    value={rotationStepSize}
                    onChange={(e) => setRotationStepSize(Number(e.target.value))}
                    min={0.1}
                    max={5}
                    step={0.1}
                  />
                </div>
              </div>
            )}

            {/* Start Analysis Button */}
            <div className="flex justify-end">
              <Button
                onClick={analyzeRotation}
                className={rotationMethod === 'cv' ? 'min-w-[200px]' : 'w-full'}
                disabled={analysisProgress > 0 && analysisProgress < 100}
              >
                <Zap className="h-4 w-4 mr-2" />
                {analysisProgress > 0 && analysisProgress < 100 ? 'Analyzing...' : 'Start Analysis'}
              </Button>
            </div>

            {/* Overall Progress */}
            {analysisProgress > 0 && analysisProgress < 100 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">
                    Analyzing photo {currentAnalyzingPhoto + 1} of {selectedPhotos.length}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {analysisProgress.toFixed(1)}% complete
                  </span>
                </div>
                <Progress value={analysisProgress} />
              </div>
            )}
          </div>
        </Card>

        {/* Analysis Results */}
        {showAnalysisResults && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Analysis Results</h2>
            
            {/* Show current photo analysis */}
            {sessionState && (
              <Card className="p-6">
                {/* Photo Header */}
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-medium">Photo {sessionState.current_photo_index + 1} of {sessionState.total_photos}</h3>
                    <p className="text-sm text-muted-foreground">{sessionState.current_photo_id}</p>
                  </div>
                  {analysisResults[sessionState.current_photo_id] && (
                    <Badge variant="success">
                      {analysisResults[sessionState.current_photo_id].optimal_angle?.toFixed(1)}° 
                      (Score: {analysisResults[sessionState.current_photo_id].optimal_score?.toFixed(2)})
                    </Badge>
                  )}
                </div>

                {/* Three-column layout: Original | Rotated | Raw Data */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Original Image */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-center">Original</h4>
                    <div className="aspect-square bg-muted rounded-lg flex items-center justify-center">
                      <img
                        src={`/api/photos/${sessionState.current_photo_id}/thumbnail`}
                        alt="Original"
                        className="max-w-full max-h-full object-contain rounded-lg"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none'
                          e.currentTarget.nextElementSibling.style.display = 'flex'
                        }}
                      />
                      <div className="hidden w-full h-full items-center justify-center">
                        <ImageIcon className="h-8 w-8 text-muted-foreground" />
                      </div>
                    </div>
                  </div>
                  
                  {/* Rotated Image */}
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-center">
                      {analysisResults[sessionState.current_photo_id] ? 
                        `Rotated & Cropped (${analysisResults[sessionState.current_photo_id].optimal_angle?.toFixed(1)}°)` : 
                        'Ready to Analyze'
                      }
                    </h4>
                    <div className="aspect-square bg-muted rounded-lg flex items-center justify-center relative">
                      {analysisResults[sessionState.current_photo_id]?.display_image_url ? (
                        <img
                          src={`${analysisResults[sessionState.current_photo_id].display_image_url}`}
                          alt="Rotated and Cropped"
                          className="max-w-full max-h-full object-contain rounded-lg"
                          onError={(e) => {
                            // Fallback to CSS-rotated thumbnail if preview image fails
                            e.currentTarget.src = `/api/photos/${sessionState.current_photo_id}/thumbnail`
                            e.currentTarget.style.transform = `rotate(${analysisResults[sessionState.current_photo_id].optimal_angle || 0}deg)`
                            e.currentTarget.className = "w-full h-full object-cover rounded-lg"
                          }}
                        />
                      ) : analysisProgress > 0 && analysisProgress < 100 ? (
                        <div className="text-center">
                          <Loader2 className="h-6 w-6 animate-spin mx-auto mb-2" />
                          <p className="text-xs capitalize">{analysisPhase === 'generating' ? 'Generating Images...' : 'Analyzing...'}</p>
                          <p className="text-xs text-muted-foreground">
                            {analysisProgress.toFixed(1)}% complete
                          </p>
                          {analysisMessage && (
                            <p className="text-xs text-muted-foreground mt-1">
                              {analysisMessage}
                            </p>
                          )}
                        </div>
                      ) : (
                        <div className="text-center">
                          <ImageIcon className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                          <p className="text-xs">Click "Start Analysis" to begin</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Raw Data & Analysis */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium text-center">Analysis Data</h4>
                    
                    {/* Score Chart - Only for OneAlign */}
                    {rotationMethod === 'onealign' && analysisResults[sessionState.current_photo_id]?.all_scores && (
                      <div className="space-y-2">
                        <p className="text-xs font-medium">Score Curve</p>
                        <div className="h-32 bg-muted rounded-lg p-2 relative">
                          <svg width="100%" height="100%" className="overflow-visible">
                            {/* Simple line chart of scores */}
                            {Object.entries(analysisResults[sessionState.current_photo_id].all_scores).map(([angle, score], i, arr) => {
                              const x = (i / (arr.length - 1)) * 100
                              const y = 100 - ((score - Math.min(...Object.values(analysisResults[sessionState.current_photo_id].all_scores))) / 
                                (Math.max(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)) - Math.min(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)))) * 80
                              
                              if (i === 0) return null
                              
                              const prevEntry = arr[i - 1]
                              const prevX = ((i - 1) / (arr.length - 1)) * 100
                              const prevY = 100 - ((prevEntry[1] - Math.min(...Object.values(analysisResults[sessionState.current_photo_id].all_scores))) / 
                                (Math.max(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)) - Math.min(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)))) * 80
                              
                              return (
                                <line
                                  key={angle}
                                  x1={`${prevX}%`}
                                  y1={`${prevY}%`}
                                  x2={`${x}%`}
                                  y2={`${y}%`}
                                  stroke="rgb(59 130 246)"
                                  strokeWidth="2"
                                />
                              )
                            })}
                            
                            {/* Mark optimal point */}
                            {analysisResults[sessionState.current_photo_id].optimal_angle && (
                              <circle
                                cx={`${(Object.keys(analysisResults[sessionState.current_photo_id].all_scores).indexOf(analysisResults[sessionState.current_photo_id].optimal_angle.toString()) / 
                                  (Object.keys(analysisResults[sessionState.current_photo_id].all_scores).length - 1)) * 100}%`}
                                cy={`${100 - ((analysisResults[sessionState.current_photo_id].optimal_score - Math.min(...Object.values(analysisResults[sessionState.current_photo_id].all_scores))) / 
                                  (Math.max(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)) - Math.min(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)))) * 80}%`}
                                r="3"
                                fill="rgb(239 68 68)"
                              />
                            )}
                          </svg>
                        </div>
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>{minRotationAngle}°</span>
                          <span>Optimal: {analysisResults[sessionState.current_photo_id].optimal_score?.toFixed(3)}</span>
                          <span>{maxRotationAngle}°</span>
                        </div>
                      </div>
                    )}

                    {/* CV Analysis Info */}
                    {rotationMethod === 'cv' && analysisResults[sessionState.current_photo_id] && (
                      <div className="space-y-2">
                        <p className="text-xs font-medium">Detection Info</p>
                        <div className="bg-muted rounded-lg p-3 text-xs space-y-1">
                          <div className="flex justify-between">
                            <span>Method Used:</span>
                            <span className="font-mono capitalize">{analysisResults[sessionState.current_photo_id].method_used}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Scene Type:</span>
                            <span className="font-mono capitalize">{analysisResults[sessionState.current_photo_id].scene_type}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Confidence:</span>
                            <span className="font-mono">{(analysisResults[sessionState.current_photo_id].optimal_score * 100).toFixed(1)}%</span>
                          </div>
                          {analysisResults[sessionState.current_photo_id].needs_perspective_correction && (
                            <div className="flex justify-between text-amber-600">
                              <span>Perspective:</span>
                              <span className="font-mono">Needs Correction</span>
                            </div>
                          )}
                          {analysisResults[sessionState.current_photo_id].needs_skew_correction && (
                            <div className="flex justify-between text-amber-600">
                              <span>Skew:</span>
                              <span className="font-mono">Needs Correction</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Raw Data Table */}
                    {analysisResults[sessionState.current_photo_id] && (
                      <div className="space-y-2">
                        <p className="text-xs font-medium">Raw Values</p>
                        <div className="bg-muted rounded-lg p-3 text-xs space-y-1">
                          <div className="flex justify-between">
                            <span>Optimal Angle:</span>
                            <span className="font-mono">{analysisResults[sessionState.current_photo_id].optimal_angle?.toFixed(2)}°</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Max Score:</span>
                            <span className="font-mono">{analysisResults[sessionState.current_photo_id].optimal_score?.toFixed(4)}</span>
                          </div>
                          {analysisResults[sessionState.current_photo_id].all_scores && (
                            <>
                              <div className="flex justify-between">
                                <span>Angles Tested:</span>
                                <span className="font-mono">{Object.keys(analysisResults[sessionState.current_photo_id].all_scores).length}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Score Range:</span>
                                <span className="font-mono">
                                  {Math.min(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)).toFixed(3)} - {Math.max(...Object.values(analysisResults[sessionState.current_photo_id].all_scores)).toFixed(3)}
                                </span>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Status */}
                    <div className="space-y-2">
                      <p className="text-xs font-medium">Status</p>
                      <div className="bg-muted rounded-lg p-3 text-xs">
                        {analysisProgress > 0 && analysisProgress < 100 ? (
                          <div className="text-blue-600">
                            <div className="flex items-center gap-2">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              <span className="capitalize">{analysisPhase === 'generating' ? 'Generating' : 'Analyzing'}</span>
                            </div>
                            <div className="mt-1 text-muted-foreground">
                              {analysisProgress.toFixed(1)}% complete
                            </div>
                            {analysisMessage && (
                              <div className="mt-1 text-xs text-muted-foreground">
                                {analysisMessage}
                              </div>
                            )}
                          </div>
                        ) : analysisResults[sessionState.current_photo_id] ? (
                          <div className="text-green-600">
                            ✓ Analysis Complete
                          </div>
                        ) : (
                          <div className="text-gray-500">
                            Ready to analyze
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            )}
          </div>
        )}

        {/* Continue Button */}
        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={() => setCurrentStep('crop')}>
            Skip Rotation
          </Button>
          <Button 
            onClick={applyRotation}
            disabled={!showAnalysisResults || Object.keys(analysisResults).length < selectedPhotos.length}
          >
            Apply Results & Continue
          </Button>
        </div>
      </div>
    )
  }

  if (currentStep === 'crop') {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Step 2: Intelligent Crop & Composition</h1>
          <p className="text-muted-foreground">
            Photo {currentPhotoIndex + 1} of {selectedPhotos.length} - AI-powered composition analysis and cropping
          </p>
        </div>

        {/* Crop Mode Selection */}
        <Card className="p-4">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Sparkles className="h-5 w-5" />
              <span className="font-medium">Cropping Mode</span>
            </div>

            <div className="flex space-x-2">
              <Button
                variant={cropMode === 'intelligent' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setCropMode('intelligent')}
              >
                <Sparkles className="h-4 w-4 mr-2" />
                AI Intelligent Crop
              </Button>
              <Button
                variant={cropMode === 'manual' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setCropMode('manual')}
              >
                <SlidersHorizontal className="h-4 w-4 mr-2" />
                Manual Crop
              </Button>
            </div>
          </div>
        </Card>

        {/* Intelligent Crop Controls */}
        {cropMode === 'intelligent' && (
          <Card className="p-6">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <Star className="h-5 w-5" />
                <span className="font-medium">AI Analysis & Generation</span>
              </div>

              {/* Step 1: Composition Analysis */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium">Step 1: Composition Analysis</h3>
                    <p className="text-sm text-muted-foreground">Analyze photo content and composition</p>
                  </div>
                  <Button
                    onClick={analyzeComposition}
                    disabled={isAnalyzingComposition || !ollamaModel}
                    size="sm"
                  >
                    {isAnalyzingComposition ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Search className="h-4 w-4 mr-2" />
                        Analyze Photo
                      </>
                    )}
                  </Button>
                </div>

                {/* Display Round 1 Results */}
                {compositionAnalysis && compositionAnalysis.round1 && (
                  <div className="bg-muted/50 rounded-lg p-4 space-y-2">
                    <div>
                      <h4 className="text-sm font-medium">Photo Description:</h4>
                      <p className="text-sm">{compositionAnalysis.round1.photo_description}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium">Crop Directions:</h4>
                      <p className="text-sm">{compositionAnalysis.round1.crop_directions}</p>
                    </div>
                  </div>
                )}

                {/* Display Round 2 Results */}
                {compositionAnalysis && compositionAnalysis.round2 && (
                  <div className="bg-muted/50 rounded-lg p-4 space-y-2">
                    <div>
                      <h4 className="text-sm font-medium">Crop Details:</h4>
                      <p className="text-sm">Aspect Ratio: {compositionAnalysis.round2.aspect_ratio}</p>
                      <p className="text-sm">Confidence: {(compositionAnalysis.round2.confidence * 100).toFixed(1)}%</p>
                      <p className="text-sm">Coordinates: {compositionAnalysis.round2.crop_coordinates.x1_px}, {compositionAnalysis.round2.crop_coordinates.y1_px} to {compositionAnalysis.round2.crop_coordinates.x2_px}, {compositionAnalysis.round2.crop_coordinates.y2_px}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Model Selection */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Ollama Vision Model</label>
                {loadingModels ? (
                  <div className="flex items-center space-x-2 px-3 py-2 border rounded-md">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm text-muted-foreground">Loading available models...</span>
                  </div>
                ) : availableModels.length > 0 ? (
                  <select
                    value={ollamaModel}
                    onChange={(e) => setOllamaModel(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md bg-background"
                  >
                    <option value="">Select a model...</option>
                    {availableModels.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name} ({model.size_gb}GB)
                        {model.description ? ` - ${model.description}` : ''}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="px-3 py-2 border rounded-md bg-muted">
                    <p className="text-sm text-muted-foreground">No vision models available. Please pull a vision model in Ollama.</p>
                  </div>
                )}
              </div>
            </div>
          </Card>
        )}


        {/* Visual Results Display */}
        {showVlmResults && compositionAnalysis && sessionState && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Analysis Results</h2>
            
            <Card className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Original Image */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-center">Original</h4>
                  <div className="aspect-square bg-muted rounded-lg flex items-center justify-center relative">
                    <img
                      src={`/api/photos/${sessionState.current_photo_id}/thumbnail`}
                      alt="Original"
                      className="max-w-full max-h-full object-contain rounded-lg"
                      onError={(e) => {
                        e.currentTarget.style.display = 'none'
                        e.currentTarget.nextElementSibling.style.display = 'flex'
                      }}
                    />
                    <div className="hidden w-full h-full items-center justify-center">
                      <ImageIcon className="h-8 w-8 text-muted-foreground" />
                    </div>
                  </div>
                </div>

                {/* Cropped Preview with Overlay */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-center">
                    {compositionAnalysis?.preview_url ? 'Intelligent Crop Preview' : 'Ready for Crop'}
                  </h4>
                  <div className="aspect-square bg-muted rounded-lg flex items-center justify-center relative">
                    {compositionAnalysis?.preview_url ? (
                      <img
                        src={`${compositionAnalysis.preview_url}`}
                        alt="Crop preview with overlay"
                        className="max-w-full max-h-full object-contain rounded-lg"
                      />
                    ) : (
                      <div className="text-center">
                        <ImageIcon className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                        <p className="text-xs">Generate crop to see preview</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Analysis Data */}
                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-center">Analysis Data</h4>
                  
                  {/* Composition Analysis */}
                  {compositionAnalysis && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium">Photo Description</p>
                      <div className="bg-muted rounded-lg p-3 text-xs">
                        <p>{compositionAnalysis?.round1?.photo_description || 'Loading...'}</p>
                      </div>
                      
                      <p className="text-xs font-medium">Main Subjects</p>
                      <div className="bg-muted rounded-lg p-3 text-xs">
                        {compositionAnalysis?.round1?.main_subjects?.map((subject, i) => (
                          <Badge key={i} variant="secondary" className="mr-1 mb-1">
                            {subject}
                          </Badge>
                        )) || <span>Loading...</span>}
                      </div>

                      <p className="text-xs font-medium">Composition Type</p>
                      <div className="bg-muted rounded-lg p-3 text-xs">
                        <Badge variant="outline">{compositionAnalysis?.round1?.composition_type || 'unknown'}</Badge>
                      </div>
                    </div>
                  )}

                  {/* Crop Results from Round 2 */}
                  {compositionAnalysis?.round2 && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium">Crop Analysis</p>
                      <div className="bg-muted rounded-lg p-3 text-xs">
                        <p>Aspect Ratio: {compositionAnalysis.round2.aspect_ratio}</p>
                        <p>Confidence: {(compositionAnalysis.round2.confidence * 100).toFixed(1)}%</p>
                      </div>
                      
                      <p className="text-xs font-medium">Crop Area in Original Image</p>
                      <div className="bg-muted rounded-lg p-3 text-xs space-y-2">
                        <div className="border-b pb-2 mb-2">
                          <p className="text-xs text-muted-foreground mb-1">Position in Original Image:</p>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span>Top-Left (X1, Y1):</span>
                              <span className="font-mono">({compositionAnalysis.round2.crop_coordinates.x1_px}, {compositionAnalysis.round2.crop_coordinates.y1_px})</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Bottom-Right (X2, Y2):</span>
                              <span className="font-mono">({compositionAnalysis.round2.crop_coordinates.x2_px}, {compositionAnalysis.round2.crop_coordinates.y2_px})</span>
                            </div>
                          </div>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground mb-1">Resulting Crop Dimensions:</p>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span>Width:</span>
                              <span className="font-mono">{compositionAnalysis.round2.crop_coordinates.x2_px - compositionAnalysis.round2.crop_coordinates.x1_px}px</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Height:</span>
                              <span className="font-mono">{compositionAnalysis.round2.crop_coordinates.y2_px - compositionAnalysis.round2.crop_coordinates.y1_px}px</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Actual Ratio:</span>
                              <span className="font-mono">{((compositionAnalysis.round2.crop_coordinates.x2_px - compositionAnalysis.round2.crop_coordinates.x1_px) / (compositionAnalysis.round2.crop_coordinates.y2_px - compositionAnalysis.round2.crop_coordinates.y1_px)).toFixed(2)}:1</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Continue Button */}
        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={() => setCurrentStep('enhance')}>
            Skip Crop
          </Button>
          <Button 
            onClick={applyCrop}
            disabled={cropMode === 'intelligent' && !compositionAnalysis?.round2}
          >
            Apply & Continue
          </Button>
        </div>
      </div>
    )
  }

  if (currentStep === 'enhance') {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Step 3: Enhancement</h1>
          <p className="text-muted-foreground">
            Photo {currentPhotoIndex + 1} of {selectedPhotos.length}
          </p>
        </div>

        <Card className="p-6">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Sparkles className="h-5 w-5" />
              <span className="font-medium">Image Enhancement</span>
            </div>

            {/* Side-by-side preview */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-center">Original</h4>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
                  {enhancementPreview.originalUrl || sessionState?.temp_image_url ? (
                    <img
                      src={enhancementPreview.originalUrl || sessionState?.temp_image_url}
                      alt="Original processed image"
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <div className="text-center">
                      <ImageIcon className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                      <p className="text-xs text-muted-foreground">Loading original...</p>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-center">Enhanced ({enhancementStrength}%)</h4>
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center relative overflow-hidden">
                  {enhancementPreview.previewUrl ? (
                    <img
                      src={enhancementPreview.previewUrl}
                      alt="Enhanced preview"
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <div className="text-center">
                      <ImageIcon className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                      <p className="text-xs text-muted-foreground">Click "Generate Preview" to see enhanced version</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Enhancement Mode Selection */}
            <div className="space-y-4">
              <div className="flex gap-2">
                <Button
                  variant={enhancementMode === 'intelligent' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setEnhancementMode('intelligent')}
                >
                  Intelligent Mode
                </Button>
                <Button
                  variant={enhancementMode === 'custom' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setEnhancementMode('custom')}
                >
                  Custom Mode
                </Button>
              </div>

              {/* Overall Strength Slider */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Overall Strength</label>
                  <span className="text-sm text-muted-foreground">
                    {enhancementStrength}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="200"
                  value={enhancementStrength}
                  onChange={(e) => setEnhancementStrength(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>0% (Original)</span>
                  <span>100% (Balanced)</span>
                  <span>200% (Intense)</span>
                </div>
              </div>
              
              {enhancementMode === 'intelligent' ? (
                <div className="bg-muted rounded-lg p-3 text-xs space-y-1">
                  <p><strong>Core Enhancement Features:</strong></p>
                  <ul className="list-disc list-inside space-y-0.5">
                    <li>Statistical white balance correction</li>
                    <li>Tone-mapped exposure optimization</li>
                    <li>Guided filter contrast enhancement</li>
                    <li>Perceptual color vibrancy</li>
                    <li>Shadow and highlight recovery</li>
                  </ul>
                  <p className="text-muted-foreground mt-2">
                    <em>Focus on color and tonal adjustments without modifying image content.</em>
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-sm font-medium">Individual Controls:</p>
                  
                  {/* White Balance */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={customEnhancements.white_balance}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, white_balance: e.target.checked})}
                        className="h-4 w-4"
                      />
                      <label className="text-xs font-medium">White Balance</label>
                    </div>
                    {customEnhancements.white_balance && (
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={customEnhancements.white_balance_strength}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, white_balance_strength: parseFloat(e.target.value)})}
                        className="w-full h-1"
                      />
                    )}
                  </div>

                  {/* Exposure */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={customEnhancements.exposure}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, exposure: e.target.checked})}
                        className="h-4 w-4"
                      />
                      <label className="text-xs font-medium">Exposure Optimization</label>
                    </div>
                    {customEnhancements.exposure && (
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={customEnhancements.exposure_strength}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, exposure_strength: parseFloat(e.target.value)})}
                        className="w-full h-1"
                      />
                    )}
                  </div>

                  {/* Contrast */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={customEnhancements.contrast}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, contrast: e.target.checked})}
                        className="h-4 w-4"
                      />
                      <label className="text-xs font-medium">Local Contrast</label>
                    </div>
                    {customEnhancements.contrast && (
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={customEnhancements.contrast_strength}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, contrast_strength: parseFloat(e.target.value)})}
                        className="w-full h-1"
                      />
                    )}
                  </div>

                  {/* Vibrance */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={customEnhancements.vibrance}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, vibrance: e.target.checked})}
                        className="h-4 w-4"
                      />
                      <label className="text-xs font-medium">Color Vibrance</label>
                    </div>
                    {customEnhancements.vibrance && (
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={customEnhancements.vibrance_strength}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, vibrance_strength: parseFloat(e.target.value)})}
                        className="w-full h-1"
                      />
                    )}
                  </div>

                  {/* Shadow/Highlight Recovery */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={customEnhancements.shadow_highlight}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, shadow_highlight: e.target.checked})}
                        className="h-4 w-4"
                      />
                      <label className="text-xs font-medium">Shadow/Highlight Recovery</label>
                    </div>
                    {customEnhancements.shadow_highlight && (
                      <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        value={customEnhancements.shadow_highlight_strength}
                        onChange={(e) => setCustomEnhancements({...customEnhancements, shadow_highlight_strength: parseFloat(e.target.value)})}
                        className="w-full h-1"
                      />
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-end space-x-2">
              <Button 
                variant="outline" 
                onClick={() => {
                  setEnhancementStrength(100)
                  setEnhancementPreview({})
                }}
              >
                Reset
              </Button>
              <Button 
                variant="outline"
                onClick={generateEnhancementPreview}
                disabled={isGeneratingPreview}
              >
                {isGeneratingPreview ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  'Generate Preview'
                )}
              </Button>
              <Button 
                onClick={applyEnhancement}
                disabled={!enhancementPreview.previewUrl}
              >
                Apply & {currentPhotoIndex < selectedPhotos.length - 1 ? 'Next Photo' : 'Complete'}
              </Button>
            </div>
          </div>
        </Card>
      </div>
    )
  }

  if (currentStep === 'complete') {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Recipe Complete!</h1>
          <p className="text-muted-foreground">
            Your custom recipe has been created from {selectedPhotos.length} sample photos
          </p>
        </div>

        <Card className="p-6">
          <div className="space-y-4">
            <div className="text-center py-8">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Save className="h-8 w-8 text-green-600" />
              </div>
              <h2 className="text-lg font-semibold mb-2">{recipeName}</h2>
              <p className="text-muted-foreground">{recipeDescription}</p>
            </div>

            <div className="bg-muted rounded-lg p-4">
              <h3 className="font-medium mb-2">Recipe Summary</h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Rotation: Auto-detect enabled</li>
                <li>• Crop: {aspectRatio} aspect ratio</li>
                <li>• Enhancement: Intelligent enhancement ({enhancementStrength}% strength)</li>
              </ul>
            </div>

            <div className="flex justify-center space-x-2">
              <Button variant="outline" onClick={() => {
                setCurrentStep('select')
                setSelectedPhotos([])
                setSessionId(null)
              }}>
                Create Another
              </Button>
              <Button onClick={saveRecipe}>
                <Save className="h-4 w-4 mr-2" />
                Save Recipe
              </Button>
            </div>
          </div>
        </Card>
      </div>
    )
  }

  return null
}