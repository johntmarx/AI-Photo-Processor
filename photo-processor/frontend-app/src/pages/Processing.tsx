import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { processingApi, photosApi, recipesApi } from '@/services/api'
import { Photo, Recipe } from '@/types/api'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { Badge } from '@/components/ui/Badge'
import { Input } from '@/components/ui/Input'
import PhotoGrid from '@/components/photos/PhotoGrid'
import { formatDuration, formatRelativeTime } from '@/lib/utils'
import { 
  Play, 
  Pause, 
  Settings,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Zap,
  Search,
  ChevronDown,
  ChevronUp,
  Wand2
} from 'lucide-react'

export default function Processing() {
  const queryClient = useQueryClient()
  const [selectedPhotos, setSelectedPhotos] = useState<Set<string>>(new Set())
  const [selectedRecipe, setSelectedRecipe] = useState<string>('')
  const [statusFilter, setStatusFilter] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [showPhotoSelection, setShowPhotoSelection] = useState(true)
  const [page, setPage] = useState(1)

  // Existing queries for processing status
  const { data: queueStatus } = useQuery(
    ['processing', 'queue'],
    () => processingApi.getQueueStatus().then(res => res.data),
    { refetchInterval: 2000 }
  )

  const { data: processingStatus } = useQuery(
    ['processing', 'status'],
    () => processingApi.getStatus().then(res => res.data),
    { refetchInterval: 1000 }
  )

  const { data: settings } = useQuery(
    ['processing', 'settings'],
    () => processingApi.getSettings().then(res => res.data)
  )

  // New queries for photos and recipes
  const { data: photosData, isLoading: photosLoading } = useQuery(
    ['photos', page, statusFilter, searchQuery],
    () => photosApi.list(page, 20, statusFilter || undefined).then(res => res.data),
    { keepPreviousData: true }
  )

  const { data: recipesData } = useQuery(
    ['recipes'],
    () => recipesApi.list().then(res => res.data)
  )

  const recipes: Recipe[] = Array.isArray(recipesData) ? recipesData : (recipesData?.recipes || [])
  const photos: Photo[] = Array.isArray(photosData) ? photosData : (photosData?.photos || [])

  // Mutations
  const pauseMutation = useMutation(
    () => processingApi.pause(),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['processing', 'status'])
        toast.success('Processing paused')
      },
      onError: () => toast.error('Failed to pause processing')
    }
  )

  const resumeMutation = useMutation(
    () => processingApi.resume(),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['processing', 'status'])
        toast.success('Processing resumed')
      },
      onError: () => toast.error('Failed to resume processing')
    }
  )

  const batchProcessMutation = useMutation(
    ({ photoIds, recipeId }: { photoIds: string[], recipeId: string }) =>
      processingApi.batchProcess(photoIds, recipeId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['processing', 'queue'])
        queryClient.invalidateQueries(['photos'])
        toast.success(`Added ${selectedPhotos.size} photos to processing queue`)
        setSelectedPhotos(new Set())
      },
      onError: () => toast.error('Failed to queue photos for processing')
    }
  )

  const handleToggleProcessing = () => {
    if (!processingStatus?.is_paused) {
      pauseMutation.mutate()
    } else {
      resumeMutation.mutate()
    }
  }

  const handlePhotoSelect = (photo: Photo, selected: boolean) => {
    const newSelection = new Set(selectedPhotos)
    if (selected) {
      newSelection.add(photo.id)
    } else {
      newSelection.delete(photo.id)
    }
    setSelectedPhotos(newSelection)
  }

  const handleSelectAll = () => {
    if (selectedPhotos.size === photos.length) {
      setSelectedPhotos(new Set())
    } else {
      setSelectedPhotos(new Set(photos.map(p => p.id)))
    }
  }

  const handleProcessSelected = () => {
    if (selectedPhotos.size === 0) {
      toast.error('Please select photos to process')
      return
    }
    if (!selectedRecipe) {
      toast.error('Please select a recipe')
      return
    }

    batchProcessMutation.mutate({
      photoIds: Array.from(selectedPhotos),
      recipeId: selectedRecipe
    })
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'processing':
        return <Zap className="h-4 w-4 text-blue-500 animate-pulse" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'processing':
        return <Badge variant="secondary">Processing</Badge>
      case 'completed':
        return <Badge variant="success">Completed</Badge>
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>
      default:
        return <Badge variant="outline">Pending</Badge>
    }
  }

  const statusFilters = [
    { label: 'All Photos', value: '' },
    { label: 'Unprocessed', value: 'pending' },
    { label: 'Completed', value: 'completed' },
    { label: 'Failed', value: 'failed' },
  ]

  return (
    <div className="space-y-6">
      {/* Processing Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Processing Control</span>
            <div className="flex items-center space-x-2">
              <Button
                variant={!processingStatus?.is_paused ? "destructive" : "default"}
                size="sm"
                onClick={handleToggleProcessing}
                disabled={pauseMutation.isLoading || resumeMutation.isLoading}
              >
                {!processingStatus?.is_paused ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Resume
                  </>
                )}
              </Button>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-2">
              <p className="text-sm font-medium">Status</p>
              <div className="flex items-center space-x-2">
                {!processingStatus?.is_paused ? (
                  <Zap className="h-4 w-4 text-green-500" />
                ) : (
                  <Pause className="h-4 w-4 text-yellow-500" />
                )}
                <span className="text-sm">
                  {!processingStatus?.is_paused ? 'Running' : 'Paused'}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium">Current Progress</p>
              <div className="space-y-1">
                <Progress value={processingStatus?.progress || 0} />
                <p className="text-xs text-muted-foreground">
                  {processingStatus?.progress || 0}% complete
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium">Processing Rate</p>
              <p className="text-lg font-semibold">
                {processingStatus?.processing_rate?.toFixed(1) || '0'}/min
              </p>
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium">Average Time</p>
              <p className="text-lg font-semibold">
                {formatDuration(processingStatus?.average_time || 0)}
              </p>
            </div>
          </div>

          {processingStatus?.current_photo && (
            <div className="mt-4 p-3 bg-muted rounded-lg">
              <p className="text-sm font-medium">Currently Processing:</p>
              <p className="text-sm text-muted-foreground">
                {processingStatus.current_photo}
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Photo Selection and Recipe Application */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Wand2 className="h-5 w-5" />
              <span>Apply Recipe to Photos</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowPhotoSelection(!showPhotoSelection)}
            >
              {showPhotoSelection ? (
                <>
                  <ChevronUp className="h-4 w-4 mr-1" />
                  Hide
                </>
              ) : (
                <>
                  <ChevronDown className="h-4 w-4 mr-1" />
                  Show
                </>
              )}
            </Button>
          </CardTitle>
        </CardHeader>
        
        {showPhotoSelection && (
          <CardContent className="space-y-4">
            {/* Recipe Selection and Actions */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1">
                <select
                  value={selectedRecipe}
                  onChange={(e) => setSelectedRecipe(e.target.value)}
                  className="w-64 px-3 py-2 border rounded-md"
                >
                  <option value="">Select a recipe...</option>
                  {recipes.map((recipe) => (
                    <option key={recipe.id} value={recipe.id}>
                      {recipe.name}
                    </option>
                  ))}
                </select>
                
                <Button
                  onClick={handleProcessSelected}
                  disabled={selectedPhotos.size === 0 || !selectedRecipe || batchProcessMutation.isLoading}
                >
                  <Wand2 className="h-4 w-4 mr-2" />
                  Process {selectedPhotos.size} Photos
                </Button>

                {selectedPhotos.size > 0 && (
                  <Button
                    variant="outline"
                    onClick={() => setSelectedPhotos(new Set())}
                  >
                    Clear Selection
                  </Button>
                )}
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={handleSelectAll}
              >
                {selectedPhotos.size === photos.length ? 'Deselect All' : 'Select All'}
              </Button>
            </div>

            {/* Filters */}
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
                {statusFilters.map((filter) => (
                  <Button
                    key={filter.value}
                    variant={statusFilter === filter.value ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setStatusFilter(filter.value)}
                  >
                    {filter.label}
                  </Button>
                ))}
              </div>
            </div>

            {/* Photo Grid */}
            <div className="border rounded-lg p-4 max-h-[600px] overflow-y-auto">
              {photosLoading ? (
                <div className="text-center py-8">Loading photos...</div>
              ) : (
                <PhotoGrid
                  photos={photos}
                  onPhotoSelect={handlePhotoSelect}
                  selectedPhotos={selectedPhotos}
                  showActions={false}
                />
              )}
            </div>

            {/* Pagination */}
            {photosData && 'pages' in photosData && photosData.pages && photosData.pages > 1 && (
              <div className="flex items-center justify-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  Previous
                </Button>
                <span className="text-sm">
                  Page {page} of {photosData.pages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => p + 1)}
                  disabled={page === photosData.pages}
                >
                  Next
                </Button>
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* Queue Overview */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total</p>
                <p className="text-2xl font-bold">{((queueStatus?.pending?.length || 0) + (queueStatus?.processing?.length || 0) + (queueStatus?.completed?.length || 0))}</p>
              </div>
              <Clock className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Pending</p>
                <p className="text-2xl font-bold">{queueStatus?.pending?.length || 0}</p>
              </div>
              <AlertCircle className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Processing</p>
                <p className="text-2xl font-bold">{queueStatus?.processing?.length || 0}</p>
              </div>
              <Zap className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold">{queueStatus?.completed?.length || 0}</p>
              </div>
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Current Queue Item */}
      {queueStatus?.processing?.[0] && (
        <Card>
          <CardHeader>
            <CardTitle>Current Item</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="font-medium">
                  {queueStatus.processing[0]?.filename || `Photo ID: ${queueStatus.processing[0]?.photo_id || 'Unknown'}`}
                </p>
                <p className="text-sm text-muted-foreground">
                  Started: {formatRelativeTime(new Date(queueStatus.processing[0].started_at || ''))}
                </p>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusIcon(queueStatus.processing[0].status || 'processing')}
                {getStatusBadge(queueStatus.processing[0].status || 'processing')}
              </div>
            </div>
            <div className="mt-4">
              <Progress value={queueStatus.processing[0].progress || 0} />
              <p className="text-xs text-muted-foreground mt-1">
                {queueStatus.processing[0].progress || 0}% complete
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processing Settings */}
      {settings && (
        <Card>
          <CardHeader>
            <CardTitle>Processing Settings</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Auto Process</span>
                <Badge variant={settings.auto_process ? "success" : "secondary"}>
                  {settings.auto_process ? "Enabled" : "Disabled"}
                </Badge>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Quality Threshold</span>
                <span className="text-sm">{settings.quality_threshold}/10</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Max Concurrent</span>
                <span className="text-sm">{settings.max_concurrent} photos</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Pause on Error</span>
                <Badge variant={settings.pause_on_error ? "destructive" : "success"}>
                  {settings.pause_on_error ? "Yes" : "No"}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}