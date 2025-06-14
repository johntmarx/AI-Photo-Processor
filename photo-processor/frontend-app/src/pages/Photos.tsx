import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { photosApi } from '@/services/api'
import { Photo } from '@/types/api'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import PhotoGrid from '@/components/photos/PhotoGrid'
import PhotoDialog from '@/components/photos/PhotoDialog'
import BatchUpload from '@/components/photos/BatchUpload'
import { 
  Upload, 
  Search, 
  Filter, 
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  Trash2,
  RotateCcw,
  SlidersHorizontal,
  ArrowUpDown,
  Star,
  Camera
} from 'lucide-react'

const statusFilters = [
  { label: 'All', value: '', icon: null },
  { label: 'Completed', value: 'completed', icon: CheckCircle },
  { label: 'Processing', value: 'processing', icon: Zap },
  { label: 'Pending', value: 'pending', icon: Clock },
  { label: 'Failed', value: 'failed', icon: XCircle },
]

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

export default function Photos() {
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null)
  const [selectedPhotos, setSelectedPhotos] = useState<Set<string>>(new Set())
  const [statusFilter, setStatusFilter] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(1)
  const [showUpload, setShowUpload] = useState(false)
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false)
  const [sortBy, setSortBy] = useState('created_at_desc')
  const [minAestheticScore, setMinAestheticScore] = useState<number | ''>('')
  const [maxAestheticScore, setMaxAestheticScore] = useState<number | ''>('')
  const [minTechnicalScore, setMinTechnicalScore] = useState<number | ''>('')
  const [maxTechnicalScore, setMaxTechnicalScore] = useState<number | ''>('')
  const queryClient = useQueryClient()

  const { data: photosData, isLoading } = useQuery(
    ['photos', page, statusFilter, searchQuery, sortBy, minAestheticScore, maxAestheticScore, minTechnicalScore, maxTechnicalScore],
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
    { keepPreviousData: true }
  )

  const deleteMutation = useMutation(
    (photoId: string) => photosApi.delete(photoId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['photos'])
        toast.success('Photo deleted successfully')
        setSelectedPhoto(null)
      },
      onError: () => {
        toast.error('Failed to delete photo')
      }
    }
  )

  const reprocessMutation = useMutation(
    ({ photoId, recipeId }: { photoId: string; recipeId: string }) => 
      photosApi.reprocess(photoId, recipeId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['photos'])
        toast.success('Photo queued for reprocessing')
        setSelectedPhoto(null)
      },
      onError: () => {
        toast.error('Failed to reprocess photo')
      }
    }
  )


  const handlePhotoClick = (photo: Photo) => {
    console.log('Photo clicked:', photo)
    setSelectedPhoto(photo)
  }

  const handlePhotoSelect = (photo: Photo, selected: boolean) => {
    const newSelected = new Set(selectedPhotos)
    if (selected) {
      newSelected.add(photo.id)
    } else {
      newSelected.delete(photo.id)
    }
    setSelectedPhotos(newSelected)
  }

  const handleSelectAll = () => {
    if (selectedPhotos.size === (photosData?.photos || photosData?.items || []).length) {
      setSelectedPhotos(new Set())
    } else {
      setSelectedPhotos(new Set((photosData?.photos || photosData?.items || []).map(p => p.id)))
    }
  }

  const handleBulkDelete = () => {
    if (selectedPhotos.size === 0) return
    
    if (confirm(`Delete ${selectedPhotos.size} photos?`)) {
      // Note: This would need batch delete API endpoint
      selectedPhotos.forEach(photoId => {
        deleteMutation.mutate(photoId)
      })
      setSelectedPhotos(new Set())
    }
  }


  const handleReprocess = (photoId: string) => {
    // For now, use a default recipe ID - this would come from a recipe selector
    reprocessMutation.mutate({ photoId, recipeId: 'default' })
  }

  const handleDelete = (photoId: string) => {
    if (confirm('Delete this photo?')) {
      deleteMutation.mutate(photoId)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Photos</h1>
          <p className="text-muted-foreground">
            {photosData?.total || 0} photos total
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button 
            onClick={() => setShowUpload(!showUpload)}
            variant={showUpload ? "default" : "outline"}
          >
            <Upload className="h-4 w-4 mr-2" />
            {showUpload ? 'Hide Upload' : 'Upload Photos'}
          </Button>
        </div>
      </div>

      {/* Batch Upload */}
      {showUpload && (
        <BatchUpload 
          onUploadComplete={() => {
            setShowUpload(false)
            queryClient.invalidateQueries(['photos'])
          }}
        />
      )}

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
                }}
              >
                Clear Filters
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Bulk Actions */}
      {selectedPhotos.size > 0 && (
        <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium">
              {selectedPhotos.size} photo{selectedPhotos.size > 1 ? 's' : ''} selected
            </span>
            <Button variant="outline" size="sm" onClick={handleSelectAll}>
              {selectedPhotos.size === (photosData?.photos || photosData?.items || []).length ? 'Deselect All' : 'Select All'}
            </Button>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <RotateCcw className="h-4 w-4 mr-1" />
              Reprocess
            </Button>
            <Button variant="destructive" size="sm" onClick={handleBulkDelete}>
              <Trash2 className="h-4 w-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>
      )}

      {/* Photo Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="aspect-square bg-muted animate-pulse rounded-lg" />
          ))}
        </div>
      ) : (
        <PhotoGrid
          photos={photosData?.photos || photosData?.items || []}
          onPhotoClick={handlePhotoClick}
          onPhotoSelect={handlePhotoSelect}
          onDelete={handleDelete}
          onReprocess={handleReprocess}
          selectedPhotos={selectedPhotos}
          showActions={true}
        />
      )}

      {/* Pagination */}
      {photosData && (photosData.total > 20 || photosData.hasNext || photosData.hasPrev || page > 1) && (
        <div className="flex justify-center space-x-2 mt-6 pb-4">
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

      {/* Photo Detail Dialog */}
      <PhotoDialog
        photo={selectedPhoto}
        isOpen={!!selectedPhoto}
        onClose={() => setSelectedPhoto(null)}
        onReprocess={handleReprocess}
        onDelete={handleDelete}
      />
    </div>
  )
}