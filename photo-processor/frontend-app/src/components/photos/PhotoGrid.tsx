import { useState, useEffect, useRef } from 'react'
import { Photo } from '@/types/api'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent } from '@/components/ui/Card'
import { formatRelativeTime, formatBytes } from '@/lib/utils'
import { cn } from '@/lib/utils'
import { 
  CheckCircle, 
  XCircle, 
  Clock, 
  Zap,
  Eye,
  Trash2,
  RotateCcw,
  Loader2
} from 'lucide-react'
import { Button } from '@/components/ui/Button'

// Thumbnail retry configuration
const RETRY_DELAYS = [1000, 2000, 4000, 8000, 16000] // Exponential backoff
const MAX_RETRIES = 5

interface ThumbnailWithRetryProps {
  photo: Photo
  className?: string
}

// Component to handle thumbnail loading with retry logic
function ThumbnailWithRetry({ photo, className }: ThumbnailWithRetryProps) {
  const [imageStatus, setImageStatus] = useState<'loading' | 'loaded' | 'error'>('loading')
  const [retryCount, setRetryCount] = useState(0)
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current)
      }
    }
  }, [])

  // Reset retry count when photo changes
  useEffect(() => {
    setRetryCount(0)
    setImageStatus('loading')
  }, [photo.id])

  // Start retry immediately for processing/pending photos
  useEffect(() => {
    if ((photo.status === 'processing' || photo.status === 'pending') && imageStatus === 'loading') {
      // Trigger initial load attempt after a short delay to ensure image element exists
      const checkTimer = setTimeout(() => {
        const img = document.getElementById(`thumbnail-${photo.id}`) as HTMLImageElement
        if (img && img.complete && img.naturalHeight === 0) {
          // Image failed to load, start retry
          handleImageError()
        }
      }, 100)
      
      return () => clearTimeout(checkTimer)
    }
  }, [photo.status, photo.id, imageStatus, retryCount]) // Added dependencies

  const handleImageError = async () => {
    if (!mountedRef.current) return

    // Check if this is a 202 response (processing)
    const img = document.getElementById(`thumbnail-${photo.id}`) as HTMLImageElement
    if (img) {
      try {
        // Fetch the URL to check the response status
        const response = await fetch(img.src)
        if (response.status === 202) {
          // Server says it's still processing
          const data = await response.json()
          const retryAfter = data.retry_after || 1
          
          if (retryCount < MAX_RETRIES) {
            retryTimeoutRef.current = setTimeout(() => {
              if (!mountedRef.current) return
              
              // Force image reload by adding timestamp
              const url = new URL(img.src)
              url.searchParams.set('retry', Date.now().toString())
              img.src = url.toString()
              
              setRetryCount(prev => prev + 1)
            }, retryAfter * 1000)
            return
          }
        }
      } catch (e) {
        // Fall through to normal error handling
      }
    }

    // Only retry if photo is in processing or pending status and we haven't exceeded max retries
    if ((photo.status === 'processing' || photo.status === 'pending') && retryCount < MAX_RETRIES) {
      const delay = RETRY_DELAYS[Math.min(retryCount, RETRY_DELAYS.length - 1)]
      
      retryTimeoutRef.current = setTimeout(() => {
        if (!mountedRef.current) return
        
        // Force image reload by adding timestamp
        const img = document.getElementById(`thumbnail-${photo.id}`) as HTMLImageElement
        if (img) {
          const url = new URL(img.src)
          url.searchParams.set('retry', Date.now().toString())
          img.src = url.toString()
        }
        
        setRetryCount(prev => prev + 1)
      }, delay)
    } else {
      // If photo is completed/failed or max retries exceeded, show error state
      setImageStatus('error')
    }
  }

  const handleImageLoad = () => {
    if (!mountedRef.current) return
    setImageStatus('loaded')
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current)
    }
  }

  return (
    <>
      {/* Loading state for processing/pending photos */}
      {imageStatus === 'loading' && (photo.status === 'processing' || photo.status === 'pending') && (
        <div className="absolute inset-0 bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="h-8 w-8 text-gray-400 animate-spin mx-auto mb-2" />
            <p className="text-xs text-gray-500">
              {photo.status === 'processing' ? 'Processing...' : 'Waiting...'}
            </p>
            {retryCount > 0 && (
              <p className="text-xs text-gray-400 mt-1">
                Retry {retryCount}/{MAX_RETRIES}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Actual image */}
      <img
        id={`thumbnail-${photo.id}`}
        src={`/api/photos/${photo.id}/thumbnail`}
        alt={photo.filename}
        className={cn(
          className,
          imageStatus === 'loading' && (photo.status === 'processing' || photo.status === 'pending') ? 'opacity-0' : 'opacity-100',
          'transition-opacity duration-300'
        )}
        loading="lazy"
        onLoad={handleImageLoad}
        onError={handleImageError}
      />

      {/* Error state - only show for truly failed loads */}
      {imageStatus === 'error' && (
        <div className="absolute inset-0 bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
          <div className="text-center">
            <Eye className="h-12 w-12 text-gray-400 mb-2" />
            <p className="text-xs text-gray-500">
              {photo.status === 'failed' || photo.status === 'rejected' 
                ? 'Processing failed' 
                : 'Thumbnail unavailable'}
            </p>
          </div>
        </div>
      )}
    </>
  )
}

interface PhotoGridProps {
  photos: Photo[]
  onPhotoClick?: (photo: Photo) => void
  onPhotoSelect?: (photo: Photo, selected: boolean) => void
  onDelete?: (photoId: string) => void
  onReprocess?: (photoId: string) => void
  selectedPhotos?: Set<string>
  showActions?: boolean
}

const getStatusIcon = (status: Photo['status']) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'processing':
      return <Zap className="h-4 w-4 text-blue-500 animate-pulse" />
    case 'failed':
    case 'rejected':
      return <XCircle className="h-4 w-4 text-red-500" />
    case 'pending':
      return <Clock className="h-4 w-4 text-yellow-500" />
    default:
      return null
  }
}

const getStatusBadge = (status: Photo['status']) => {
  switch (status) {
    case 'completed':
      return <Badge variant="success">Completed</Badge>
    case 'processing':
      return <Badge variant="secondary">Processing</Badge>
    case 'failed':
      return <Badge variant="destructive">Failed</Badge>
    case 'rejected':
      return <Badge variant="destructive">Rejected</Badge>
    case 'pending':
      return <Badge variant="outline">Pending</Badge>
    default:
      return null
  }
}

export default function PhotoGrid({ 
  photos, 
  onPhotoClick, 
  onPhotoSelect,
  onDelete,
  onReprocess,
  selectedPhotos = new Set(),
  showActions = false
}: PhotoGridProps) {
  const [hoveredPhoto, setHoveredPhoto] = useState<string | null>(null)
  const [, forceUpdate] = useState(0) // Force re-render when needed

  // Listen for WebSocket events to trigger thumbnail retry
  useEffect(() => {
    const handleProcessingCompleted = (event: CustomEvent) => {
      const photoId = event.detail?.data?.photo_id
      if (photoId && photos.some(p => p.id === photoId)) {
        // Force component re-render to retry thumbnail loading
        forceUpdate(prev => prev + 1)
      }
    }

    window.addEventListener('processing_completed', handleProcessingCompleted as EventListener)
    
    return () => {
      window.removeEventListener('processing_completed', handleProcessingCompleted as EventListener)
    }
  }, [photos])

  const handlePhotoClick = (photo: Photo) => {
    onPhotoClick?.(photo)
  }

  const handleSelectChange = (photo: Photo, selected: boolean) => {
    onPhotoSelect?.(photo, selected)
  }

  if (photos.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-muted-foreground">
          <Eye className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium">No photos found</p>
          <p className="text-sm">Upload some photos to get started</p>
        </div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {photos.map((photo) => (
        <Card 
          key={photo.id}
          className={cn(
            "cursor-pointer transition-all duration-200 hover:shadow-lg",
            selectedPhotos.has(photo.id) && "ring-2 ring-primary"
          )}
          onMouseEnter={() => setHoveredPhoto(photo.id)}
          onMouseLeave={() => setHoveredPhoto(null)}
        >
          <CardContent className="p-0">
            {/* Photo Image */}
            <div className="aspect-square bg-muted relative overflow-hidden rounded-t-lg">
              {/* Use ThumbnailWithRetry component for smart loading */}
              <ThumbnailWithRetry 
                photo={photo} 
                className="w-full h-full object-cover"
              />
              
              {/* Status overlay */}
              <div className="absolute top-2 left-2">
                {getStatusIcon(photo.status)}
              </div>

              {/* Selection checkbox */}
              {onPhotoSelect && (
                <div 
                  className="absolute top-2 right-2 z-30"
                  style={{ pointerEvents: 'auto' }}
                >
                  <button
                    type="button"
                    className="flex items-center justify-center w-8 h-8 bg-white/95 rounded-md shadow-lg border-2 border-gray-300 hover:bg-white hover:border-blue-400 cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      console.log('Checkbox clicked for photo:', photo.id, 'current selected:', selectedPhotos.has(photo.id))
                      handleSelectChange(photo, !selectedPhotos.has(photo.id))
                    }}
                    onMouseDown={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                    }}
                    aria-label={`${selectedPhotos.has(photo.id) ? 'Deselect' : 'Select'} photo ${photo.filename}`}
                  >
                    {selectedPhotos.has(photo.id) ? (
                      <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      <div className="w-5 h-5 border-2 border-gray-400 rounded-sm bg-white"></div>
                    )}
                  </button>
                </div>
              )}

              {/* Action buttons on hover */}
              {showActions && hoveredPhoto === photo.id && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center space-x-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={(e) => {
                      e.stopPropagation()
                      handlePhotoClick(photo)
                    }}
                  >
                    <Eye className="h-4 w-4" />
                  </Button>
                  {photo.status === 'failed' && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation()
                        onReprocess?.(photo.id)
                      }}
                    >
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete?.(photo.id)
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>

            {/* Photo Details */}
            <div 
              className="p-4 space-y-2"
              onClick={() => handlePhotoClick(photo)}
            >
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-sm truncate flex-1 mr-2">
                  {photo.filename}
                </h3>
                {getStatusBadge(photo.status)}
              </div>

              <div className="text-xs text-muted-foreground space-y-1">
                <div className="flex justify-between">
                  <span>Size:</span>
                  <span>{formatBytes(photo.fileSize)}</span>
                </div>
                
                <div className="flex justify-between">
                  <span>Created:</span>
                  <span>{formatRelativeTime(photo.createdAt)}</span>
                </div>

                {photo.processedAt && (
                  <div className="flex justify-between">
                    <span>Processed:</span>
                    <span>{formatRelativeTime(photo.processedAt)}</span>
                  </div>
                )}

              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}