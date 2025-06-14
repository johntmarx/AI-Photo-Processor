// API Response Types
export interface Photo {
  id: string
  filename: string
  originalPath: string
  processedPath?: string
  thumbnailPath?: string
  webPath?: string
  fileSize: number
  createdAt: string
  processedAt?: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'rejected'
}

export interface PhotoDetail extends Photo {
  metadata?: Record<string, any>
  processingHistory?: ProcessingRecord[]
  recipeId?: string
  aiAnalysis?: AIAnalysis
  errorMessage?: string
  processingTime?: number
}

export interface PhotoComparison {
  original: Photo
  processed?: Photo
  differences: ComparisonMetrics
}

export interface ComparisonMetrics {
  size_difference: number
  quality_improvement: number
  processing_time: number
}

export interface AIAnalysis {
  id?: number
  photoId?: string
  analysisType?: string
  status?: 'pending' | 'processing' | 'completed' | 'failed' | 'not_available' | 'error'
  queuedAt?: string | null
  startedAt?: string | null
  completedAt?: string | null
  aestheticScore?: number
  technicalScore?: number | null
  qualityLevel?: string
  confidence?: number
  nimaResults?: {
    aesthetic?: {
      qualityScore?: number
      qualityStd?: number
      qualityDistribution?: number[]
      qualityLevel?: string
      confidence?: number
      modelType?: string
      inferenceTime?: number
      imageSize?: number[]
    }
    technical?: {
      qualityScore?: number
      qualityStd?: number
      qualityDistribution?: number[]
      qualityLevel?: string
      confidence?: number
      modelType?: string
      inferenceTime?: number
      imageSize?: number[]
    }
  }
  modelInfo?: {
    aestheticModel?: string
    technicalModel?: string
    inferenceTime?: number
  }
  error?: string | null
  taskId?: string | null
  // Combined score (average of quality and aesthetics)
  combined_score?: number
  // Legacy fields for backward compatibility
  subjects?: string[]
  composition_score?: number
  suggested_crop?: {
    x: number
    y: number
    width: number
    height: number
  }
  recommendations?: string[]
}

export interface ProcessingRecord {
  id: string
  timestamp: string
  action: string
  recipe_id?: string
  result: 'success' | 'failed'
  details: string
}

// Processing Queue Types
export interface QueueItem {
  photo_id: string
  filename?: string
  position?: number
  added_at?: string
  created_at?: string
  priority: 'low' | 'normal' | 'high'
  recipe_id?: string
  manual_approval?: boolean
  estimated_time?: number
  // For current_item
  status?: 'pending' | 'processing' | 'completed' | 'failed'
  started_at?: string
  progress?: number
}

export interface QueueStatus {
  pending: QueueItem[]
  processing: QueueItem[]
  completed: QueueItem[]
  is_paused: boolean
  stats: ProcessingStatus
  // Computed client-side
  total?: number
  current_item?: QueueItem
}

export interface ProcessingStatus {
  is_running?: boolean
  is_paused?: boolean
  current_photo?: any
  queue_length?: number
  progress?: number
  processing_rate?: number
  average_time?: number
  estimated_completion?: string
  errors_today?: number
}

export interface ProcessingSettings {
  auto_process: boolean
  quality_threshold: number
  max_concurrent: number
  pause_on_error: boolean
}

// Recipe Types
export interface Recipe {
  id: string
  name: string
  description: string
  steps: ProcessingStep[]
  created_at: string
  updated_at: string
  is_preset: boolean
  usage_count: number
}

export interface ProcessingStep {
  operation: string
  parameters: Record<string, any>
  enabled: boolean
}

export interface RecipePreset {
  id: string
  name: string
  description: string
  category?: string
  steps?: ProcessingStep[]
  operations?: ProcessingStep[]
}

// Statistics Types
export interface DashboardStats {
  // Support both snake_case and camelCase from API
  total_photos?: number
  totalPhotos?: number
  processed_today?: number
  processedToday?: number
  processing_rate?: number
  processingRate?: number
  storage_used?: number
  storageUsed?: {
    total?: { bytes: number; formatted: string }
    originals?: { bytes: number; formatted: string }
    processed?: { bytes: number; formatted: string }
    disk?: { total: string; used: string; free: string; percentUsed: number }
  }
  queue_length?: number
  inQueue?: number
  success_rate?: number
  successRate?: number
  averageProcessingTime?: number
  failedToday?: number
  recentActivity?: any[]
  systemStatus?: {
    processing: string
    aiModels: string
    storage: string
  }
}

export interface ProcessingStats {
  total_processed: number
  failed_count: number
  average_time: number
  processing_rate: number
  queue_stats: QueueStatus
}

export interface StorageStats {
  total_size: number
  originals_size: number
  processed_size: number
  available_space: number
  usage_by_month: Array<{
    month: string
    size: number
  }>
}

export interface ActivityItem {
  id: string
  type: 'photo_processed' | 'recipe_created' | 'error' | 'warning'
  message: string
  timestamp: string
  photo_id?: string
  recipe_id?: string
}

// WebSocket Event Types
export interface WebSocketEvent {
  type: 'processing_started' | 'processing_completed' | 'processing_failed' | 
        'queue_updated' | 'stats_updated' | 'photo_uploaded' | 'recipe_updated' |
        'rotation_analysis_progress' | 'rotation_analysis_complete' | 'rotation_analysis_failed' |
        'recipe_builder_started' | 'connection'
  data: any
  timestamp: string
}

// API Error Types
export interface APIError {
  detail: string
  status_code: number
}

// Pagination
export interface PaginatedResponse<T> {
  items?: T[]  // Generic version
  photos?: T[]  // Photos API specific
  total: number
  page: number
  per_page?: number
  page_size?: number
  pages?: number
  has_next?: boolean
  has_prev?: boolean
}