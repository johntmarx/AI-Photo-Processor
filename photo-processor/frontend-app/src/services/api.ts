import axios from 'axios'
import { 
  Photo, 
  PhotoDetail, 
  PhotoComparison,
  QueueStatus, 
  ProcessingStatus, 
  ProcessingSettings,
  Recipe,
  RecipePreset,
  DashboardStats,
  ProcessingStats,
  StorageStats,
  ActivityItem,
  PaginatedResponse
} from '@/types/api'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// Request interceptor for logging
api.interceptors.request.use((config) => {
  console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Photos API
export const photosApi = {
  list: (params: {
    page?: number
    per_page?: number
    status?: string
    search?: string
    sort?: string
    min_aesthetic_score?: number
    max_aesthetic_score?: number
    min_technical_score?: number
    max_technical_score?: number
  }) => 
    api.get<PaginatedResponse<Photo>>('/photos', { 
      params: {
        page: params.page || 1,
        per_page: params.per_page || 20,
        ...params
      } 
    }),

  get: (id: string) => 
    api.get<PhotoDetail>(`/photos/${id}`),

  getComparison: (id: string) => 
    api.get<PhotoComparison>(`/photos/${id}/comparison`),

  // Batch upload system for large file uploads
  createUploadSession: (expectedFiles: number, totalSize?: number, recipeId?: string, autoProcess = true) =>
    api.post('/upload/session', {
      expected_files: expectedFiles,
      total_size: totalSize,
      recipe_id: recipeId,
      auto_process: autoProcess
    }),

  uploadFileToSession: (sessionId: string, file: File, recipeId?: string, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    formData.append('file', file)
    if (recipeId) formData.append('recipe_id', recipeId)
    formData.append('auto_process', 'true')
    
    return api.post(`/upload/session/${sessionId}/file`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minute timeout for large files
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      }
    })
  },

  getUploadSessionProgress: (sessionId: string) =>
    api.get(`/upload/session/${sessionId}/progress`),

  completeUploadSession: (sessionId: string) =>
    api.post(`/upload/session/${sessionId}/complete`),

  // Legacy single upload - deprecated
  upload: (file: File, recipeId?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (recipeId) formData.append('recipe_id', recipeId)
    
    return api.post<Photo>('/upload/single', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },

  delete: (id: string, deleteOriginal = false) => 
    api.delete(`/photos/${id}`, { 
      params: { delete_original: deleteOriginal } 
    }),

  reprocess: (id: string, recipeId: string) => 
    api.post(`/photos/${id}/reprocess`, { recipe_id: recipeId }),

  getAIAnalysis: (id: string) => 
    api.get(`/photos/${id}/ai-analysis`),
}

// Processing API
export const processingApi = {
  getQueueStatus: () => 
    api.get<QueueStatus>('/processing/queue'),

  getStatus: () => 
    api.get<ProcessingStatus>('/processing/status'),

  pause: () => 
    api.put('/processing/pause'),

  resume: () => 
    api.put('/processing/resume'),

  approve: (photoId: string) => 
    api.post('/processing/approve', { photo_id: photoId }),

  reject: (photoId: string, reason?: string) => 
    api.post('/processing/reject', { photo_id: photoId, reason }),

  batchProcess: (photoIds: string[], recipeId: string) => 
    api.post('/processing/batch', { 
      photo_ids: photoIds, 
      recipe_id: recipeId, 
      priority: 'normal',
      skip_ai: false 
    }),

  reorderQueue: (photoId: string, newPosition: number) => 
    api.post('/processing/reorder', { photo_id: photoId, new_position: newPosition }),

  removeFromQueue: (photoId: string) => 
    api.delete(`/processing/queue/${photoId}`),

  getSettings: () => 
    api.get<ProcessingSettings>('/processing/settings'),

  updateSettings: (settings: Partial<ProcessingSettings>) => 
    api.put('/processing/settings', settings),
}

// Recipes API
export const recipesApi = {
  list: () => 
    api.get<{ recipes: Recipe[]; total: number; page: number; pageSize: number }>('/recipes'),

  get: (id: string) => 
    api.get<Recipe>(`/recipes/${id}`),

  create: (recipe: Omit<Recipe, 'id' | 'created_at' | 'updated_at' | 'usage_count'>) => 
    api.post<Recipe>('/recipes', recipe),

  update: (id: string, recipe: Partial<Recipe>) => 
    api.put<Recipe>(`/recipes/${id}`, recipe),

  delete: (id: string) => 
    api.delete(`/recipes/${id}`),

  duplicate: (id: string, name: string) => 
    api.post<Recipe>(`/recipes/${id}/duplicate`, { name }),

  applyToPhotos: (id: string, photoIds: string[]) => 
    api.post(`/recipes/${id}/apply`, { photo_ids: photoIds }),

  preview: (id: string, photoId: string) => 
    api.post(`/recipes/${id}/preview`, { photo_id: photoId }),

  getPresets: () => 
    api.get<{ presets: RecipePreset[] }>('/recipes/presets'),
}

// Statistics API
export const statsApi = {
  getDashboard: () => 
    api.get<DashboardStats>('/stats/dashboard'),

  getProcessing: (days = 7) => 
    api.get<ProcessingStats>('/stats/processing', { params: { days } }),

  getStorage: () => 
    api.get<StorageStats>('/stats/storage'),

  getActivity: (limit = 50) => 
    api.get<{ activities: ActivityItem[] }>('/stats/activity', { params: { limit } }),

  getPerformance: (hours = 24) => 
    api.get('/stats/performance', { params: { hours } }),

  getTrends: (days = 30) => 
    api.get('/stats/trends', { params: { days } }),

  getAIPerformance: () => 
    api.get('/stats/ai-performance'),

  getErrors: (days = 7) => 
    api.get('/stats/errors', { params: { days } }),
}

export default api