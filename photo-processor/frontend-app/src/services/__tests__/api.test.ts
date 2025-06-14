// Mock axios before importing api
vi.mock('axios', () => {
  const mockApi = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() }
    }
  }
  
  return {
    default: {
      create: vi.fn(() => mockApi)
    }
  }
})

import axios from 'axios'
import { photosApi, processingApi, recipesApi, statsApi } from '../api'

// Get the mocked api instance
const mockApi = (axios.create as any)()

describe('API Services', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('photosApi', () => {
    it('calls list endpoint with correct parameters', async () => {
      mockApi.get.mockResolvedValue({ data: { items: [], total: 0 } })

      await photosApi.list(1, 20, 'processed')

      expect(mockApi.get).toHaveBeenCalledWith('/photos/', {
        params: { page: 1, per_page: 20, status: 'processed' }
      })
    })

    it('calls get endpoint with photo id', async () => {
      mockApi.get.mockResolvedValue({ data: {} })

      await photosApi.get('123')

      expect(mockApi.get).toHaveBeenCalledWith('/photos/123')
    })

    it('calls upload endpoint with form data', async () => {
      mockApi.post.mockResolvedValue({ data: {} })

      const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' })
      await photosApi.upload(file, 'recipe-123')

      expect(mockApi.post).toHaveBeenCalledWith(
        '/photos/upload/',
        expect.any(FormData),
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
    })

    it('calls delete endpoint with correct parameters', async () => {
      mockApi.delete.mockResolvedValue({})

      await photosApi.delete('123', true)

      expect(mockApi.delete).toHaveBeenCalledWith('/photos/123', {
        params: { delete_original: true }
      })
    })
  })

  describe('processingApi', () => {
    it('calls queue status endpoint', async () => {
      mockApi.get.mockResolvedValue({ data: { total: 5 } })

      await processingApi.getQueueStatus()

      expect(mockApi.get).toHaveBeenCalledWith('/processing/queue/')
    })

    it('calls pause endpoint', async () => {
      mockApi.post.mockResolvedValue({})

      await processingApi.pause()

      expect(mockApi.post).toHaveBeenCalledWith('/processing/pause/')
    })

    it('calls approve endpoint with photo id', async () => {
      mockApi.post.mockResolvedValue({})

      await processingApi.approve('photo-123')

      expect(mockApi.post).toHaveBeenCalledWith('/processing/approve/', {
        photo_id: 'photo-123'
      })
    })
  })

  describe('recipesApi', () => {
    it('calls list endpoint', async () => {
      mockApi.get.mockResolvedValue({ data: [] })

      await recipesApi.list()

      expect(mockApi.get).toHaveBeenCalledWith('/recipes/')
    })

    it('calls create endpoint with recipe data', async () => {
      mockApi.post.mockResolvedValue({ data: {} })

      const recipe = {
        name: 'Test Recipe',
        description: 'Test description',
        steps: [],
        is_preset: false
      }

      await recipesApi.create(recipe)

      expect(mockApi.post).toHaveBeenCalledWith('/recipes/', recipe)
    })

    it('calls duplicate endpoint', async () => {
      mockApi.post.mockResolvedValue({ data: {} })

      await recipesApi.duplicate('recipe-123', 'New Name')

      expect(mockApi.post).toHaveBeenCalledWith('/recipes/recipe-123/duplicate/', {
        name: 'New Name'
      })
    })
  })

  describe('statsApi', () => {
    it('calls dashboard endpoint', async () => {
      mockApi.get.mockResolvedValue({ data: {} })

      await statsApi.getDashboard()

      expect(mockApi.get).toHaveBeenCalledWith('/stats/dashboard/')
    })

    it('calls processing stats with days parameter', async () => {
      mockApi.get.mockResolvedValue({ data: {} })

      await statsApi.getProcessing(14)

      expect(mockApi.get).toHaveBeenCalledWith('/stats/processing/', {
        params: { days: 14 }
      })
    })

    it('calls activity with limit parameter', async () => {
      mockApi.get.mockResolvedValue({ data: [] })

      await statsApi.getActivity(100)

      expect(mockApi.get).toHaveBeenCalledWith('/stats/activity/', {
        params: { limit: 100 }
      })
    })
  })
})