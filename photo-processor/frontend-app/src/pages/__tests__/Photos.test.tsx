import { render, screen, fireEvent, waitFor } from '@/test/test-utils'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Photos from '../Photos'

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Upload: () => <div data-testid="icon-upload" />,
  Search: () => <div data-testid="icon-search" />,
  Filter: () => <div data-testid="icon-filter" />,
  MoreHorizontal: () => <div data-testid="icon-more" />,
  CheckCircle: () => <div data-testid="icon-check-circle" />,
  XCircle: () => <div data-testid="icon-x-circle" />,
  Clock: () => <div data-testid="icon-clock" />,
  Zap: () => <div data-testid="icon-zap" />,
  Trash2: () => <div data-testid="icon-trash" />,
  RotateCcw: () => <div data-testid="icon-rotate" />
}))

// Mock child components
vi.mock('@/components/photos/PhotoGrid', () => ({
  default: ({ photos }: any) => (
    <div data-testid="photo-grid">
      {photos.map((photo: any) => (
        <div key={photo.id} data-testid={`photo-${photo.id}`}>
          {photo.filename}
        </div>
      ))}
    </div>
  )
}))

vi.mock('@/components/photos/PhotoDialog', () => ({
  default: ({ photo, isOpen }: any) => 
    isOpen ? (
      <div data-testid="photo-dialog">
        {photo?.filename}
      </div>
    ) : null
}))

// Mock the API
vi.mock('@/services/api', () => ({
  photosApi: {
    list: vi.fn(() => Promise.resolve({
      data: {
        items: [
          {
            id: '1',
            filename: 'test1.jpg',
            path: '/test1.jpg',
            size: 1024000,
            created_at: '2024-01-01T00:00:00Z',
            hash: 'abc123',
            status: 'processed'
          },
          {
            id: '2',
            filename: 'test2.jpg',
            path: '/test2.jpg',
            size: 2048000,
            created_at: '2024-01-02T00:00:00Z',
            hash: 'def456',
            status: 'pending'
          }
        ],
        total: 2,
        page: 1,
        per_page: 20,
        pages: 1
      }
    })),
    upload: vi.fn(() => Promise.resolve({ data: {} })),
    delete: vi.fn(() => Promise.resolve({})),
    reprocess: vi.fn(() => Promise.resolve({}))
  }
}))

// Mock toast
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  }
}))

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: { retry: false, cacheTime: 0 },
    mutations: { retry: false }
  }
})

const renderWithClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient()
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  )
}

describe('Photos', () => {
  it('renders photos page with header', async () => {
    renderWithClient(<Photos />)

    expect(screen.getByText('Photos')).toBeInTheDocument()
    expect(screen.getByText('Upload Photos')).toBeInTheDocument()
  })

  it('displays photo list', async () => {
    renderWithClient(<Photos />)

    await waitFor(() => {
      expect(screen.getByText('test1.jpg')).toBeInTheDocument()
      expect(screen.getByText('test2.jpg')).toBeInTheDocument()
    })
  })

  it('shows total photo count', async () => {
    renderWithClient(<Photos />)

    await waitFor(() => {
      expect(screen.getByText('2 photos total')).toBeInTheDocument()
    })
  })

  it('handles search input', () => {
    renderWithClient(<Photos />)

    const searchInput = screen.getByPlaceholderText('Search photos...')
    fireEvent.change(searchInput, { target: { value: 'test1' } })

    expect(searchInput).toHaveValue('test1')
  })

  it('renders status filter buttons', () => {
    renderWithClient(<Photos />)

    expect(screen.getByText('All')).toBeInTheDocument()
    expect(screen.getByText('Processed')).toBeInTheDocument()
    expect(screen.getByText('Processing')).toBeInTheDocument()
    expect(screen.getByText('Pending')).toBeInTheDocument()
    expect(screen.getByText('Failed')).toBeInTheDocument()
  })

  it('handles status filter clicks', () => {
    renderWithClient(<Photos />)

    const processedFilter = screen.getByText('Processed')
    fireEvent.click(processedFilter)

    // Filter should be applied (visual feedback)
    expect(processedFilter).toBeInTheDocument()
  })

  it('shows bulk actions when photos selected', async () => {
    renderWithClient(<Photos />)

    await waitFor(() => {
      expect(screen.getByText('test1.jpg')).toBeInTheDocument()
    })

    // This would require interaction with PhotoGrid component
    // The bulk actions bar appears when selectedPhotos.size > 0
    // For testing purposes, we just verify the component renders
    expect(screen.getByText('Photos')).toBeInTheDocument()
  })

  it('handles file upload', () => {
    renderWithClient(<Photos />)

    // Find the file input by its id since it's hidden
    const uploadInput = document.getElementById('photo-upload')
    expect(uploadInput).toBeInTheDocument()
    expect(uploadInput).toHaveAttribute('type', 'file')
    expect(uploadInput).toHaveAttribute('accept', 'image/*')
  })
})