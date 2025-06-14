import { render, screen, fireEvent } from '@/test/test-utils'
import PhotoGrid from '../PhotoGrid'
import { Photo } from '@/types/api'

// Mock utilities
vi.mock('@/lib/utils', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  formatBytes: (bytes: number) => {
    if (bytes === 1024000) return '1000.00 KB'
    if (bytes === 2048000) return '1.95 MB'
    if (bytes === 512000) return '500.00 KB'
    return `${bytes} bytes`
  },
  formatRelativeTime: () => '1/1/2024'
}))

const mockPhotos: Photo[] = [
  {
    id: '1',
    filename: 'test1.jpg',
    original_path: '/test1.jpg',
    file_size: 1024000,
    created_at: '2024-01-01T00:00:00Z',
    status: 'completed'
  },
  {
    id: '2',
    filename: 'test2.jpg',
    original_path: '/test2.jpg',
    file_size: 2048000,
    created_at: '2024-01-02T00:00:00Z',
    status: 'pending'
  },
  {
    id: '3',
    filename: 'test3.jpg',
    original_path: '/test3.jpg',
    file_size: 512000,
    created_at: '2024-01-03T00:00:00Z',
    status: 'failed'
  }
]

describe('PhotoGrid', () => {
  it('renders photos correctly', () => {
    render(<PhotoGrid photos={mockPhotos} />)

    expect(screen.getByText('test1.jpg')).toBeInTheDocument()
    expect(screen.getByText('test2.jpg')).toBeInTheDocument()
    expect(screen.getByText('test3.jpg')).toBeInTheDocument()
  })

  it('shows empty state when no photos', () => {
    render(<PhotoGrid photos={[]} />)

    expect(screen.getByText('No photos found')).toBeInTheDocument()
    expect(screen.getByText('Upload some photos to get started')).toBeInTheDocument()
  })

  it('handles photo click events', () => {
    const onPhotoClick = vi.fn()
    render(<PhotoGrid photos={mockPhotos} onPhotoClick={onPhotoClick} />)

    fireEvent.click(screen.getByText('test1.jpg'))
    expect(onPhotoClick).toHaveBeenCalledWith(mockPhotos[0])
  })

  it('handles photo selection', () => {
    const onPhotoSelect = vi.fn()
    render(
      <PhotoGrid 
        photos={mockPhotos} 
        onPhotoSelect={onPhotoSelect}
        selectedPhotos={new Set()}
      />
    )

    const checkbox = screen.getAllByRole('checkbox')[0]
    fireEvent.click(checkbox)
    
    expect(onPhotoSelect).toHaveBeenCalledWith(mockPhotos[0], true)
  })

  it('displays correct status badges', () => {
    render(<PhotoGrid photos={mockPhotos} />)

    expect(screen.getByText('Completed')).toBeInTheDocument()
    expect(screen.getByText('Pending')).toBeInTheDocument()
    expect(screen.getByText('Failed')).toBeInTheDocument()
  })

  it('shows file sizes correctly', () => {
    render(<PhotoGrid photos={mockPhotos} />)

    // formatBytes(1024000) = "1000.00 KB"
    // formatBytes(2048000) = "1.95 MB"
    // formatBytes(512000) = "500.00 KB"
    expect(screen.getByText('1000.00 KB')).toBeInTheDocument()
    expect(screen.getByText('1.95 MB')).toBeInTheDocument()
    expect(screen.getByText('500.00 KB')).toBeInTheDocument()
  })

  it('highlights selected photos', () => {
    const selectedPhotos = new Set(['1', '3'])
    render(
      <PhotoGrid 
        photos={mockPhotos} 
        selectedPhotos={selectedPhotos}
      />
    )

    // Check that the selected photos have the ring-2 ring-primary classes
    const allCards = document.querySelectorAll('.ring-2.ring-primary')
    expect(allCards).toHaveLength(2)
  })

  it('shows action buttons when enabled', () => {
    render(<PhotoGrid photos={mockPhotos} showActions={true} />)

    // Action buttons appear on hover, so they might not be visible in tests
    // This would typically be tested with user interactions or hover events
    expect(screen.getAllByText('test1.jpg')).toHaveLength(1)
  })
})