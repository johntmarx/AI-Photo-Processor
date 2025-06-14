import { render, screen, waitFor } from '@/test/test-utils'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Dashboard from '../Dashboard'

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Image: () => <div data-testid="icon-images" />,
  Images: () => <div data-testid="icon-images" />,
  Clock: () => <div data-testid="icon-clock" />,
  HardDrive: () => <div data-testid="icon-hard-drive" />,
  TrendingUp: () => <div data-testid="icon-trending-up" />,
  CheckCircle: () => <div data-testid="icon-check-circle" />,
  AlertCircle: () => <div data-testid="icon-alert-circle" />,
  ImageIcon: () => <div data-testid="icon-image" />,
  BookOpen: () => <div data-testid="icon-book-open" />,
  XCircle: () => <div data-testid="icon-x-circle" />,
  Info: () => <div data-testid="icon-info" />
}))

// Mock child components
vi.mock('@/components/dashboard/StatsCard', () => ({
  default: ({ title, value, description }: any) => (
    <div data-testid="stats-card">
      <div>{title}</div>
      <div>{value}</div>
      <div>{description}</div>
    </div>
  )
}))

vi.mock('@/components/dashboard/ProcessingChart', () => ({
  default: ({ title }: any) => (
    <div data-testid="processing-chart">
      <div>{title || 'Processing Activity'}</div>
    </div>
  )
}))

vi.mock('@/components/dashboard/ActivityFeed', () => ({
  default: ({ title }: any) => (
    <div data-testid="activity-feed">
      <div>{title || 'Recent Activity'}</div>
    </div>
  )
}))

// Mock utilities
vi.mock('@/lib/utils', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  formatBytes: (bytes: number) => `${bytes} bytes`,
  formatDuration: (seconds: number) => `${seconds}s`,
  formatRelativeTime: () => '1m ago'
}))

// Mock the API
vi.mock('@/services/api', () => ({
  statsApi: {
    getDashboard: vi.fn(() => Promise.resolve({
      data: {
        total_photos: 150,
        processed_today: 25,
        processing_rate: 2.5,
        queue_length: 3,
        success_rate: 0.95
      }
    })),
    getProcessing: vi.fn(() => Promise.resolve({
      data: {
        total_processed: 1000,
        failed_count: 50,
        average_time: 45
      }
    })),
    getStorage: vi.fn(() => Promise.resolve({
      data: {
        total_size: 10485760,
        available_space: 5242880
      }
    })),
    getActivity: vi.fn(() => Promise.resolve({
      data: []
    })),
    getPerformance: vi.fn(() => Promise.resolve({
      data: {}
    }))
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

describe('Dashboard', () => {
  it('renders dashboard with stats cards', async () => {
    renderWithClient(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText('Total Photos')).toBeInTheDocument()
      expect(screen.getByText('Processed Today')).toBeInTheDocument()
      expect(screen.getByText('Processing Rate')).toBeInTheDocument()
      expect(screen.getByText('Storage Used')).toBeInTheDocument()
    })
  })

  it('displays correct stat values', async () => {
    renderWithClient(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument()
      expect(screen.getByText('25')).toBeInTheDocument()
      expect(screen.getByText('2.5/min')).toBeInTheDocument()
      expect(screen.getByText('3')).toBeInTheDocument()
      expect(screen.getByText('10485760 bytes')).toBeInTheDocument() // Storage used
      expect(screen.getByText('95.0%')).toBeInTheDocument() // Success rate
    })
  })

  it('shows loading state initially', () => {
    renderWithClient(<Dashboard />)

    // The Dashboard component renders loading skeletons with animate-pulse
    const loadingElements = document.querySelectorAll('.animate-pulse')
    
    // Should show 6 loading skeleton cards initially
    expect(loadingElements).toHaveLength(6)
  })

  it('renders processing chart', async () => {
    renderWithClient(<Dashboard />)

    await waitFor(() => {
      // Chart component should be present (mocked by recharts)
      expect(screen.getByText('Processing Activity')).toBeInTheDocument()
    })
  })

  it('renders activity feed', async () => {
    renderWithClient(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText('Recent Activity')).toBeInTheDocument()
    })
  })
})