import { render, screen } from '@/test/test-utils'
import Layout from '../Layout'

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  LayoutDashboard: () => <div data-testid="icon-layout-dashboard" />,
  Image: () => <div data-testid="icon-images" />,
  Images: () => <div data-testid="icon-images" />,
  Cpu: () => <div data-testid="icon-cpu" />,
  BookOpen: () => <div data-testid="icon-book-open" />,
  Settings: () => <div data-testid="icon-settings" />,
  Camera: () => <div data-testid="icon-camera" />,
  Activity: () => <div data-testid="icon-activity" />
}))

// Mock the ConnectionStatus component
vi.mock('../ConnectionStatus', () => ({
  default: () => <div data-testid="connection-status">Connected</div>
}))

// Mock the useRealTimeUpdates hook
vi.mock('@/hooks/useRealTimeUpdates', () => ({
  useRealTimeUpdates: () => ({
    isConnected: true,
    status: 'connected'
  })
}))

describe('Layout', () => {
  it('renders navigation menu', () => {
    render(
      <Layout>
        <div>Test content</div>
      </Layout>
    )

    // Check for navigation items - use role to be more specific
    const navLinks = screen.getAllByRole('link')
    const linkTexts = navLinks.map(link => link.textContent)
    
    expect(linkTexts).toContain('Dashboard')
    expect(linkTexts).toContain('Photos')
    expect(linkTexts).toContain('Processing')
    expect(linkTexts).toContain('Recipes')
    expect(linkTexts).toContain('Settings')
  })

  it('renders main content', () => {
    render(
      <Layout>
        <div data-testid="main-content">Test content</div>
      </Layout>
    )

    expect(screen.getByTestId('main-content')).toBeInTheDocument()
  })

  it('shows connection status', () => {
    render(
      <Layout>
        <div>Test content</div>
      </Layout>
    )

    expect(screen.getByTestId('connection-status')).toBeInTheDocument()
  })

  it('displays app title', () => {
    render(
      <Layout>
        <div>Test content</div>
      </Layout>
    )

    expect(screen.getByText('Photo Processor')).toBeInTheDocument()
  })

  it('shows real-time monitoring indicator', () => {
    render(
      <Layout>
        <div>Test content</div>
      </Layout>
    )

    expect(screen.getByText('Real-time monitoring active')).toBeInTheDocument()
  })
})