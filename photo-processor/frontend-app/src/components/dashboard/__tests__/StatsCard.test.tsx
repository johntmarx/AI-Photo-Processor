import { render, screen } from '@/test/test-utils'
import StatsCard from '../StatsCard'
import { CheckCircle } from 'lucide-react'

describe('StatsCard', () => {
  it('renders basic stats card', () => {
    render(
      <StatsCard
        title="Test Metric"
        value="100"
        description="Test description"
      />
    )

    expect(screen.getByText('Test Metric')).toBeInTheDocument()
    expect(screen.getByText('100')).toBeInTheDocument()
    expect(screen.getByText('Test description')).toBeInTheDocument()
  })

  it('renders with icon', () => {
    render(
      <StatsCard
        title="Test Metric"
        value="100"
        icon={<CheckCircle data-testid="test-icon" />}
      />
    )

    expect(screen.getByTestId('test-icon')).toBeInTheDocument()
  })

  it('renders with positive trend', () => {
    render(
      <StatsCard
        title="Test Metric"
        value="100"
        trend={{
          value: 12,
          label: "vs yesterday",
          positive: true
        }}
      />
    )

    expect(screen.getByText('+12%')).toBeInTheDocument()
    expect(screen.getByText('vs yesterday')).toBeInTheDocument()
    
    const trendElement = screen.getByText('+12%')
    expect(trendElement).toHaveClass('text-green-600')
  })

  it('renders with negative trend', () => {
    render(
      <StatsCard
        title="Test Metric"
        value="100"
        trend={{
          value: -5,
          label: "vs last week",
          positive: false
        }}
      />
    )

    expect(screen.getByText('-5%')).toBeInTheDocument()
    expect(screen.getByText('vs last week')).toBeInTheDocument()
    
    const trendElement = screen.getByText('-5%')
    expect(trendElement).toHaveClass('text-red-600')
  })

  it('accepts custom className', () => {
    render(
      <StatsCard
        title="Test Metric"
        value="100"
        className="custom-class"
      />
    )

    const card = screen.getByText('Test Metric').closest('.custom-class')
    expect(card).toBeInTheDocument()
  })

  it('handles numeric values', () => {
    render(
      <StatsCard
        title="Numeric Metric"
        value={42}
      />
    )

    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('renders without optional props', () => {
    render(
      <StatsCard
        title="Minimal Card"
        value="Simple"
      />
    )

    expect(screen.getByText('Minimal Card')).toBeInTheDocument()
    expect(screen.getByText('Simple')).toBeInTheDocument()
  })
})