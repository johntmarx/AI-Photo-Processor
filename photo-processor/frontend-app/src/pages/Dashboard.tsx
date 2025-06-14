import { useQuery } from '@tanstack/react-query'
import { statsApi } from '@/services/api'
import { formatBytes, formatDuration } from '@/lib/utils'
import StatsCard from '@/components/dashboard/StatsCard'
import ProcessingChart from '@/components/dashboard/ProcessingChart'
import ActivityFeed from '@/components/dashboard/ActivityFeed'
import { 
  Image as Images, 
  Clock, 
  HardDrive, 
  TrendingUp, 
  CheckCircle,
  AlertCircle
} from 'lucide-react'

export default function Dashboard() {
  const { data: dashboardStats, isLoading: statsLoading } = useQuery(
    ['stats', 'dashboard'],
    () => statsApi.getDashboard().then(res => res.data),
    { refetchInterval: 5000 } // Refresh every 5 seconds
  )

  const { data: processingStats } = useQuery(
    ['stats', 'processing'],
    () => statsApi.getProcessing().then(res => res.data),
    { refetchInterval: 5000 }
  )

  const { data: chartDataResponse } = useQuery(
    ['stats', 'processing', 'chart'],
    () => statsApi.getProcessing(1).then(res => res.data), // Last 24 hours for chart
    { refetchInterval: 30000 } // Refresh every 30 seconds
  )

  const { data: storageStats } = useQuery(
    ['stats', 'storage'],
    () => statsApi.getStorage().then(res => res.data),
    { refetchInterval: 30000 } // Refresh every 30 seconds
  )

  const { data: activityData } = useQuery(
    ['stats', 'activity'],
    () => statsApi.getActivity(20).then(res => res.data),
    { refetchInterval: 10000 } // Refresh every 10 seconds
  )

  // Use real chart data from API
  const chartData = (chartDataResponse as any)?.hourly_data || []

  if (statsLoading) {
    return (
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Total Photos"
          value={(dashboardStats?.totalPhotos || dashboardStats?.total_photos || 0).toLocaleString()}
          description="Photos in system"
          icon={<Images className="h-4 w-4" />}
        />
        
        <StatsCard
          title="Processed Today"
          value={dashboardStats?.processedToday || dashboardStats?.processed_today || 0}
          description="Photos processed in last 24h"
          icon={<CheckCircle className="h-4 w-4" />}
          trend={undefined}
        />

        <StatsCard
          title="Processing Rate"
          value={`${(dashboardStats?.processingRate || dashboardStats?.processing_rate || 0).toFixed(1)}/min`}
          description="Average processing speed"
          icon={<TrendingUp className="h-4 w-4" />}
        />

        <StatsCard
          title="Storage Used"
          value={formatBytes(dashboardStats?.storageUsed?.total?.bytes || storageStats?.total_size || 0)}
          description={`${dashboardStats?.storageUsed?.disk?.free || formatBytes(storageStats?.available_space || 0)} available`}
          icon={<HardDrive className="h-4 w-4" />}
        />

        <StatsCard
          title="Queue Length"
          value={dashboardStats?.inQueue || dashboardStats?.queue_length || 0}
          description="Photos waiting for processing"
          icon={<Clock className="h-4 w-4" />}
        />

        <StatsCard
          title="Success Rate"
          value={`${((dashboardStats?.successRate || dashboardStats?.success_rate || 0) * 100).toFixed(1)}%`}
          description="Processing success rate"
          icon={<CheckCircle className="h-4 w-4" />}
          trend={undefined}
        />
      </div>

      {/* Charts and Activity */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ProcessingChart data={chartData} />
        <ActivityFeed activities={activityData?.activities || []} />
      </div>

      {/* Processing Status */}
      {processingStats && (
        <div className="grid gap-6 md:grid-cols-3">
          <StatsCard
            title="Average Processing Time"
            value={formatDuration(processingStats?.average_time || 0)}
            description="Per photo"
            icon={<Clock className="h-4 w-4" />}
          />

          <StatsCard
            title="Failed Photos"
            value={processingStats?.failed_count || 0}
            description={`${(((processingStats?.failed_count || 0) / (processingStats?.total_processed || 1)) * 100).toFixed(1)}% failure rate`}
            icon={<AlertCircle className="h-4 w-4" />}
          />

          <StatsCard
            title="Total Processed"
            value={(processingStats?.total_processed || 0).toLocaleString()}
            description="All time"
            icon={<Images className="h-4 w-4" />}
          />
        </div>
      )}
    </div>
  )
}