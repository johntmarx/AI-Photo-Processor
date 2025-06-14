import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { formatRelativeTime } from '@/lib/utils'
import { ActivityItem } from '@/types/api'
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Info, 
  ImageIcon,
  BookOpen
} from 'lucide-react'

interface ActivityFeedProps {
  activities: ActivityItem[]
  title?: string
}

const getActivityIcon = (type: ActivityItem['type']) => {
  switch (type) {
    case 'photo_processed':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'recipe_created':
      return <BookOpen className="h-4 w-4 text-blue-500" />
    case 'error':
      return <XCircle className="h-4 w-4 text-red-500" />
    case 'warning':
      return <AlertCircle className="h-4 w-4 text-yellow-500" />
    default:
      return <Info className="h-4 w-4 text-gray-500" />
  }
}

const getActivityBadge = (type: ActivityItem['type']) => {
  switch (type) {
    case 'photo_processed':
      return <Badge variant="success" className="text-xs">Processed</Badge>
    case 'recipe_created':
      return <Badge variant="secondary" className="text-xs">Recipe</Badge>
    case 'error':
      return <Badge variant="destructive" className="text-xs">Error</Badge>
    case 'warning':
      return <Badge variant="outline" className="text-xs">Warning</Badge>
    default:
      return <Badge variant="secondary" className="text-xs">Info</Badge>
  }
}

export default function ActivityFeed({ activities, title = "Recent Activity" }: ActivityFeedProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px]">
          <div className="space-y-4">
            {activities.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                <ImageIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p>No recent activity</p>
              </div>
            ) : (
              activities.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    {getActivityIcon(activity.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-foreground truncate">
                        {activity.message}
                      </p>
                      {getActivityBadge(activity.type)}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatRelativeTime(new Date(activity.timestamp))}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}