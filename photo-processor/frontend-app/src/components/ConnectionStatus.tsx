import { Wifi, WifiOff, AlertCircle } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { useWebSocketStatus } from '@/providers/WebSocketProvider'
import { useEffect, useState } from 'react'

export default function ConnectionStatus() {
  const { status } = useWebSocketStatus()
  const [apiHealth, setApiHealth] = useState<'checking' | 'healthy' | 'unhealthy'>('checking')

  useEffect(() => {
    // Check API health
    const checkHealth = async () => {
      try {
        const response = await fetch('/health')
        if (response.ok) {
          setApiHealth('healthy')
        } else {
          setApiHealth('unhealthy')
        }
      } catch (error) {
        setApiHealth('unhealthy')
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30 seconds

    return () => clearInterval(interval)
  }, [])

  const getStatusIcon = () => {
    switch (status) {
      case 'connected':
        return <Wifi className="h-4 w-4" />
      case 'connecting':
        return <AlertCircle className="h-4 w-4 animate-pulse" />
      case 'disconnected':
      case 'error':
        return <WifiOff className="h-4 w-4" />
      default:
        return <WifiOff className="h-4 w-4" />
    }
  }

  const getStatusVariant = () => {
    switch (status) {
      case 'connected':
        return 'success'
      case 'connecting':
        return 'secondary'
      case 'disconnected':
      case 'error':
        return 'destructive'
      default:
        return 'secondary'
    }
  }

  const getStatusText = () => {
    if (apiHealth === 'unhealthy') return 'API Offline'
    if (apiHealth === 'checking') return 'Checking...'
    
    switch (status) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return 'Connecting...'
      case 'disconnected':
        return 'Disconnected'
      case 'error':
        return 'Connection Error'
      default:
        return 'Unknown'
    }
  }

  return (
    <div className="flex items-center space-x-2">
      <Badge variant={getStatusVariant()} className="flex items-center space-x-1">
        {getStatusIcon()}
        <span className="text-xs">{getStatusText()}</span>
      </Badge>
    </div>
  )
}