import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { processingApi } from '@/services/api'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { 
  Settings as SettingsIcon, 
  Save, 
  RotateCcw,
  AlertCircle,
  CheckCircle,
  Info
} from 'lucide-react'

export default function Settings() {
  const queryClient = useQueryClient()
  const [hasChanges, setHasChanges] = useState(false)

  const { data: settings, isLoading } = useQuery(
    ['processing', 'settings'],
    () => processingApi.getSettings().then(res => res.data)
  )

  const [formData, setFormData] = useState({
    auto_process: true,
    quality_threshold: 7,
    max_concurrent: 2,
    pause_on_error: true,
  })

  // Update form data when settings are loaded
  useEffect(() => {
    if (settings) {
      setFormData({
        auto_process: settings.auto_process ?? true,
        quality_threshold: settings.quality_threshold ?? 7,
        max_concurrent: settings.max_concurrent ?? 2,
        pause_on_error: settings.pause_on_error ?? true,
      })
    }
  }, [settings])

  const updateMutation = useMutation(
    (newSettings: typeof formData) => processingApi.updateSettings(newSettings),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['processing', 'settings'])
        setHasChanges(false)
        toast.success('Settings saved successfully')
      },
      onError: () => toast.error('Failed to save settings')
    }
  )

  const handleInputChange = (field: keyof typeof formData, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    setHasChanges(true)
  }

  const handleSave = () => {
    updateMutation.mutate(formData)
  }

  const handleReset = () => {
    if (settings) {
      setFormData({
        auto_process: settings.auto_process,
        quality_threshold: settings.quality_threshold,
        max_concurrent: settings.max_concurrent,
        pause_on_error: settings.pause_on_error,
      })
      setHasChanges(false)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-48 bg-muted animate-pulse rounded-lg" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-muted-foreground">
            Configure photo processing behavior
          </p>
        </div>
        
        {hasChanges && (
          <div className="flex items-center space-x-2">
            <Button variant="outline" onClick={handleReset}>
              <RotateCcw className="h-4 w-4 mr-2" />
              Reset
            </Button>
            <Button onClick={handleSave} disabled={updateMutation.isLoading}>
              <Save className="h-4 w-4 mr-2" />
              Save Changes
            </Button>
          </div>
        )}
      </div>

      {/* Processing Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <SettingsIcon className="h-5 w-5 mr-2" />
            Processing Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Auto Process */}
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <label className="text-sm font-medium">Auto Process</label>
              <p className="text-xs text-muted-foreground">
                Automatically process new photos when uploaded
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={formData.auto_process}
                onChange={(e) => handleInputChange('auto_process', e.target.checked)}
                className="h-4 w-4"
              />
              <Badge variant={formData.auto_process ? "success" : "secondary"}>
                {formData.auto_process ? "Enabled" : "Disabled"}
              </Badge>
            </div>
          </div>

          {/* Quality Threshold */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Quality Threshold</label>
            <p className="text-xs text-muted-foreground">
              Minimum quality score (0-10) for automatic processing
            </p>
            <div className="flex items-center space-x-4">
              <Input
                type="number"
                min="0"
                max="10"
                step="0.1"
                value={formData.quality_threshold}
                onChange={(e) => handleInputChange('quality_threshold', parseFloat(e.target.value))}
                className="w-24"
              />
              <span className="text-sm text-muted-foreground">
                Current: {formData.quality_threshold}/10
              </span>
            </div>
          </div>

          {/* Max Concurrent */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Max Concurrent Processing</label>
            <p className="text-xs text-muted-foreground">
              Maximum number of photos to process simultaneously
            </p>
            <div className="flex items-center space-x-4">
              <Input
                type="number"
                min="1"
                max="10"
                value={formData.max_concurrent}
                onChange={(e) => handleInputChange('max_concurrent', parseInt(e.target.value))}
                className="w-24"
              />
              <span className="text-sm text-muted-foreground">
                photos at once
              </span>
            </div>
          </div>

          {/* Pause on Error */}
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <label className="text-sm font-medium">Pause on Error</label>
              <p className="text-xs text-muted-foreground">
                Pause processing when an error occurs
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={formData.pause_on_error}
                onChange={(e) => handleInputChange('pause_on_error', e.target.checked)}
                className="h-4 w-4"
              />
              <Badge variant={formData.pause_on_error ? "destructive" : "success"}>
                {formData.pause_on_error ? "Pause" : "Continue"}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Info className="h-5 w-5 mr-2" />
            System Information
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <span className="text-sm font-medium">API Version</span>
              <p className="text-sm text-muted-foreground">1.0.0</p>
            </div>
            
            <div className="space-y-2">
              <span className="text-sm font-medium">Backend Status</span>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm">Connected</span>
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-sm font-medium">WebSocket Status</span>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm">Connected</span>
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-sm font-medium">Last Updated</span>
              <p className="text-sm text-muted-foreground">
                {new Date().toLocaleString()}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Storage Settings */}
      <Card>
        <CardHeader>
          <CardTitle>Storage Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium">Storage Path</label>
              <Input 
                value="/app/data" 
                disabled 
                className="text-muted-foreground"
              />
              <p className="text-xs text-muted-foreground">
                Default storage location for processed photos
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Backup Location</label>
              <Input 
                value="/app/data/originals" 
                disabled 
                className="text-muted-foreground"
              />
              <p className="text-xs text-muted-foreground">
                Location where original photos are preserved
              </p>
            </div>
          </div>

          <div className="p-4 bg-muted rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertCircle className="h-4 w-4 text-yellow-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium">Storage Configuration</p>
                <p className="text-xs text-muted-foreground">
                  Storage paths are configured at the Docker container level and cannot be changed from the UI.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}