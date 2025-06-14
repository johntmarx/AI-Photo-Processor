import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { photosApi } from '@/services/api'
import { Photo } from '@/types/api'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { formatBytes, formatRelativeTime } from '@/lib/utils'
import { 
  X, 
  Download, 
  RotateCcw, 
  Trash2,
  Eye,
  EyeOff,
  Info,
  Activity,
  ZoomIn,
  ZoomOut,
  RotateCw
} from 'lucide-react'

interface PhotoDialogProps {
  photo: Photo | null
  isOpen: boolean
  onClose: () => void
  onReprocess?: (photoId: string) => void
  onDelete?: (photoId: string) => void
}

export default function PhotoDialog({ 
  photo, 
  isOpen, 
  onClose, 
  onReprocess, 
  onDelete 
}: PhotoDialogProps) {
  const [showComparison, setShowComparison] = useState(false)
  const [activeTab, setActiveTab] = useState<'details' | 'ai' | 'history'>('details')
  const [zoom, setZoom] = useState(1)
  const [rotation, setRotation] = useState(0)

  const { data: photoDetail } = useQuery(
    ['photos', photo?.id],
    () => photo ? photosApi.get(photo.id).then(res => res.data) : null,
    { enabled: !!photo }
  )

  const { data: comparison } = useQuery(
    ['photos', photo?.id, 'comparison'],
    () => photo ? photosApi.getComparison(photo.id).then(res => res.data) : null,
    { enabled: !!photo && showComparison }
  )

  console.log('PhotoDialog render:', { isOpen, photo })
  
  if (!isOpen || !photo) {
    console.log('PhotoDialog not rendering - isOpen:', isOpen, 'photo:', photo)
    return null
  }

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3))
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5))
  const handleRotate = () => setRotation(prev => (prev + 90) % 360)

  const getImageUrl = () => {
    // Use the API endpoint for images
    return `/api/photos/${photo.id}/preview`
  }

  const handleDownload = () => {
    const downloadUrl = `/api/photos/${photo.id}/download`
    
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = photo.filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const imageUrl = getImageUrl()

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-background rounded-lg shadow-lg max-w-6xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div>
            <h2 className="text-xl font-semibold">{photo.filename}</h2>
            <p className="text-sm text-muted-foreground">
              {formatRelativeTime(photo.createdAt)}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowComparison(!showComparison)}
            >
              {showComparison ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              {showComparison ? 'Hide' : 'Show'} Comparison
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Image Preview */}
          <div className="flex-1 p-6">
            <div className="aspect-video bg-black rounded-lg flex items-center justify-center mb-4 overflow-hidden">
              {showComparison && comparison ? (
                <div className="grid grid-cols-2 gap-4 w-full h-full">
                  <div className="bg-gray-100 rounded flex items-center justify-center">
                    <div className="text-center">
                      <Eye className="h-12 w-12 mx-auto text-gray-400 mb-2" />
                      <p className="text-sm text-gray-600">Original</p>
                    </div>
                  </div>
                  <div className="bg-gray-100 rounded flex items-center justify-center">
                    <div className="text-center">
                      <Eye className="h-12 w-12 mx-auto text-gray-400 mb-2" />
                      <p className="text-sm text-gray-600">Processed</p>
                    </div>
                  </div>
                </div>
              ) : imageUrl ? (
                <div 
                  className="w-full h-full flex items-center justify-center overflow-auto"
                  style={{ cursor: zoom > 1 ? 'move' : 'default' }}
                >
                  <img
                    src={imageUrl}
                    alt={photo.filename}
                    className="max-w-none transition-transform duration-200"
                    style={{
                      transform: `scale(${zoom}) rotate(${rotation}deg)`,
                      maxWidth: zoom === 1 ? '100%' : 'none',
                      maxHeight: zoom === 1 ? '100%' : 'none',
                    }}
                    draggable={false}
                  />
                </div>
              ) : (
                <div className="text-center">
                  <Eye className="h-16 w-16 mx-auto text-gray-400 mb-2" />
                  <p className="text-gray-600">No image available</p>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex justify-between">
              <div className="flex space-x-2">
                <Button size="sm" variant="outline" onClick={handleZoomOut} disabled={zoom <= 0.5}>
                  <ZoomOut className="h-4 w-4" />
                </Button>
                <span className="text-sm text-muted-foreground min-w-[60px] text-center flex items-center">
                  {Math.round(zoom * 100)}%
                </span>
                <Button size="sm" variant="outline" onClick={handleZoomIn} disabled={zoom >= 3}>
                  <ZoomIn className="h-4 w-4" />
                </Button>
                <Button size="sm" variant="outline" onClick={handleRotate}>
                  <RotateCw className="h-4 w-4" />
                </Button>
              </div>
              
              <div className="flex space-x-2">
                <Button variant="outline" size="sm" onClick={handleDownload}>
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
                {photo.status === 'failed' && onReprocess && (
                  <Button variant="outline" size="sm" onClick={() => onReprocess(photo.id)}>
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reprocess
                  </Button>
                )}
                {onDelete && (
                  <Button variant="destructive" size="sm" onClick={() => onDelete(photo.id)}>
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </Button>
                )}
              </div>
            </div>
          </div>

          {/* Details Panel */}
          <div className="w-96 border-l overflow-hidden flex flex-col">
            {/* Tabs */}
            <div className="flex border-b">
              <button
                className={`flex-1 px-4 py-2 text-sm font-medium ${
                  activeTab === 'details' 
                    ? 'border-b-2 border-primary text-primary' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setActiveTab('details')}
              >
                <Info className="h-4 w-4 inline mr-1" />
                Details
              </button>
              <button
                className={`flex-1 px-4 py-2 text-sm font-medium ${
                  activeTab === 'ai' 
                    ? 'border-b-2 border-primary text-primary' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setActiveTab('ai')}
              >
                AI Analysis
              </button>
              <button
                className={`flex-1 px-4 py-2 text-sm font-medium ${
                  activeTab === 'history' 
                    ? 'border-b-2 border-primary text-primary' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setActiveTab('history')}
              >
                <Activity className="h-4 w-4 inline mr-1" />
                History
              </button>
            </div>

            <ScrollArea className="flex-1 p-4">
              {activeTab === 'details' && (
                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">File Information</CardTitle>
                    </CardHeader>
                    <CardContent className="text-sm space-y-2">
                      <div className="flex justify-between">
                        <span>Status:</span>
                        <Badge variant={photo.status === 'completed' ? 'success' : 'secondary'}>
                          {photo.status}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Size:</span>
                        <span>{formatBytes(photo.fileSize)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>ID:</span>
                        <code className="text-xs">{photo.id.slice(0, 8)}...</code>
                      </div>
                    </CardContent>
                  </Card>

                  {photoDetail?.metadata && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Metadata</CardTitle>
                      </CardHeader>
                      <CardContent className="text-sm">
                        <pre className="text-xs whitespace-pre-wrap">
                          {JSON.stringify(photoDetail.metadata, null, 2)}
                        </pre>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}

              {activeTab === 'ai' && (
                <div className="space-y-4">
                  {photoDetail?.aiAnalysis && typeof photoDetail.aiAnalysis === 'object' && photoDetail.aiAnalysis.status === 'completed' && (photoDetail.aiAnalysis.aestheticScore || photoDetail.aiAnalysis.technicalScore) ? (
                    <>
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-sm">Quality Analysis</CardTitle>
                        </CardHeader>
                        <CardContent className="text-sm space-y-2">
                          <div className="flex justify-between">
                            <span>Technical Quality:</span>
                            <span>{photoDetail.aiAnalysis.technicalScore ? photoDetail.aiAnalysis.technicalScore.toFixed(1) : 'N/A'}/10</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Aesthetic Appeal:</span>
                            <span>{photoDetail.aiAnalysis.aestheticScore ? photoDetail.aiAnalysis.aestheticScore.toFixed(1) : 'N/A'}/10</span>
                          </div>
                          {(photoDetail.aiAnalysis.combined_score || photoDetail.aiAnalysis.combinedScore) && (
                            <div className="flex justify-between">
                              <span>Overall Score:</span>
                              <span className="font-medium">{(photoDetail.aiAnalysis.combined_score || photoDetail.aiAnalysis.combinedScore)?.toFixed(1)}/10</span>
                            </div>
                          )}
                          <div className="flex justify-between">
                            <span>Quality Level:</span>
                            <span className="capitalize">{photoDetail.aiAnalysis.qualityLevel || 'N/A'}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Aesthetics Level:</span>
                            <span className="capitalize">{photoDetail.aiAnalysis.aestheticsLevel || 'N/A'}</span>
                          </div>
                        </CardContent>
                      </Card>

                      {photoDetail.aiAnalysis.subjects && Array.isArray(photoDetail.aiAnalysis.subjects) && photoDetail.aiAnalysis.subjects.length > 0 && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Detected Subjects</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-1">
                          {photoDetail.aiAnalysis.subjects.map((subject, index) => (
                            <Badge key={index} variant="outline" className="text-xs">
                              {subject}
                            </Badge>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                      {photoDetail.aiAnalysis.recommendations && Array.isArray(photoDetail.aiAnalysis.recommendations) && photoDetail.aiAnalysis.recommendations.length > 0 && (
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-sm">Recommendations</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <ul className="text-sm space-y-1">
                              {photoDetail.aiAnalysis.recommendations.map((rec, index) => (
                                <li key={index} className="text-muted-foreground">
                                  • {rec}
                                </li>
                              ))}
                            </ul>
                          </CardContent>
                        </Card>
                      )}
                    </>
                  ) : (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">AI Analysis</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-muted-foreground">
                          {photoDetail?.aiAnalysis?.status === 'pending' 
                            ? 'AI analysis is in progress...'
                            : photoDetail?.aiAnalysis?.status === 'processing'
                            ? 'AI analysis is currently running...'
                            : photoDetail?.aiAnalysis?.status === 'failed'
                            ? 'AI analysis failed. Try reprocessing the photo.'
                            : 'AI analysis not available for this photo.'}
                        </p>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}

              {activeTab === 'history' && (
                <div className="space-y-4">
                  {photoDetail?.processingHistory?.length ? (
                    photoDetail.processingHistory.map((record) => (
                      <Card key={record.id}>
                        <CardContent className="p-4">
                          <div className="flex justify-between items-start mb-2">
                            <span className="font-medium text-sm">{record.action}</span>
                            <Badge variant={record.result === 'success' ? 'success' : 'destructive'}>
                              {record.result}
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground mb-1">
                            {formatRelativeTime(new Date(record.timestamp))}
                          </p>
                          <p className="text-xs">{record.details}</p>
                        </CardContent>
                      </Card>
                    ))
                  ) : (
                    <div className="text-center text-muted-foreground py-8">
                      <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No processing history</p>
                    </div>
                  )}
                </div>
              )}
            </ScrollArea>
          </div>
        </div>
      </div>
    </div>
  )
}