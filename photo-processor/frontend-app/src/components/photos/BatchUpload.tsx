import React, { useState, useRef } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { photosApi } from '@/services/api'
import { Button } from '@/components/ui/Button'
import { Progress } from '@/components/ui/Progress'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ScrollArea } from '@/components/ui/ScrollArea'
import { 
  Upload, 
  File as FileIcon, 
  X, 
  CheckCircle, 
  AlertCircle,
  Pause,
  Play,
  FolderOpen
} from 'lucide-react'
import { formatBytes } from '@/lib/utils'

interface FileUploadItem {
  file: File
  id: string
  progress: number
  status: 'pending' | 'uploading' | 'completed' | 'error' | 'paused'
  error?: string
}

interface BatchUploadProps {
  onUploadComplete?: () => void
  recipeId?: string
}

export default function BatchUpload({ onUploadComplete, recipeId }: BatchUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const queryClient = useQueryClient()
  
  const [files, setFiles] = useState<FileUploadItem[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [overallProgress, setOverallProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentUploadIndex, setCurrentUploadIndex] = useState(0)

  const createSessionMutation = useMutation(
    ({ expectedFiles, totalSize }: { expectedFiles: number, totalSize: number }) =>
      photosApi.createUploadSession(expectedFiles, totalSize, recipeId),
    {
      onSuccess: (response) => {
        console.log('Upload session created:', response.data)
        const sessionId = response.data.session_id
        setSessionId(sessionId)
        startUploading(sessionId)
      },
      onError: (error) => {
        console.error('Failed to create upload session:', error)
        toast.error('Failed to create upload session')
      }
    }
  )

  const uploadFileMutation = useMutation(
    ({ sessionId, file, onProgress }: { sessionId: string, file: File, onProgress: (progress: number) => void }) =>
      photosApi.uploadFileToSession(sessionId, file, recipeId, onProgress),
    {
      onSuccess: (response, variables) => {
        console.log('File uploaded successfully:', variables.file.name, response.data)
        const fileIndex = files.findIndex(f => f.file === variables.file)
        if (fileIndex !== -1) {
          updateFileStatus(fileIndex, 'completed', 100)
        }
        uploadNextFile(variables.sessionId)
      },
      onError: (error, variables) => {
        console.error('File upload failed:', variables.file.name, error)
        const fileIndex = files.findIndex(f => f.file === variables.file)
        if (fileIndex !== -1) {
          updateFileStatus(fileIndex, 'error', 0, 'Upload failed')
        }
        uploadNextFile(variables.sessionId)
      }
    }
  )

  const completeSessionMutation = useMutation(
    (sessionId: string) => photosApi.completeUploadSession(sessionId),
    {
      onSuccess: () => {
        toast.success(`Successfully uploaded ${files.length} files`)
        queryClient.invalidateQueries(['photos'])
        onUploadComplete?.()
        resetUpload()
      },
      onError: () => toast.error('Failed to complete upload session')
    }
  )

  const updateFileStatus = (index: number, status: FileUploadItem['status'], progress: number, error?: string) => {
    setFiles(prev => prev.map((file, i) => 
      i === index ? { ...file, status, progress, error } : file
    ))
  }

  const updateOverallProgress = () => {
    const completedFiles = files.filter(f => f.status === 'completed').length
    const totalFiles = files.length
    const progress = totalFiles > 0 ? Math.round((completedFiles / totalFiles) * 100) : 0
    setOverallProgress(progress)
  }

  const startUploading = async (sessionIdParam?: string) => {
    const sessionToUse = sessionIdParam || sessionId
    console.log('Starting upload with sessionId:', sessionToUse, 'files count:', files.length)
    if (!sessionToUse || files.length === 0) {
      console.log('Cannot start upload - missing sessionId or no files')
      return
    }
    
    setIsUploading(true)
    setCurrentUploadIndex(0)
    setSessionId(sessionToUse)
    uploadNextFile(sessionToUse)
  }

  const uploadNextFile = (sessionIdParam?: string) => {
    const sessionToUse = sessionIdParam || sessionId
    if (isPaused) {
      console.log('Upload paused, skipping next file')
      return
    }

    const nextFileIndex = files.findIndex(f => f.status === 'pending')
    console.log('Looking for next file to upload, found index:', nextFileIndex)
    
    if (nextFileIndex === -1) {
      // All files uploaded, complete session
      console.log('All files uploaded, completing session')
      if (sessionToUse) {
        completeSessionMutation.mutate(sessionToUse)
      }
      setIsUploading(false)
      return
    }

    setCurrentUploadIndex(nextFileIndex)
    updateFileStatus(nextFileIndex, 'uploading', 0)

    const file = files[nextFileIndex].file
    console.log('Starting upload for file:', file.name, 'with sessionId:', sessionToUse)
    
    uploadFileMutation.mutate({
      sessionId: sessionToUse!,
      file,
      onProgress: (progress) => {
        updateFileStatus(nextFileIndex, 'uploading', progress)
        updateOverallProgress()
      }
    })
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || [])
    
    if (selectedFiles.length === 0) return

    // Validate file types - allow all common image and RAW formats
    const allowedExtensions = [
      // Standard image formats
      'jpg', 'jpeg', 'png', 'tiff', 'tif', 'webp', 'heic', 'heif', 'bmp', 'gif',
      // RAW formats
      'arw', 'cr2', 'cr3', 'dng', 'nef', 'orf', 'raf', 'rw2', 'srw', 'x3f',
      '3fr', 'dcr', 'kdc', 'mrw', 'pef', 'ptx', 'r3d', 'rwl', 'iiq', 'fff',
      'mef', 'mos', 'nrw', 'raw', 'rwz', 'sr2', 'srf', 'ari', 'bay', 'crw',
      'erf', 'k25', 'kc2'
    ]
    
    const validFiles = selectedFiles.filter(file => {
      const extension = file.name.toLowerCase().split('.').pop()
      return extension && allowedExtensions.includes(extension)
    })

    if (validFiles.length !== selectedFiles.length) {
      toast.warning(`${selectedFiles.length - validFiles.length} files were skipped (unsupported format)`)
    }

    if (validFiles.length > 1000) {
      toast.error('Maximum 1000 files per upload session')
      return
    }

    const fileItems: FileUploadItem[] = validFiles.map((file, index) => ({
      file,
      id: `${Date.now()}-${index}`,
      progress: 0,
      status: 'pending'
    }))

    setFiles(fileItems)
    
    // Calculate total size
    const totalSize = validFiles.reduce((sum, file) => sum + file.size, 0)
    
    if (totalSize > 10 * 1024 * 1024 * 1024) { // 10GB limit
      toast.error('Total file size cannot exceed 10GB')
      return
    }

    // Create upload session
    console.log('Creating upload session for', validFiles.length, 'files, total size:', totalSize)
    createSessionMutation.mutate({
      expectedFiles: validFiles.length,
      totalSize
    })
  }

  const handlePauseResume = () => {
    setIsPaused(!isPaused)
    if (isPaused && sessionId) {
      uploadNextFile(sessionId)
    }
  }

  const removeFile = (index: number) => {
    if (isUploading) {
      toast.warning('Cannot remove files during upload')
      return
    }
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const resetUpload = () => {
    setFiles([])
    setSessionId(null)
    setOverallProgress(0)
    setIsUploading(false)
    setIsPaused(false)
    setCurrentUploadIndex(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getStatusIcon = (status: FileUploadItem['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      case 'uploading':
        return <Upload className="h-4 w-4 text-blue-500 animate-pulse" />
      default:
        return <FileIcon className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status: FileUploadItem['status']) => {
    switch (status) {
      case 'completed':
        return <Badge variant="success" className="text-xs">Complete</Badge>
      case 'error':
        return <Badge variant="destructive" className="text-xs">Error</Badge>
      case 'uploading':
        return <Badge variant="secondary" className="text-xs">Uploading</Badge>
      case 'paused':
        return <Badge variant="outline" className="text-xs">Paused</Badge>
      default:
        return <Badge variant="secondary" className="text-xs">Pending</Badge>
    }
  }

  const totalSize = files.reduce((sum, item) => sum + item.file.size, 0)
  const completedFiles = files.filter(f => f.status === 'completed').length
  const errorFiles = files.filter(f => f.status === 'error').length

  return (
    <div className="space-y-6">
      {/* Upload Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5" />
            Batch Photo Upload
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <Button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="flex items-center gap-2"
            >
              <Upload className="h-4 w-4" />
              Select Photos
            </Button>
            
            {files.length > 0 && !isUploading && (
              <Button
                onClick={resetUpload}
                variant="outline"
              >
                Clear All
              </Button>
            )}
            
            {isUploading && (
              <Button
                onClick={handlePauseResume}
                variant="outline"
                className="flex items-center gap-2"
              >
                {isPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                {isPaused ? 'Resume' : 'Pause'}
              </Button>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.heic,.heif,.arw,.cr2,.cr3,.dng,.nef,.orf,.raf,.rw2,.srw,.x3f,.3fr,.dcr,.kdc,.mrw,.pef,.ptx,.r3d,.rwl,.iiq,.fff,.mef,.mos,.nrw,.raw,.rwz,.sr2,.srf,.ari,.bay,.crw,.erf,.k25,.kc2,.tif,.tiff"
            onChange={handleFileSelect}
            className="hidden"
          />

          {files.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="font-medium">{files.length}</div>
                <div className="text-muted-foreground">Total Files</div>
              </div>
              <div>
                <div className="font-medium">{formatBytes(totalSize)}</div>
                <div className="text-muted-foreground">Total Size</div>
              </div>
              <div>
                <div className="font-medium text-green-600">{completedFiles}</div>
                <div className="text-muted-foreground">Completed</div>
              </div>
              <div>
                <div className="font-medium text-red-600">{errorFiles}</div>
                <div className="text-muted-foreground">Errors</div>
              </div>
            </div>
          )}

          {/* Overall Progress */}
          {isUploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Overall Progress</span>
                <span>{overallProgress}%</span>
              </div>
              <Progress value={overallProgress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* File List */}
      {files.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Files ({files.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px]">
              <div className="space-y-2">
                {files.map((item, index) => (
                  <div
                    key={item.id}
                    className={`flex items-center justify-between p-3 rounded-lg border ${
                      index === currentUploadIndex && isUploading ? 'border-blue-200 bg-blue-50' : ''
                    }`}
                  >
                    <div className="flex items-center space-x-3 flex-1">
                      {getStatusIcon(item.status)}
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">
                          {item.file.name}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {formatBytes(item.file.size)}
                        </div>
                        {item.status === 'uploading' && (
                          <Progress value={item.progress} className="h-1 mt-1" />
                        )}
                        {item.error && (
                          <div className="text-xs text-red-500 mt-1">{item.error}</div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {getStatusBadge(item.status)}
                      {!isUploading && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => removeFile(index)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  )
}