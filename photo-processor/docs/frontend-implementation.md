# Frontend Implementation Plan

## Overview

The frontend will provide a comprehensive dashboard for monitoring, controlling, and manually intervening in the photo processing pipeline. Built with React and modern web technologies, it will offer real-time updates and an intuitive interface.

## Technology Stack

- **Framework**: React 18+ with TypeScript
- **State Management**: Zustand (lightweight, TypeScript-friendly)
- **UI Components**: Radix UI + Tailwind CSS
- **Real-time Updates**: WebSocket (Socket.io)
- **Data Fetching**: TanStack Query (React Query)
- **Image Handling**: Sharp.js (client-side via WASM)
- **Charts/Visualization**: Recharts
- **Development**: Vite, ESLint, Prettier

## Architecture

```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard/
│   │   │   ├── StatsOverview.tsx
│   │   │   ├── ProcessingQueue.tsx
│   │   │   └── RecentActivity.tsx
│   │   ├── PhotoViewer/
│   │   │   ├── ImageCanvas.tsx
│   │   │   ├── ComparisonView.tsx
│   │   │   └── MetadataPanel.tsx
│   │   ├── Processing/
│   │   │   ├── QueueManager.tsx
│   │   │   ├── BatchProcessor.tsx
│   │   │   └── RecipeEditor.tsx
│   │   ├── AI/
│   │   │   ├── DetectionOverlay.tsx
│   │   │   ├── QualityScores.tsx
│   │   │   └── SuggestionsList.tsx
│   │   └── Common/
│   │       ├── Layout.tsx
│   │       ├── Navigation.tsx
│   │       └── NotificationCenter.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── usePhotoProcessor.ts
│   │   └── useKeyboardShortcuts.ts
│   ├── stores/
│   │   ├── processingStore.ts
│   │   ├── photoStore.ts
│   │   └── settingsStore.ts
│   ├── services/
│   │   ├── api.ts
│   │   ├── websocket.ts
│   │   └── imageProcessor.ts
│   ├── types/
│   │   ├── photo.types.ts
│   │   ├── processing.types.ts
│   │   └── ai.types.ts
│   └── utils/
│       ├── imageHelpers.ts
│       ├── recipeHelpers.ts
│       └── formatters.ts
```

## Core Components

### 1. Dashboard Component

```tsx
import React from 'react';
import { useProcessingStore } from '@/stores/processingStore';
import { Card, Grid, Flex } from '@/components/ui';
import { StatsOverview } from './StatsOverview';
import { ProcessingQueue } from './ProcessingQueue';
import { RecentActivity } from './RecentActivity';
import { SystemHealth } from './SystemHealth';

export const Dashboard: React.FC = () => {
  const { stats, queue, recentFiles } = useProcessingStore();
  
  return (
    <div className="p-6 space-y-6">
      {/* Stats Overview */}
      <StatsOverview stats={stats} />
      
      {/* Main Grid */}
      <Grid cols={12} gap={6}>
        {/* Processing Queue */}
        <Card className="col-span-8">
          <ProcessingQueue queue={queue} />
        </Card>
        
        {/* System Health */}
        <Card className="col-span-4">
          <SystemHealth />
        </Card>
      </Grid>
      
      {/* Recent Activity */}
      <Card>
        <RecentActivity files={recentFiles} />
      </Card>
    </div>
  );
};

// Stats Overview Component
const StatsOverview: React.FC<{ stats: ProcessingStats }> = ({ stats }) => {
  const statCards = [
    {
      label: 'Photos Processed',
      value: stats.totalProcessed,
      change: stats.processedToday,
      icon: 'photo'
    },
    {
      label: 'Queue Size',
      value: stats.queueSize,
      change: stats.queueChange,
      icon: 'queue'
    },
    {
      label: 'Average Quality',
      value: stats.avgQuality.toFixed(1),
      suffix: '/10',
      icon: 'star'
    },
    {
      label: 'Processing Rate',
      value: stats.processingRate,
      suffix: '/min',
      icon: 'speed'
    }
  ];
  
  return (
    <Grid cols={4} gap={4}>
      {statCards.map((stat) => (
        <StatCard key={stat.label} {...stat} />
      ))}
    </Grid>
  );
};
```

### 2. Photo Viewer with AI Overlays

```tsx
import React, { useState, useRef, useEffect } from 'react';
import { Canvas, Layer } from '@/components/Canvas';
import { usePhotoStore } from '@/stores/photoStore';
import { Detection, Segmentation, CropSuggestion } from '@/types';

interface PhotoViewerProps {
  photoId: string;
  mode: 'view' | 'edit' | 'compare';
}

export const PhotoViewer: React.FC<PhotoViewerProps> = ({ photoId, mode }) => {
  const { photo, aiResults, processingRecipe } = usePhotoStore(photoId);
  const [showOverlays, setShowOverlays] = useState({
    detections: true,
    segments: false,
    cropGuides: true,
    quality: true
  });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  
  return (
    <div className="relative h-full bg-gray-900">
      {/* Toolbar */}
      <div className="absolute top-4 left-4 right-4 z-10">
        <Toolbar>
          <ToggleGroup>
            <Toggle
              pressed={showOverlays.detections}
              onPressedChange={(v) => 
                setShowOverlays(prev => ({ ...prev, detections: v }))
              }
            >
              <IconBox /> Objects
            </Toggle>
            <Toggle
              pressed={showOverlays.segments}
              onPressedChange={(v) => 
                setShowOverlays(prev => ({ ...prev, segments: v }))
              }
            >
              <IconSegment /> Segments
            </Toggle>
            <Toggle
              pressed={showOverlays.cropGuides}
              onPressedChange={(v) => 
                setShowOverlays(prev => ({ ...prev, cropGuides: v }))
              }
            >
              <IconCrop /> Crop
            </Toggle>
          </ToggleGroup>
          
          <Separator orientation="vertical" />
          
          <ZoomControls zoom={zoom} onZoomChange={setZoom} />
        </Toolbar>
      </div>
      
      {/* Canvas */}
      <Canvas
        ref={canvasRef}
        className="w-full h-full"
        zoom={zoom}
        pan={pan}
        onPanChange={setPan}
      >
        {/* Base Image Layer */}
        <Layer name="image">
          <Image src={photo.url} onLoad={handleImageLoad} />
        </Layer>
        
        {/* Detection Overlay */}
        {showOverlays.detections && (
          <Layer name="detections" opacity={0.8}>
            <DetectionOverlay detections={aiResults.detections} />
          </Layer>
        )}
        
        {/* Segmentation Masks */}
        {showOverlays.segments && (
          <Layer name="segments" opacity={0.5}>
            <SegmentationOverlay segments={aiResults.segments} />
          </Layer>
        )}
        
        {/* Crop Guides */}
        {showOverlays.cropGuides && aiResults.cropSuggestion && (
          <Layer name="crop" opacity={1}>
            <CropGuideOverlay
              suggestion={aiResults.cropSuggestion}
              editable={mode === 'edit'}
              onChange={handleCropChange}
            />
          </Layer>
        )}
      </Canvas>
      
      {/* Side Panel */}
      <SidePanel position="right" defaultOpen>
        <Tabs defaultValue="ai">
          <TabsList>
            <TabsTrigger value="ai">AI Analysis</TabsTrigger>
            <TabsTrigger value="metadata">Metadata</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>
          
          <TabsContent value="ai">
            <AIAnalysisPanel results={aiResults} />
          </TabsContent>
          
          <TabsContent value="metadata">
            <MetadataPanel photo={photo} />
          </TabsContent>
          
          <TabsContent value="history">
            <ProcessingHistory photoId={photoId} />
          </TabsContent>
        </Tabs>
      </SidePanel>
    </div>
  );
};

// Detection Overlay Component
const DetectionOverlay: React.FC<{ detections: Detection[] }> = ({ 
  detections 
}) => {
  return (
    <>
      {detections.map((detection, idx) => (
        <DetectionBox
          key={idx}
          bounds={detection.bbox}
          label={detection.class_name}
          confidence={detection.confidence}
          color={getColorForClass(detection.class_name)}
        />
      ))}
    </>
  );
};

// Crop Guide Overlay
const CropGuideOverlay: React.FC<{
  suggestion: CropSuggestion;
  editable: boolean;
  onChange: (crop: CropSuggestion) => void;
}> = ({ suggestion, editable, onChange }) => {
  const [handles, setHandles] = useState(suggestion.bounds);
  
  return (
    <CropGuide
      bounds={handles}
      showRuleOfThirds
      showGoldenRatio={suggestion.useGoldenRatio}
      editable={editable}
      onBoundsChange={(bounds) => {
        setHandles(bounds);
        onChange({ ...suggestion, bounds });
      }}
    />
  );
};
```

### 3. Processing Queue Manager

```tsx
import React, { useState } from 'react';
import { useProcessingQueue } from '@/hooks/useProcessingQueue';
import { QueueItem, ProcessingMode } from '@/types';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';

export const QueueManager: React.FC = () => {
  const {
    queue,
    updatePriority,
    pauseItem,
    resumeItem,
    removeItem,
    batchUpdate
  } = useProcessingQueue();
  
  const [selectedItems, setSelectedItems] = useState<Set<string>>(new Set());
  const [filterMode, setFilterMode] = useState<ProcessingMode | 'all'>('all');
  
  const filteredQueue = queue.filter(
    item => filterMode === 'all' || item.mode === filterMode
  );
  
  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return;
    
    const items = Array.from(filteredQueue);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    
    // Update priorities based on new order
    items.forEach((item, index) => {
      updatePriority(item.id, items.length - index);
    });
  };
  
  return (
    <div className="space-y-4">
      {/* Queue Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Select value={filterMode} onValueChange={setFilterMode}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Items</SelectItem>
              <SelectItem value="auto">Auto Process</SelectItem>
              <SelectItem value="semi-auto">Semi-Auto</SelectItem>
              <SelectItem value="manual">Manual</SelectItem>
            </SelectContent>
          </Select>
          
          <div className="text-sm text-muted-foreground">
            {filteredQueue.length} items • 
            {selectedItems.size} selected
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {selectedItems.size > 0 && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => batchUpdate(selectedItems, { paused: true })}
              >
                <IconPause className="mr-1" />
                Pause Selected
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => batchUpdate(selectedItems, { mode: 'manual' })}
              >
                <IconEdit className="mr-1" />
                Set Manual
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => {
                  batchUpdate(selectedItems, { remove: true });
                  setSelectedItems(new Set());
                }}
              >
                <IconTrash className="mr-1" />
                Remove
              </Button>
            </>
          )}
        </div>
      </div>
      
      {/* Queue List */}
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="queue">
          {(provided) => (
            <div
              {...provided.droppableProps}
              ref={provided.innerRef}
              className="space-y-2"
            >
              {filteredQueue.map((item, index) => (
                <Draggable
                  key={item.id}
                  draggableId={item.id}
                  index={index}
                >
                  {(provided, snapshot) => (
                    <QueueItemCard
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      {...provided.dragHandleProps}
                      item={item}
                      selected={selectedItems.has(item.id)}
                      onSelect={(selected) => {
                        const newSelection = new Set(selectedItems);
                        if (selected) {
                          newSelection.add(item.id);
                        } else {
                          newSelection.delete(item.id);
                        }
                        setSelectedItems(newSelection);
                      }}
                      isDragging={snapshot.isDragging}
                    />
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
    </div>
  );
};

// Queue Item Card
const QueueItemCard = React.forwardRef<HTMLDivElement, {
  item: QueueItem;
  selected: boolean;
  onSelect: (selected: boolean) => void;
  isDragging: boolean;
}>(({ item, selected, onSelect, isDragging, ...props }, ref) => {
  const progress = useQueueItemProgress(item.id);
  
  return (
    <Card
      ref={ref}
      className={cn(
        "p-4 transition-all",
        isDragging && "opacity-50",
        selected && "ring-2 ring-primary"
      )}
      {...props}
    >
      <div className="flex items-center gap-4">
        {/* Selection Checkbox */}
        <Checkbox
          checked={selected}
          onCheckedChange={onSelect}
          onClick={(e) => e.stopPropagation()}
        />
        
        {/* Drag Handle */}
        <IconGripVertical className="text-muted-foreground cursor-move" />
        
        {/* Thumbnail */}
        <div className="relative w-20 h-20 rounded overflow-hidden bg-muted">
          <img
            src={item.thumbnailUrl}
            alt={item.filename}
            className="w-full h-full object-cover"
          />
          {item.status === 'processing' && (
            <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
              <Spinner className="text-white" />
            </div>
          )}
        </div>
        
        {/* Item Details */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="font-medium truncate">{item.filename}</h4>
            <Badge variant={getModeVariant(item.mode)}>
              {item.mode}
            </Badge>
          </div>
          
          <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
            <span>{formatFileSize(item.fileSize)}</span>
            <span>•</span>
            <span>Priority: {item.priority}</span>
            {item.aiScore && (
              <>
                <span>•</span>
                <span>AI Score: {item.aiScore.toFixed(1)}/10</span>
              </>
            )}
          </div>
          
          {/* Progress Bar */}
          {item.status === 'processing' && progress && (
            <div className="mt-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span>{progress.stage}</span>
                <span>{Math.round(progress.percent)}%</span>
              </div>
              <Progress value={progress.percent} className="h-1" />
            </div>
          )}
        </div>
        
        {/* Actions */}
        <div className="flex items-center gap-2">
          {item.status === 'queued' && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => pauseItem(item.id)}
            >
              <IconPause />
            </Button>
          )}
          
          {item.status === 'paused' && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => resumeItem(item.id)}
            >
              <IconPlay />
            </Button>
          )}
          
          <Button
            variant="ghost"
            size="icon"
            onClick={() => removeItem(item.id)}
          >
            <IconX />
          </Button>
        </div>
      </div>
    </Card>
  );
});
```

### 4. Recipe Editor

```tsx
import React, { useState, useEffect } from 'react';
import { ProcessingRecipe, ProcessingOperation } from '@/types';
import { useRecipeStore } from '@/stores/recipeStore';
import { OperationEditor } from './OperationEditor';

interface RecipeEditorProps {
  recipeId?: string;
  initialOperations?: ProcessingOperation[];
  onSave: (recipe: ProcessingRecipe) => void;
}

export const RecipeEditor: React.FC<RecipeEditorProps> = ({
  recipeId,
  initialOperations = [],
  onSave
}) => {
  const { loadRecipe, saveRecipe } = useRecipeStore();
  const [recipe, setRecipe] = useState<ProcessingRecipe>(() => {
    if (recipeId) {
      return loadRecipe(recipeId);
    }
    return {
      id: generateId(),
      name: 'Untitled Recipe',
      operations: initialOperations,
      version: 1,
      created: new Date()
    };
  });
  
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Real-time preview generation
  useEffect(() => {
    const generatePreview = async () => {
      if (!recipe.operations.length) return;
      
      setIsProcessing(true);
      try {
        const preview = await processPreview(recipe.operations);
        setPreviewUrl(preview.url);
      } finally {
        setIsProcessing(false);
      }
    };
    
    const debounced = debounce(generatePreview, 500);
    debounced();
    
    return () => debounced.cancel();
  }, [recipe.operations]);
  
  const addOperation = (type: string) => {
    const newOp = createDefaultOperation(type);
    setRecipe(prev => ({
      ...prev,
      operations: [...prev.operations, newOp]
    }));
  };
  
  const updateOperation = (index: number, updates: Partial<ProcessingOperation>) => {
    setRecipe(prev => ({
      ...prev,
      operations: prev.operations.map((op, i) => 
        i === index ? { ...op, ...updates } : op
      )
    }));
  };
  
  const removeOperation = (index: number) => {
    setRecipe(prev => ({
      ...prev,
      operations: prev.operations.filter((_, i) => i !== index)
    }));
  };
  
  const reorderOperations = (fromIndex: number, toIndex: number) => {
    setRecipe(prev => {
      const ops = [...prev.operations];
      const [removed] = ops.splice(fromIndex, 1);
      ops.splice(toIndex, 0, removed);
      return { ...prev, operations: ops };
    });
  };
  
  return (
    <div className="flex h-full">
      {/* Editor Panel */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="space-y-6">
          {/* Recipe Header */}
          <div>
            <Input
              value={recipe.name}
              onChange={(e) => setRecipe(prev => ({ 
                ...prev, 
                name: e.target.value 
              }))}
              className="text-2xl font-semibold"
              placeholder="Recipe Name"
            />
            <p className="text-sm text-muted-foreground mt-2">
              Version {recipe.version} • 
              Created {formatDate(recipe.created)}
            </p>
          </div>
          
          {/* Operations List */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">Operations</h3>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm">
                    <IconPlus className="mr-1" />
                    Add Operation
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onClick={() => addOperation('crop')}>
                    <IconCrop className="mr-2" />
                    Crop
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => addOperation('rotate')}>
                    <IconRotate className="mr-2" />
                    Rotate
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => addOperation('enhance')}>
                    <IconWand className="mr-2" />
                    Enhance
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => addOperation('filter')}>
                    <IconFilter className="mr-2" />
                    Filter
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            
            <DndContext
              sensors={[useSensor(PointerSensor)]}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <SortableContext
                items={recipe.operations.map((_, i) => i)}
                strategy={verticalListSortingStrategy}
              >
                {recipe.operations.map((operation, index) => (
                  <SortableOperationCard
                    key={operation.id}
                    operation={operation}
                    index={index}
                    onUpdate={(updates) => updateOperation(index, updates)}
                    onRemove={() => removeOperation(index)}
                  />
                ))}
              </SortableContext>
            </DndContext>
            
            {recipe.operations.length === 0 && (
              <Card className="p-8 text-center text-muted-foreground">
                <IconWand className="mx-auto mb-3 h-8 w-8" />
                <p>No operations yet. Add one to get started!</p>
              </Card>
            )}
          </div>
          
          {/* Save Actions */}
          <div className="flex gap-3">
            <Button
              onClick={() => {
                saveRecipe(recipe);
                onSave(recipe);
              }}
              className="flex-1"
            >
              Save Recipe
            </Button>
            <Button
              variant="outline"
              onClick={() => exportRecipe(recipe)}
            >
              <IconDownload className="mr-1" />
              Export
            </Button>
          </div>
        </div>
      </div>
      
      {/* Preview Panel */}
      <div className="w-1/2 border-l bg-muted/10">
        <div className="sticky top-0 p-4 border-b bg-background">
          <h3 className="font-medium">Preview</h3>
        </div>
        
        <div className="p-4">
          {isProcessing ? (
            <div className="aspect-video bg-muted rounded flex items-center justify-center">
              <Spinner />
              <span className="ml-2">Generating preview...</span>
            </div>
          ) : previewUrl ? (
            <img
              src={previewUrl}
              alt="Recipe preview"
              className="w-full rounded"
            />
          ) : (
            <div className="aspect-video bg-muted rounded flex items-center justify-center text-muted-foreground">
              <IconImage className="h-8 w-8" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
```

### 5. WebSocket Integration

```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { useProcessingStore } from '@/stores/processingStore';
import { ProcessingEvent } from '@/types';

export const useWebSocket = () => {
  const socketRef = useRef<Socket | null>(null);
  const { updateQueue, updateStats, addNotification } = useProcessingStore();
  
  useEffect(() => {
    // Connect to WebSocket server
    socketRef.current = io(import.meta.env.VITE_WS_URL, {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5
    });
    
    const socket = socketRef.current;
    
    // Event handlers
    socket.on('connect', () => {
      console.log('Connected to processing server');
      addNotification({
        type: 'success',
        message: 'Connected to processing server'
      });
    });
    
    socket.on('disconnect', () => {
      addNotification({
        type: 'warning',
        message: 'Disconnected from processing server'
      });
    });
    
    socket.on('processing:update', (event: ProcessingEvent) => {
      switch (event.type) {
        case 'queue:item:added':
          updateQueue(queue => [...queue, event.data]);
          break;
          
        case 'queue:item:updated':
          updateQueue(queue => 
            queue.map(item => 
              item.id === event.data.id ? event.data : item
            )
          );
          break;
          
        case 'queue:item:removed':
          updateQueue(queue => 
            queue.filter(item => item.id !== event.data.id)
          );
          break;
          
        case 'stats:updated':
          updateStats(event.data);
          break;
          
        case 'processing:progress':
          // Update specific item progress
          updateQueue(queue =>
            queue.map(item =>
              item.id === event.data.id
                ? { ...item, progress: event.data.progress }
                : item
            )
          );
          break;
      }
    });
    
    socket.on('ai:analysis:complete', (data) => {
      // Handle AI analysis results
      usePhotoStore.getState().updateAIResults(data.photoId, data.results);
    });
    
    socket.on('notification', (notification) => {
      addNotification(notification);
    });
    
    // Cleanup
    return () => {
      socket.disconnect();
    };
  }, []);
  
  const emit = (event: string, data?: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    }
  };
  
  return {
    socket: socketRef.current,
    emit,
    connected: socketRef.current?.connected ?? false
  };
};

// services/websocket.ts
export class WebSocketService {
  private socket: Socket;
  private listeners: Map<string, Set<Function>> = new Map();
  
  constructor(url: string) {
    this.socket = io(url);
    this.setupBaseListeners();
  }
  
  private setupBaseListeners() {
    this.socket.on('connect', () => {
      this.emit('connected');
    });
    
    this.socket.on('disconnect', () => {
      this.emit('disconnected');
    });
  }
  
  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
      
      // Set up socket listener
      this.socket.on(event, (...args) => {
        this.listeners.get(event)?.forEach(cb => cb(...args));
      });
    }
    
    this.listeners.get(event)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.listeners.get(event)?.delete(callback);
    };
  }
  
  emit(event: string, data?: any) {
    this.socket.emit(event, data);
  }
  
  // Typed methods for common operations
  requestProcessing(photoId: string, options: ProcessingOptions) {
    this.emit('processing:request', { photoId, options });
  }
  
  updateProcessingMode(photoId: string, mode: ProcessingMode) {
    this.emit('processing:updateMode', { photoId, mode });
  }
  
  approveAISuggestions(photoId: string, approved: string[]) {
    this.emit('ai:approve', { photoId, approved });
  }
}
```

### 6. State Management

```typescript
// stores/processingStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

interface ProcessingState {
  queue: QueueItem[];
  stats: ProcessingStats;
  notifications: Notification[];
  settings: ProcessingSettings;
  
  // Actions
  updateQueue: (updater: (queue: QueueItem[]) => QueueItem[]) => void;
  updateStats: (stats: Partial<ProcessingStats>) => void;
  addNotification: (notification: Notification) => void;
  removeNotification: (id: string) => void;
  updateSettings: (settings: Partial<ProcessingSettings>) => void;
  
  // Complex actions
  batchUpdateQueue: (ids: Set<string>, updates: Partial<QueueItem>) => void;
  reorderQueue: (fromIndex: number, toIndex: number) => void;
}

export const useProcessingStore = create<ProcessingState>()(
  devtools(
    persist(
      immer((set) => ({
        queue: [],
        stats: {
          totalProcessed: 0,
          processedToday: 0,
          queueSize: 0,
          avgQuality: 0,
          processingRate: 0
        },
        notifications: [],
        settings: {
          autoProcess: true,
          defaultMode: 'auto',
          qualityThreshold: 6.0,
          maxConcurrent: 3
        },
        
        updateQueue: (updater) => set((state) => {
          state.queue = updater(state.queue);
        }),
        
        updateStats: (stats) => set((state) => {
          Object.assign(state.stats, stats);
        }),
        
        addNotification: (notification) => set((state) => {
          state.notifications.push({
            ...notification,
            id: generateId(),
            timestamp: new Date()
          });
        }),
        
        removeNotification: (id) => set((state) => {
          state.notifications = state.notifications.filter(n => n.id !== id);
        }),
        
        updateSettings: (settings) => set((state) => {
          Object.assign(state.settings, settings);
        }),
        
        batchUpdateQueue: (ids, updates) => set((state) => {
          state.queue = state.queue.map(item => 
            ids.has(item.id) ? { ...item, ...updates } : item
          );
        }),
        
        reorderQueue: (fromIndex, toIndex) => set((state) => {
          const item = state.queue[fromIndex];
          state.queue.splice(fromIndex, 1);
          state.queue.splice(toIndex, 0, item);
        })
      })),
      {
        name: 'processing-store',
        partialize: (state) => ({
          settings: state.settings
        })
      }
    )
  )
);

// stores/photoStore.ts
export const usePhotoStore = create<PhotoState>()(
  immer((set, get) => ({
    photos: new Map(),
    currentPhotoId: null,
    
    loadPhoto: async (photoId: string) => {
      const photo = await api.getPhoto(photoId);
      set((state) => {
        state.photos.set(photoId, photo);
      });
      return photo;
    },
    
    updateAIResults: (photoId: string, results: AIResults) => {
      set((state) => {
        const photo = state.photos.get(photoId);
        if (photo) {
          photo.aiResults = results;
        }
      });
    },
    
    updateRecipe: (photoId: string, recipe: ProcessingRecipe) => {
      set((state) => {
        const photo = state.photos.get(photoId);
        if (photo) {
          photo.processingRecipe = recipe;
        }
      });
    }
  }))
);
```

## Key Features

### 1. Real-time Updates
- WebSocket connection for live processing status
- Progress tracking for individual photos
- System health monitoring

### 2. Interactive Photo Editing
- AI overlay visualization
- Manual crop adjustment
- Before/after comparison
- Recipe-based editing

### 3. Queue Management
- Drag-and-drop prioritization
- Batch operations
- Processing mode selection
- Pause/resume capabilities

### 4. AI Integration
- View detection results
- Adjust AI suggestions
- Quality score visualization
- Confidence indicators

### 5. Recipe System
- Create custom processing recipes
- Save and share recipes
- Real-time preview
- Version control

## Responsive Design

The frontend is fully responsive with breakpoints:
- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

## Performance Optimizations

1. **Virtual Scrolling**: For large photo lists
2. **Image Lazy Loading**: Load images as needed
3. **WebWorker Processing**: Offload heavy computations
4. **Optimistic Updates**: Immediate UI feedback
5. **Request Debouncing**: Prevent API spam

## Deployment

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```