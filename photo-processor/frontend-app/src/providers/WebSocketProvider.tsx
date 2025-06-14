import React, { createContext, useContext, useEffect, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { WebSocketEvent } from '@/types/api'

type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface WebSocketContextType {
  status: WebSocketStatus
  isConnected: boolean
}

const WebSocketContext = createContext<WebSocketContextType>({
  status: 'disconnected',
  isConnected: false
})

export const useWebSocketStatus = () => useContext(WebSocketContext)

// Global WebSocket singleton to prevent multiple connections in StrictMode
let globalWS: WebSocket | null = null
let globalReconnectTimeout: ReturnType<typeof setTimeout> | null = null
let globalPingInterval: ReturnType<typeof setInterval> | null = null
let globalReconnectAttempts = 0
let isInitialConnection = true
let currentStatus: WebSocketStatus = 'disconnected'
const subscribers = new Set<(status: WebSocketStatus) => void>()
const queryClients = new Set<any>()
let lastPongReceived = Date.now()

const notifySubscribers = (status: WebSocketStatus) => {
  currentStatus = status
  subscribers.forEach(callback => callback(status))
}

const cleanupPingInterval = () => {
  if (globalPingInterval) {
    clearInterval(globalPingInterval)
    globalPingInterval = null
  }
}

const connectWebSocket = () => {
  // Don't create new connection if one exists and is connecting/open
  if (globalWS && (globalWS.readyState === WebSocket.CONNECTING || globalWS.readyState === WebSocket.OPEN)) {
    return
  }

  // Clean up existing connection and intervals
  cleanupPingInterval()
  if (globalWS) {
    globalWS.close()
    globalWS = null
  }

  try {
    // Construct WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`
    
    // Only log in development
    if (import.meta.env.DEV) {
      console.log('WebSocket connecting to:', wsUrl)
    }
    
    globalWS = new WebSocket(wsUrl)
    notifySubscribers('connecting')

    globalWS.onopen = () => {
      notifySubscribers('connected')
      globalReconnectAttempts = 0
      lastPongReceived = Date.now()
      
      if (!isInitialConnection) {
        toast.success('Real-time connection restored')
      }
      isInitialConnection = false
      
      // Setup ping/pong keepalive
      cleanupPingInterval()
      globalPingInterval = setInterval(() => {
        if (globalWS && globalWS.readyState === WebSocket.OPEN) {
          // Check if we've received a pong recently
          const timeSinceLastPong = Date.now() - lastPongReceived
          if (timeSinceLastPong > 30000) { // 30 seconds timeout
            console.warn('No pong received in 30 seconds, closing connection')
            globalWS.close()
            return
          }
          
          try {
            globalWS.send('ping')
          } catch (error) {
            console.error('Failed to send ping:', error)
          }
        }
      }, 10000) // Send ping every 10 seconds
    }

    globalWS.onmessage = (event) => {
      try {
        // Handle ping/pong messages
        if (event.data === 'pong') {
          lastPongReceived = Date.now()
          return
        }
        
        if (event.data === 'ping') {
          if (globalWS && globalWS.readyState === WebSocket.OPEN) {
            globalWS.send('pong')
          }
          return
        }
        
        const data: WebSocketEvent = JSON.parse(event.data)

        // Notify all query clients
        queryClients.forEach(queryClient => {
          switch (data.type) {
            case 'processing_started':
              toast.info(`Processing started: ${data.data.filename}`)
              queryClient.invalidateQueries(['processing', 'status'])
              queryClient.invalidateQueries(['processing', 'queue'])
              // Emit custom event for components to listen to
              window.dispatchEvent(new CustomEvent('processing_started', { detail: data }))
              break

            case 'processing_completed':
              toast.success(`Processing completed: ${data.data.filename}`)
              queryClient.invalidateQueries(['photos'])
              queryClient.invalidateQueries(['processing', 'status'])
              queryClient.invalidateQueries(['processing', 'queue'])
              queryClient.invalidateQueries(['stats'])
              // Emit custom event for components to listen to
              window.dispatchEvent(new CustomEvent('processing_completed', { detail: data }))
              break

            case 'processing_failed':
              toast.error(`Processing failed: ${data.data.filename} - ${data.data.error}`)
              queryClient.invalidateQueries(['processing', 'status'])
              queryClient.invalidateQueries(['processing', 'queue'])
              break

            case 'queue_updated':
              queryClient.invalidateQueries(['processing', 'queue'])
              break

            case 'stats_updated':
              queryClient.invalidateQueries(['stats'])
              break

            case 'photo_uploaded':
              toast.success(`Photo uploaded: ${data.data.filename}`)
              queryClient.invalidateQueries(['photos'])
              queryClient.invalidateQueries(['stats'])
              break

            case 'recipe_updated':
              queryClient.invalidateQueries(['recipes'])
              if (data.data.action === 'created') {
                toast.success(`Recipe created: ${data.data.recipe.name}`)
              } else if (data.data.action === 'updated') {
                toast.info(`Recipe updated: ${data.data.recipe.name}`)
              } else if (data.data.action === 'deleted') {
                toast.info(`Recipe deleted: ${data.data.recipe.name}`)
              }
              break

            case 'rotation_analysis_progress':
              // Emit custom event for recipe builder to listen to
              window.dispatchEvent(new CustomEvent('rotation_analysis_progress', { detail: data }))
              break

            case 'rotation_analysis_complete':
              // Emit custom event for recipe builder to listen to
              window.dispatchEvent(new CustomEvent('rotation_analysis_complete', { detail: data }))
              break

            case 'rotation_analysis_failed':
              // Emit custom event for recipe builder to listen to
              window.dispatchEvent(new CustomEvent('rotation_analysis_failed', { detail: data }))
              break

            case 'composition_analysis_progress':
              // Emit custom event for recipe builder to listen to
              window.dispatchEvent(new CustomEvent('composition_analysis_progress', { detail: data }))
              break

            case 'composition_analysis_complete':
              // Emit custom event for recipe builder to listen to
              window.dispatchEvent(new CustomEvent('composition_analysis_complete', { detail: data }))
              break

            case 'crop_generation_complete':
              // Emit custom event for recipe builder to listen to
              window.dispatchEvent(new CustomEvent('crop_generation_complete', { detail: data }))
              break

            case 'recipe_builder_started':
              // Emit custom event for recipe builder to listen to
              window.dispatchEvent(new CustomEvent('recipe_builder_started', { detail: data }))
              break

            case 'connection':
              // Connection status message from server, ignore
              break

            default:
              // Ignore unhandled event types silently
          }
        })
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    globalWS.onerror = () => {
      console.error('WebSocket connection error')
      notifySubscribers('error')
    }

    globalWS.onclose = (event) => {
      notifySubscribers('disconnected')
      cleanupPingInterval()
      
      // Only log unexpected closures
      if (!event.wasClean && event.code !== 1000) {
        console.warn('WebSocket closed unexpectedly:', event.code, event.reason)
      }
      
      const maxReconnectAttempts = 10
      const baseReconnectInterval = 1000
      const maxReconnectInterval = 30000
      
      // Only attempt reconnection if we haven't exceeded max attempts
      if (globalReconnectAttempts < maxReconnectAttempts) {
        globalReconnectAttempts++
        
        // Exponential backoff with jitter
        const reconnectInterval = Math.min(
          baseReconnectInterval * Math.pow(2, globalReconnectAttempts - 1) + Math.random() * 1000,
          maxReconnectInterval
        )
        
        if (globalReconnectAttempts === 1) {
          toast.warning('Real-time connection lost. Attempting to reconnect...')
        } else if (globalReconnectAttempts % 3 === 0) {
          toast.info(`Reconnecting... (attempt ${globalReconnectAttempts}/${maxReconnectAttempts})`)
        }
        
        globalReconnectTimeout = setTimeout(() => {
          connectWebSocket()
        }, reconnectInterval)
      } else {
        toast.error('Unable to establish real-time connection. Please refresh the page.')
      }
    }
  } catch (error) {
    console.error('Failed to create WebSocket connection:', error)
    notifySubscribers('error')
  }
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<WebSocketStatus>(currentStatus)
  const queryClient = useQueryClient()

  useEffect(() => {
    // Add this provider to subscribers
    subscribers.add(setStatus)
    queryClients.add(queryClient)
    
    let connectTimer: ReturnType<typeof setTimeout> | null = null
    
    // Connect if not already connected
    if (!globalWS || globalWS.readyState === WebSocket.CLOSED) {
      // Add a small delay to ensure API is ready
      connectTimer = setTimeout(() => {
        connectWebSocket()
      }, 500)
    } else {
      // Sync current status
      setStatus(currentStatus)
    }

    // Cleanup function
    return () => {
      if (connectTimer) {
        clearTimeout(connectTimer)
      }
      
      subscribers.delete(setStatus)
      queryClients.delete(queryClient)
      
      // Only cleanup global connection if no more subscribers
      if (subscribers.size === 0) {
        if (globalReconnectTimeout) {
          clearTimeout(globalReconnectTimeout)
          globalReconnectTimeout = null
        }
        cleanupPingInterval()
        if (globalWS) {
          globalWS.close()
          globalWS = null
        }
        globalReconnectAttempts = 0
        isInitialConnection = true
      }
    }
  }, [queryClient])

  return (
    <WebSocketContext.Provider value={{ status, isConnected: status === 'connected' }}>
      {children}
    </WebSocketContext.Provider>
  )
}