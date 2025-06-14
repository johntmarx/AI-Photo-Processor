import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatBytes(bytes: number, decimals = 2) {
  // Handle null, undefined, or non-numeric values
  if (bytes == null || isNaN(bytes) || bytes < 0) return '0 Bytes'
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return (bytes / Math.pow(k, i)).toFixed(dm) + ' ' + sizes[i]
}

export function formatDuration(seconds: number) {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.round(seconds / 3600)}h`
}

export function formatRelativeTime(date: Date | string | null | undefined) {
  // Handle null, undefined, or invalid dates
  if (!date) return 'Unknown'
  
  let dateObj: Date
  try {
    dateObj = typeof date === 'string' ? new Date(date) : date
    
    // Check if date is valid
    if (isNaN(dateObj.getTime())) {
      return 'Invalid date'
    }
  } catch {
    return 'Invalid date'
  }
  
  const now = new Date()
  const diffMs = now.getTime() - dateObj.getTime()
  const diffSeconds = diffMs / 1000
  const diffMinutes = diffSeconds / 60
  const diffHours = diffMinutes / 60
  const diffDays = diffHours / 24

  if (diffSeconds < 60) return 'just now'
  if (diffMinutes < 60) return `${Math.round(diffMinutes)}m ago`
  if (diffHours < 24) return `${Math.round(diffHours)}h ago`
  if (diffDays < 7) return `${Math.round(diffDays)}d ago`
  
  return dateObj.toLocaleDateString()
}