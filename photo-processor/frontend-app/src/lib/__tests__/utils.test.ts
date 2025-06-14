import { formatBytes, formatDuration, formatRelativeTime, cn } from '../utils'

describe('utils', () => {
  describe('formatBytes', () => {
    it('formats bytes correctly', () => {
      expect(formatBytes(0)).toBe('0 Bytes')
      expect(formatBytes(1024)).toBe('1.00 KB')
      expect(formatBytes(1048576)).toBe('1.00 MB')
      expect(formatBytes(1073741824)).toBe('1.00 GB')
      expect(formatBytes(1536)).toBe('1.50 KB')
    })

    it('handles custom decimals', () => {
      expect(formatBytes(1536, 0)).toBe('2 KB')
      expect(formatBytes(1536, 1)).toBe('1.5 KB')
      expect(formatBytes(1536, 3)).toBe('1.500 KB')
    })

    it('handles negative decimals', () => {
      expect(formatBytes(1536, -1)).toBe('2 KB')
    })
  })

  describe('formatDuration', () => {
    it('formats seconds correctly', () => {
      expect(formatDuration(30)).toBe('30s')
      expect(formatDuration(45.7)).toBe('46s')
    })

    it('formats minutes correctly', () => {
      expect(formatDuration(90)).toBe('2m')
      expect(formatDuration(150)).toBe('3m')
    })

    it('formats hours correctly', () => {
      expect(formatDuration(3600)).toBe('1h')
      expect(formatDuration(7200)).toBe('2h')
      expect(formatDuration(5400)).toBe('2h')
    })
  })

  describe('formatRelativeTime', () => {
    const now = new Date('2024-01-01T12:00:00Z')
    
    beforeEach(() => {
      vi.useFakeTimers()
      vi.setSystemTime(now)
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('formats recent times', () => {
      const recent = new Date('2024-01-01T11:59:30Z')
      expect(formatRelativeTime(recent)).toBe('just now')
    })

    it('formats minutes ago', () => {
      const minutes = new Date('2024-01-01T11:55:00Z')
      expect(formatRelativeTime(minutes)).toBe('5m ago')
    })

    it('formats hours ago', () => {
      const hours = new Date('2024-01-01T10:00:00Z')
      expect(formatRelativeTime(hours)).toBe('2h ago')
    })

    it('formats days ago', () => {
      const days = new Date('2023-12-30T12:00:00Z')
      expect(formatRelativeTime(days)).toBe('2d ago')
    })

    it('formats older dates', () => {
      const old = new Date('2023-12-20T12:00:00Z')
      expect(formatRelativeTime(old)).toBe('12/20/2023')
    })
  })

  describe('cn', () => {
    it('merges class names correctly', () => {
      expect(cn('base', 'additional')).toBe('base additional')
    })

    it('handles conditional classes', () => {
      expect(cn('base', true && 'conditional')).toBe('base conditional')
      expect(cn('base', false && 'conditional')).toBe('base')
    })

    it('handles Tailwind conflicts', () => {
      // twMerge should resolve conflicts
      expect(cn('px-2', 'px-4')).toBe('px-4')
      expect(cn('text-red-500', 'text-blue-500')).toBe('text-blue-500')
    })

    it('handles undefined and null values', () => {
      expect(cn('base', undefined, null, 'additional')).toBe('base additional')
    })

    it('handles arrays', () => {
      expect(cn(['base', 'additional'])).toBe('base additional')
    })

    it('handles objects', () => {
      expect(cn({
        'base': true,
        'conditional': false,
        'included': true
      })).toBe('base included')
    })
  })
})