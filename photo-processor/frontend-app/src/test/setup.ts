import '@testing-library/jest-dom'
import { beforeAll, afterAll, afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

// Mock WebSocket for tests
(globalThis as any).WebSocket = class MockWebSocket {
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  readyState: number = 1
  
  constructor(public url: string) {}
  
  send() {
    // Mock implementation
  }
  
  close() {
    // Mock implementation
  }
} as any

// runs a cleanup after each test case (e.g. clearing jsdom)
afterEach(() => {
  cleanup()
})

beforeAll(() => {
  // Mock IntersectionObserver
  (globalThis as any).IntersectionObserver = class IntersectionObserver {
    constructor() {}
    disconnect() {}
    observe() {}
    unobserve() {}
  }
})

afterAll(() => {
  // cleanup
})