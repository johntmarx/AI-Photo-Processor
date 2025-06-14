import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://photo_processor_api:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://photo_processor_api:8000',
        changeOrigin: true,
      },
      '/images': {
        target: 'http://photo_processor_api:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://photo_processor_api:8000',
        ws: true,
        changeOrigin: true,
        secure: false,
        logLevel: 'debug',
        // Configure WebSocket proxy
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.error('WebSocket proxy error:', err);
          });
          proxy.on('proxyReqWs', (proxyReq, req, socket, options, head) => {
            console.log('WebSocket upgrade request:', req.url);
            socket.on('error', (error) => {
              console.error('WebSocket socket error:', error);
            });
          });
          proxy.on('open', (proxySocket) => {
            console.log('WebSocket proxy connection opened');
          });
          proxy.on('close', (res, socket, head) => {
            console.log('WebSocket proxy connection closed');
          });
        },
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/dist/**',
      ],
    },
  },
})