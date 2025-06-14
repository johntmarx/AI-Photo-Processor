# Photo Processor Frontend

A modern React TypeScript frontend for the Photo Processor application, built with Vite and comprehensive testing.

## 🚀 Features

- **Real-time Dashboard** - Live statistics and monitoring
- **Photo Management** - Upload, view, and manage photos with before/after comparison
- **Recipe System** - Create and manage processing recipes
- **Processing Control** - Pause/resume processing, queue management
- **WebSocket Integration** - Real-time updates and notifications
- **Responsive Design** - Mobile-friendly interface
- **No Authentication** - Designed for local network access

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **State Management**: React Query for server state
- **Routing**: React Router v6
- **Testing**: Vitest + React Testing Library
- **WebSocket**: Custom hooks for real-time updates
- **Deployment**: Docker + Nginx

## 📦 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── ui/             # Base UI components (Button, Card, etc.)
│   │   ├── dashboard/      # Dashboard-specific components
│   │   ├── photos/         # Photo management components
│   │   └── Layout.tsx      # Main app layout
│   ├── pages/              # Page components
│   │   ├── Dashboard.tsx   # Main dashboard
│   │   ├── Photos.tsx      # Photo management
│   │   ├── Processing.tsx  # Processing controls
│   │   ├── Recipes.tsx     # Recipe management
│   │   └── Settings.tsx    # Application settings
│   ├── hooks/              # Custom React hooks
│   │   ├── useWebSocket.ts # WebSocket management
│   │   └── useRealTimeUpdates.ts # Real-time data updates
│   ├── services/           # API layer
│   │   └── api.ts          # API client and endpoints
│   ├── types/              # TypeScript type definitions
│   │   └── api.ts          # API response types
│   ├── lib/                # Utility functions
│   │   └── utils.ts        # Helper functions
│   └── test/               # Test utilities
│       ├── setup.ts        # Test configuration
│       └── test-utils.tsx  # Testing utilities
├── public/                 # Static assets
├── Docker/                 # Docker configuration
│   ├── Dockerfile          # Production build
│   ├── Dockerfile.dev      # Development build
│   └── nginx.conf          # Nginx configuration
└── __tests__/              # Component tests
```

## 🔧 Development

### Prerequisites

- Node.js 18+
- npm or yarn

### Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm run test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage

# Build for production
npm run build

# Preview production build
npm run preview
```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.frontend.yml up -d

# Access the application
open http://localhost:3000
```

## 🧪 Testing

The frontend includes comprehensive test coverage:

- **Unit Tests**: Individual components and hooks
- **Integration Tests**: Component interactions
- **Mocking**: WebSocket, API calls, and external dependencies

### Test Structure

```
src/
├── components/
│   ├── __tests__/          # Component tests
│   └── ui/
│       └── __tests__/      # UI component tests
├── hooks/
│   └── __tests__/          # Hook tests
└── test/
    ├── setup.ts            # Global test setup
    └── test-utils.tsx      # Custom render functions
```

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm test -- PhotoGrid.test.tsx
```

## 🎨 UI Components

The frontend uses a custom component library built on TailwindCSS:

### Base Components
- `Button` - Configurable button with variants and sizes
- `Card` - Container component with header, content, footer
- `Badge` - Status indicators and labels
- `Input` - Form input with validation styling
- `Progress` - Progress bars and loading indicators

### Feature Components
- `PhotoGrid` - Responsive photo gallery with selection
- `PhotoDialog` - Detailed photo view with AI analysis
- `StatsCard` - Dashboard metrics display
- `ProcessingChart` - Real-time processing graphs
- `ActivityFeed` - Live activity stream

## 🔄 Real-time Features

### WebSocket Integration

The frontend maintains a persistent WebSocket connection for real-time updates:

```typescript
// Automatic connection management
const { isConnected, status } = useRealTimeUpdates()

// Manual WebSocket usage
const { sendMessage, lastMessage } = useWebSocket('/ws', {
  onMessage: (event) => console.log('Received:', event),
  autoReconnect: true
})
```

### Event Types

- `processing_started` - Photo processing begins
- `processing_completed` - Photo processing finished
- `processing_failed` - Processing error occurred
- `queue_updated` - Processing queue changed
- `stats_updated` - Dashboard statistics updated
- `photo_uploaded` - New photo uploaded
- `recipe_updated` - Recipe created/modified

## 📡 API Integration

### React Query Configuration

The frontend uses React Query for efficient server state management:

```typescript
// Automatic caching and background refetching
const { data: photos, isLoading } = useQuery(
  ['photos', page, filter],
  () => photosApi.list(page, 20, filter),
  { refetchInterval: 5000 }
)

// Optimistic updates
const mutation = useMutation(photosApi.upload, {
  onSuccess: () => {
    queryClient.invalidateQueries(['photos'])
    toast.success('Photo uploaded!')
  }
})
```

### API Endpoints

- `GET /api/photos` - List photos with pagination
- `POST /api/photos/upload` - Upload new photos
- `GET /api/processing/queue` - Get processing queue status
- `POST /api/processing/pause` - Pause processing
- `GET /api/recipes` - List processing recipes
- `GET /api/stats/dashboard` - Get dashboard statistics

## 🐳 Docker Deployment

### Production Deployment

```yaml
# docker-compose.frontend.yml
services:
  frontend:
    build: 
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - api
```

### Multi-stage Build

The Dockerfile uses multi-stage builds for optimal production images:

1. **Build Stage**: Compile TypeScript and build assets
2. **Production Stage**: Serve with Nginx

### Nginx Configuration

- **Client-side Routing**: Fallback to `index.html`
- **API Proxy**: Forward `/api/*` to backend
- **WebSocket Proxy**: Handle `/ws` upgrades
- **Asset Caching**: Optimize static file delivery
- **Gzip Compression**: Reduce bandwidth usage

## 🔧 Configuration

### Environment Variables

The frontend supports configuration through build-time variables:

```bash
# API endpoint (proxied through Nginx in production)
VITE_API_URL=http://localhost:8000

# WebSocket endpoint
VITE_WS_URL=ws://localhost:8000/ws
```

### Vite Configuration

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true }
    }
  }
})
```

## 🚀 Deployment

### Development

```bash
# Start backend API first
cd ../api && docker-compose up -d

# Start frontend
npm run dev
```

### Production

```bash
# Build and deploy full stack
docker-compose -f docker-compose.frontend.yml up -d

# Access application
open http://localhost:3000
```

### Health Checks

The frontend includes health check endpoints:

- `GET /health` - Simple health check
- WebSocket connection status in UI
- Real-time backend connectivity monitoring

## 📊 Performance

- **Bundle Size**: Optimized with tree-shaking
- **Code Splitting**: Route-based lazy loading
- **Caching**: Aggressive caching of API responses
- **Real-time**: Efficient WebSocket management
- **Images**: Lazy loading and optimization
- **Metrics**: Built-in performance monitoring

## 🔒 Security

- **No Authentication**: Designed for trusted local networks
- **CORS**: Properly configured for local development
- **XSS Protection**: React's built-in protections
- **Input Validation**: Client-side validation for UX
- **API Validation**: Server-side validation required

## 📱 Responsive Design

The interface adapts to different screen sizes:

- **Desktop**: Full sidebar navigation
- **Tablet**: Collapsible navigation
- **Mobile**: Bottom navigation bar
- **Grid Layouts**: Responsive photo grids
- **Touch Support**: Mobile-friendly interactions

## 🎯 Future Enhancements

- **Progressive Web App**: Offline capability
- **Advanced Filtering**: More photo filter options
- **Bulk Operations**: Multi-photo editing
- **Keyboard Shortcuts**: Power user features
- **Themes**: Dark/light mode toggle
- **Export Features**: Batch download capabilities