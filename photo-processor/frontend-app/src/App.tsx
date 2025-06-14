import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'sonner'
import { WebSocketProvider } from './providers/WebSocketProvider'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Photos from './pages/Photos'
import Recipes from './pages/Recipes'
import RecipeEditor from './pages/RecipeEditor'
import RecipeBuilder from './pages/RecipeBuilder'
import Settings from './pages/Settings'
import Processing from './pages/Processing'
import BatchProcessing from './pages/BatchProcessing'

function App() {
  return (
    <WebSocketProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/photos" element={<Photos />} />
          <Route path="/processing" element={<Processing />} />
          <Route path="/recipes" element={<Recipes />} />
          <Route path="/recipes/new" element={<RecipeEditor />} />
          <Route path="/recipes/builder" element={<RecipeBuilder />} />
          <Route path="/recipes/:id/edit" element={<RecipeEditor />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/batch" element={<BatchProcessing />} />
        </Routes>
      </Layout>
      <Toaster 
        position="bottom-right"
        toastOptions={{
          duration: 4000,
        }}
      />
    </WebSocketProvider>
  )
}

export default App