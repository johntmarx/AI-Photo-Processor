import React, { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { 
  LayoutDashboard, 
  Image as Images, 
  Cpu, 
  BookOpen, 
  Settings,
  Camera,
  Activity,
  Package
} from 'lucide-react'
import ConnectionStatus from './ConnectionStatus'

interface LayoutProps {
  children: ReactNode
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Photos', href: '/photos', icon: Images },
  { name: 'Processing', href: '/processing', icon: Cpu },
  { name: 'Batch Process', href: '/batch', icon: Package },
  { name: 'Recipes', href: '/recipes', icon: BookOpen },
  { name: 'Settings', href: '/settings', icon: Settings },
]

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border">
        <div className="flex h-16 items-center px-6 border-b border-border">
          <div className="flex items-center space-x-2">
            <Camera className="h-8 w-8 text-primary" />
            <span className="text-xl font-bold">Photo Processor</span>
          </div>
        </div>
        
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                className={cn(
                  "flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                )}
              >
                {React.createElement(item.icon, { className: "h-5 w-5" })}
                <span>{item.name}</span>
              </Link>
            )
          })}
        </nav>

        {/* Connection Status */}
        <div className="p-4 border-t border-border">
          <ConnectionStatus />
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-semibold">
              {navigation.find(item => item.href === location.pathname)?.name || 
               (location.pathname.startsWith('/recipes/') ? 'Recipes' : 'Dashboard')}
            </h1>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm text-muted-foreground">
              <Activity className="h-4 w-4" />
              <span>Real-time monitoring active</span>
            </div>
          </div>
        </header>

        <main className="p-6">
          {children}
        </main>
      </div>
    </div>
  )
}