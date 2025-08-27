import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  BarChart3,
  TrendingUp,
  Truck,
  Settings,
  Activity,
  Monitor,
  Menu,
  X,
  Bell,
  Search,
  User
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: BarChart3 },
  { name: 'Prévision Demande', href: '/demand', icon: TrendingUp },
  { name: 'Transport', href: '/transport', icon: Truck },
  { name: 'Maintenance', href: '/maintenance', icon: Settings },
  { name: 'Métriques', href: '/metrics', icon: Activity },
  { name: 'Système', href: '/system', icon: Monitor },
]

function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen bg-pattern-dark cyber-grid">
      {/* Mobile sidebar */}
      {sidebarOpen && (
        <div className="fixed inset-0 flex z-40 md:hidden">
          <div className="fixed inset-0 bg-black bg-opacity-70 backdrop-blur-sm" onClick={() => setSidebarOpen(false)} />
          <div className="relative flex-1 flex flex-col max-w-xs w-full glass-card">
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              <button
                className="ml-1 flex items-center justify-center h-12 w-12 rounded-full hover-bounce glass-card pulse-neon"
                onClick={() => setSidebarOpen(false)}
              >
                <X className="h-6 w-6 dark-text-primary" />
              </button>
            </div>
            <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
              <div className="flex-shrink-0 flex items-center px-4 mb-8">
                <div className="glass-card p-4 mr-4 pulse-neon">
                  <BarChart3 className="h-10 w-10 text-blue-400" />
                </div>
                <h1 className="text-xl font-bold gradient-text text-glow">SUPPLY CHAIN AI</h1>
              </div>
              <nav className="mt-5 px-2 space-y-3">
                {navigation.map((item) => {
                  const isActive = location.pathname === item.href
                  return (
                    <Link
                      key={item.name}
                      to={item.href}
                      className={`group flex items-center px-2 py-2 text-base font-medium rounded-md ${
                        isActive
                          ? 'bg-primary-100 text-primary-900'
                          : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                      }`}
                      onClick={() => setSidebarOpen(false)}
                    >
                      <item.icon className="mr-4 h-6 w-6" />
                      {item.name}
                    </Link>
                  )
                })}
              </nav>
            </div>
          </div>
        </div>
      )}

      {/* Static sidebar for desktop */}
      <div className="hidden md:flex md:w-64 md:flex-col md:fixed md:inset-y-0">
        <div className="glass-card m-4 flex-1 flex flex-col overflow-y-auto">
          <div className="flex-1 flex flex-col pt-5 pb-4">
            <div className="flex items-center flex-shrink-0 px-4 mb-8">
              <div className="stat-card p-3 mr-3">
                <BarChart3 className="h-8 w-8 text-blue-600 icon-bounce" />
              </div>
              <h1 className="text-xl font-bold gradient-text">
                � Supply Chain AI
              </h1>
            </div>
            <nav className="mt-5 flex-1 px-2 space-y-2">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`group flex items-center px-3 py-3 text-sm font-medium rounded-xl transition-all duration-300 hover-float ${
                      isActive
                        ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg'
                        : 'text-gray-700 hover:bg-white/30 hover:text-gray-900'
                    }`}
                  >
                    <span className="text-xl mr-3">{item.emoji}</span>
                    <item.icon className="mr-2 h-5 w-5" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
          </div>
          <div className="flex-shrink-0 p-4">
            <div className="glass-card p-4 hover-bounce">
              <div className="flex items-center">
                <div className="h-10 w-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
                  <User className="h-5 w-5 text-white" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-800">Admin</p>
                  <p className="text-xs text-gray-600">admin@company.com</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="md:pl-64 flex flex-col flex-1">
        <div className="md:hidden p-4">
          <button
            className="glass-card p-3 hover-bounce"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-6 w-6 text-gray-600" />
          </button>
        </div>

        {/* Top header */}
        <header className="glass-card m-6 shadow-soft pulse-neon">
          <div className="px-8 py-6">
            <div className="flex justify-between items-center h-16">
              <div className="flex items-center">
                <h2 className="text-3xl font-bold gradient-text text-glow">
                  COMMAND CENTER NEURAL NETWORK
                </h2>
              </div>
              <div className="flex items-center space-x-6">
                <div className="relative">
                  <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-blue-400" />
                  <input
                    type="text"
                    placeholder="Recherche quantique..."
                    className="input pl-12 pr-6 py-3 w-80 text-lg"
                  />
                </div>
                <button className="relative glass-card p-4 hover-bounce pulse-neon">
                  <Bell className="h-6 w-6 text-blue-400 icon-bounce" />
                  <span className="absolute -top-2 -right-2 block h-4 w-4 rounded-full bg-red-500 ring-2 ring-white pulse-glow text-xs text-white font-bold flex items-center justify-center">3</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 px-4">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

export default Layout
