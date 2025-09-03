import React, { useState } from 'react'
import { Link, NavLink } from 'react-router-dom'
import {
  Activity,
  Bell,
  Menu,
  X,
  Search,
  User,
  BarChart3,
  TrendingUp,
  Truck,
  Settings,
  Monitor,
  ChevronDown,
  Sparkles
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: BarChart3 },
  { name: 'Prévision Demande', href: '/demand', icon: TrendingUp },
  { name: 'Transport', href: '/transport', icon: Truck },
  { name: 'Maintenance', href: '/maintenance', icon: Settings },
  { name: 'Métriques', href: '/metrics', icon: Activity },
  { name: 'Système', href: '/system', icon: Monitor }
]

export default function Layout({ children, apiConnected = true }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Full-width Header */}
      <header className="fixed top-0 inset-x-0 z-40 h-20">
        <div className="h-full bg-white/90 backdrop-blur-xl border-b border-gray-200/50 shadow-lg">
          {/* Decorative gradient overlay */}
         
          <div className="relative h-full flex items-center justify-between px-6 md:px-8">
            {/* Left side - Brand and Mobile Menu */}
            <div className="flex items-center space-x-4">
              {/* Mobile menu button */}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="md:hidden p-2 rounded-xl bg-white/80 shadow-md hover:shadow-lg transition-all duration-200"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>

              {/* Brand */}
              <Link to="/" className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 via-purple-600 to-cyan-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <Activity className="w-7 h-7 text-white" />
                </div>
                <div className="hidden sm:block">
                  <h1 className="text-2xl font-black bg-gradient-to-r from-gray-800 via-blue-700 to-purple-700 bg-clip-text text-transparent">
                    Centre de Commande Neuronal
                  </h1>
                  <p className="text-sm text-gray-600 font-medium">IA pour l'optimisation de la chaîne d'approvisionnement</p>
                </div>
              </Link>
            </div>

            {/* Center - Search */}
            <div className="hidden lg:flex flex-1 max-w-lg mx-8">
              <div className="relative w-full">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Rechercher dans le système..."
                  className="w-full pl-12 pr-4 py-3 bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all duration-200"
                />
              </div>
            </div>

            {/* Right side - Status and Actions */}
            <div className="flex items-center space-x-4">
              {/* API Status */}
              <div className={`hidden sm:flex items-center space-x-2 px-4 py-2 rounded-2xl font-bold text-sm transition-all duration-200 ${
                apiConnected 
                  ? 'bg-emerald-100/80 text-emerald-800 border border-emerald-200/50' 
                  : 'bg-red-100/80 text-red-800 border border-red-200/50'
              }`}>
                <div className={`w-2 h-2 rounded-full ${apiConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
                <span>{apiConnected ? 'SYSTÈME ACTIF' : 'HORS LIGNE'}</span>
              </div>

              {/* Notifications */}
              <button className="relative p-3 rounded-xl bg-white/80 shadow-md hover:shadow-lg transition-all duration-200 group">
                <Bell className="w-5 h-5 text-gray-600 group-hover:text-blue-600 transition-colors" />
                <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse"></span>
              </button>

              {/* Profile */}
              <div className="flex items-center space-x-2 px-3 py-2 bg-white/80 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 cursor-pointer group">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <User className="w-5 h-5 text-white" />
                </div>
                <ChevronDown className="w-4 h-4 text-gray-500 group-hover:text-gray-700 transition-colors hidden sm:block" />
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex pt-20 ">
        {/* Full-height Sidebar */}
        <aside className={`fixed inset-y-0 mt-20 left-0 z-30 w-72 transform transition-transform duration-300 ease-in-out ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:translate-x-0`}> 
          <div className="h-full bg-white/80 backdrop-blur-xl border-r border-gray-200/50 shadow-xl flex flex-col min-h-0">
            {/* Navigation */}
            <nav className="flex-1 p-6 overflow-y-auto pr-4 md:pr-6 min-h-0">
              <div className="space-y-2">
                {navigation.map((item) => (
                  <NavLink
                    key={item.name}
                    to={item.href}
                    end={item.href === '/'}
                    onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) => `group flex items-center space-x-3 px-4 py-3 rounded-2xl font-medium transition-all duration-200 w-full text-left min-w-0 pr-3 ${
                      isActive
                        ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg scale-[1.02]'
                        : 'text-gray-700 hover:bg-gray-100/80 hover:scale-[1.01] hover:shadow-md'
                    }`}
                  >
                    {({ isActive }) => (
                      <>
                        <div className={`p-1.5 rounded-lg ${
                          isActive
                            ? 'bg-white/20'
                            : 'bg-gray-100 group-hover:bg-gray-200 transition-colors'
                        }`}>
                          <item.icon className={`w-5 h-5 ${isActive ? 'text-white' : 'text-gray-600'}`} />
                        </div>
                        <span className="text-sm truncate">{item.name}</span>
                        {isActive && (
                          <div className="ml-auto w-2 h-2 bg-white/80 rounded-full animate-pulse"></div>
                        )}
                      </>
                    )}
                  </NavLink>
                ))}
              </div>
            </nav>

            {/* Sidebar Footer */}
            <div className="p-6 border-t border-gray-200/50">
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-4 border border-gray-200/50">
                <div className="flex items-center space-x-3 mb-2">
                  <Activity className="w-5 h-5 text-blue-600" />
                  <span className="text-sm font-bold text-gray-800">Système Neural</span>
                </div>
                <div className="text-xs text-gray-600 mb-3">
                  Intelligence artificielle avancée pour l'optimisation de la chaîne d'approvisionnement
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">Version 2.1.0</span>
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* Mobile overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-20 bg-black/50 backdrop-blur-sm md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 min-h-screen ml-0 md:ml-72">
          <div className="h-full p-6 md:p-8">
            <div className="max-w-7xl mx-auto">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}