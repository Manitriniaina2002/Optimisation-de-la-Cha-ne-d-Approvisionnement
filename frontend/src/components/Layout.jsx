import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
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
  Monitor
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
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative overflow-hidden">
      {/* Decorative background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-200/30 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-200/30 rounded-full blur-3xl"></div>
      </div>

      {/* Header card */}
      <div className="relative z-10 p-6">
        <div className="relative overflow-hidden bg-white/80 backdrop-blur-xl rounded-3xl p-6 border border-gray-200/50 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-md">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-purple-700">Neural Command Center</h1>
                <p className="text-sm text-gray-600">AI-Powered Supply Chain Intelligence</p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <div className={`px-3 py-2 rounded-2xl text-sm font-bold ${apiConnected ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-700'}`}>
                {apiConnected ? 'SYSTÈME ACTIF' : 'HORS LIGNE'}
              </div>
              <div className="hidden md:block">
                <input className="input" placeholder="Recherche..." />
              </div>
              <button className="p-2 rounded-lg bg-white shadow"><Bell className="w-5 h-5 text-gray-600" /></button>
            </div>
          </div>
        </div>
      </div>

      {/* Main layout */}
      <div className="relative z-10 md:flex">
        {/* Sidebar */}
        <aside className="hidden md:block md:w-64 p-4">
          <div className="bg-white rounded-2xl p-4 shadow-md border">
            <div className="flex items-center mb-6">
              <div className="p-2 bg-blue-50 rounded-lg mr-3"><BarChart3 className="w-6 h-6 text-blue-600" /></div>
              <div>
                <div className="text-sm font-bold">SUPPLY CHAIN AI</div>
                <div className="text-xs text-gray-500">Admin</div>
              </div>
            </div>

            <nav className="space-y-2">
              {navigation.map(item => {
                const active = location.pathname === item.href
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg ${active ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white' : 'text-gray-700 hover:bg-gray-50'}`}>
                    <item.icon className="w-5 h-5" />
                    <span className="text-sm font-medium">{item.name}</span>
                  </Link>
                )
              })}
            </nav>
          </div>
        </aside>

        {/* Content area */}
        <main className="flex-1 p-6">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
