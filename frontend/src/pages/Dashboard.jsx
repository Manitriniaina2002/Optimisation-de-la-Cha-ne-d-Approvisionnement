import React, { useState, useEffect } from 'react'
import { 
  TrendingUp, 
  Truck, 
  Settings, 
  AlertTriangle,
  DollarSign,
  Package,
  Users,
  Zap
} from 'lucide-react'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { dashboardAPI, generalAPI } from '../services/api'

// Mock data for charts
const demandData = [
  { date: '2024-01', actual: 1200, predicted: 1180 },
  { date: '2024-02', actual: 1350, predicted: 1320 },
  { date: '2024-03', actual: 1180, predicted: 1200 },
  { date: '2024-04', actual: 1420, predicted: 1380 },
  { date: '2024-05', actual: 1250, predicted: 1290 },
  { date: '2024-06', actual: 1380, predicted: 1350 },
]

const transportData = [
  { route: 'Route A', efficiency: 92, cost: 1200 },
  { route: 'Route B', efficiency: 87, cost: 1100 },
  { route: 'Route C', efficiency: 95, cost: 1350 },
  { route: 'Route D', efficiency: 89, cost: 1050 },
]

const equipmentStatus = [
  { name: 'Healthy', value: 65, color: '#22c55e' },
  { name: 'Warning', value: 25, color: '#f59e0b' },
  { name: 'Critical', value: 10, color: '#ef4444' }
]

function StatCard({ title, value, change, icon: Icon, trend }) {
  const isPositive = trend === 'up'
  
  // Professional icons for different metrics
  const getIconComponent = (title) => {
    if (title.includes('Commandes')) return Zap
    if (title.includes('Valeur')) return DollarSign
    if (title.includes('Transport')) return Truck
    if (title.includes('Efficacité')) return TrendingUp
    return Package
  }

  const getAccentColor = (title) => {
    if (title.includes('Commandes')) return 'text-blue-500'
    if (title.includes('Valeur')) return 'text-green-500'
    if (title.includes('Transport')) return 'text-purple-500'
    if (title.includes('Efficacité')) return 'text-pink-500'
    return 'text-blue-500'
  }

  const IconComponent = getIconComponent(title)
  
  return (
    <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20 hover:shadow-2xl transition-all duration-300 hover:scale-105">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center ${getAccentColor(title)}`}>
            <IconComponent className="h-6 w-6" />
          </div>
          <p className="text-sm text-gray-600 font-semibold uppercase tracking-wide">{title}</p>
        </div>
      </div>
      
      <div className="mb-4">
        <p className="text-3xl font-bold text-gray-900">{value}</p>
      </div>
      
      <div className={`flex items-center text-sm font-semibold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
        <TrendingUp className={`h-4 w-4 mr-2 ${isPositive ? '' : 'rotate-180'}`} />
        <span>{change}</span>
        <span className="text-gray-500 ml-2">période précédente</span>
      </div>
    </div>
  )
}

function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [apiConnected, setApiConnected] = useState(false)

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true)
        
        // Test API connection first
        const connectionTest = await generalAPI.testConnection()
        setApiConnected(connectionTest.connected)
        
        if (connectionTest.connected) {
          // Load real data from API
          const [overviewData, kpis, alerts] = await Promise.all([
            dashboardAPI.getOverview().catch(() => null),
            dashboardAPI.getKPIs().catch(() => null),
            dashboardAPI.getAlerts().catch(() => null)
          ])

          setMetrics({
            ordersPerHour: kpis?.orders_per_hour || 65.4,
            transportEfficiency: kpis?.transport_efficiency || 92.3,
            predictiveMaintenance: kpis?.equipment_alerts || 12,
            totalRevenue: kpis?.avg_order_value * kpis?.orders_per_hour * 24 || 125690,
            alerts: alerts || []
          })
        } else {
          // Fallback to mock data if API is not available
          setMetrics({
            ordersPerHour: 65.4,
            transportEfficiency: 92.3,
            predictiveMaintenance: 12,
            totalRevenue: 125690,
            alerts: []
          })
        }
      } catch (error) {
        console.error('Error loading dashboard data:', error)
        // Fallback to mock data on error
        setMetrics({
          ordersPerHour: 65.4,
          transportEfficiency: 92.3,
          predictiveMaintenance: 12,
          totalRevenue: 125690,
          alerts: []
        })
      } finally {
        setLoading(false)
      }
    }

    loadDashboardData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-indigo-50 to-purple-50">
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-12 text-center shadow-xl border border-white/20">
          <div className="animate-spin w-12 h-12 border-4 border-indigo-200 border-t-indigo-500 rounded-full mx-auto mb-6"></div>
          <span className="text-2xl font-bold text-gray-900">
            SUPPLY CHAIN AI
          </span>
          <p className="text-sm text-gray-600 mt-3">
            Initialisation du système en cours...
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8 bg-gradient-to-br from-indigo-50 to-purple-50 min-h-screen p-6">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-3">
              Supply Chain Command Center
            </h1>
            <p className="text-lg text-gray-600">
              Intelligence artificielle avancée pour l'optimisation
            </p>
          </div>
          <div className="flex items-center gap-3 bg-white/60 backdrop-blur-sm px-6 py-3 rounded-xl border border-white/30">
            <div className={`w-3 h-3 rounded-full ${apiConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-sm font-bold text-gray-900">
              {apiConnected ? 'SYSTÈME OPÉRATIONNEL' : 'HORS LIGNE'}
            </span>
          </div>
          <div className="ml-4 flex items-center space-x-3">
            <HealthCheck />
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Commandes / Heure"
          value={metrics.ordersPerHour}
          change="+12.5%"
          icon={Package}
          trend="up"
        />
        <StatCard
          title="Valeur Moyenne Commande"
          value={`€${metrics.avgOrderValue || 1250}`}
          change="+8.2%"
          icon={DollarSign}
          trend="up"
        />
        <StatCard
          title="Efficacité Transport"
          value={`${metrics.transportEfficiency}%`}
          change="+5.1%"
          icon={Truck}
          trend="up"
        />
        <StatCard
          title="Efficacité Globale"
          value={`${metrics.overallEfficiency || 94}%`}
          change="+3.7%"
          icon={Zap}
          trend="up"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Demand Forecasting Chart */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-2xl font-bold text-gray-900">Prévision IA</h3>
              <p className="text-sm text-gray-600">Intelligence prédictive avancée</p>
            </div>
            <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-xs font-semibold">
              TEMPS RÉEL
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={demandData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
              <XAxis dataKey="date" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="#3b82f6" 
                strokeWidth={3}
                name="Réel"
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              />
              <Line 
                type="monotone" 
                dataKey="predicted" 
                stroke="#8b5cf6" 
                strokeWidth={3}
                strokeDasharray="10 10"
                name="IA Prédit"
                dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Transport Efficiency */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-2xl font-bold text-gray-900">Routes Optimisées</h3>
              <p className="text-sm text-gray-600">Optimisation multi-dimensionnelle</p>
            </div>
            <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs font-semibold">
              OPTIMISÉ
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={transportData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
              <XAxis dataKey="route" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)'
                }}
              />
              <Bar 
                dataKey="efficiency" 
                fill="url(#gradient)"
                name="Efficacité %" 
                radius={[8, 8, 0, 0]}
              />
              <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={1}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.8}/>
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Equipment Status */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900">État des Équipements</h3>
              <p className="text-sm text-gray-600">Répartition du statut de santé</p>
            </div>
            <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-xs font-semibold">
              Surveillance
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={equipmentStatus}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}%`}
                labelLine={false}
              >
                {equipmentStatus.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Alerts */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-white/20">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-xl font-bold text-gray-900">Alertes Récentes</h3>
              <p className="text-sm text-gray-600">Dernières notifications système</p>
            </div>
            <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-xs font-semibold">
              3 alertes
            </div>
          </div>
          <div className="space-y-4">
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 hover:shadow-md transition-all duration-300">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <AlertTriangle className="h-5 w-5 text-yellow-600" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">
                    Maintenance préventive requise
                  </p>
                  <p className="text-sm text-gray-500">
                    Convoyeur B - dans 2 jours
                  </p>
                </div>
                <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-semibold">Warning</span>
              </div>
            </div>
            
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 hover:shadow-md transition-all duration-300">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">
                    Pic de demande détecté
                  </p>
                  <p className="text-sm text-gray-500">
                    Produit A - augmentation 25%
                  </p>
                </div>
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-semibold">Info</span>
              </div>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded-xl p-4 hover:shadow-md transition-all duration-300">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <Truck className="h-5 w-5 text-green-600" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">
                    Route optimisée
                  </p>
                  <p className="text-sm text-gray-500">
                    Économie de 15% sur Route C
                  </p>
                </div>
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-semibold">Success</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 shadow-xl border border-white/20">
        <div className="flex items-center justify-between mb-8">
          <h3 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Centre de Commande</h3>
          <div className="text-sm text-gray-600">Interface de contrôle avancée</div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <button className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 flex items-center justify-center space-x-3">
            <TrendingUp className="h-6 w-6" />
            <span className="font-semibold">IA PRÉDICTIVE</span>
          </button>
          <button className="bg-gradient-to-r from-green-500 to-emerald-600 text-white p-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 flex items-center justify-center space-x-3">
            <Truck className="h-6 w-6" />
            <span className="font-semibold">ROUTES OPTIMISÉES</span>
          </button>
          <button className="bg-gradient-to-r from-yellow-500 to-orange-600 text-white p-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 flex items-center justify-center space-x-3">
            <Settings className="h-6 w-6" />
            <span className="font-semibold">MAINTENANCE</span>
          </button>
        </div>
      </div>
    </div>
  )
}

export default Dashboard

function HealthCheck() {
  const [checking, setChecking] = React.useState(false)
  const [status, setStatus] = React.useState(null)
  const [lastChecked, setLastChecked] = React.useState(null)

  const runCheck = async () => {
    setChecking(true)
    try {
      const res = await generalAPI.checkHealth()
      setStatus(res)
      setLastChecked(new Date().toLocaleTimeString())
    } catch (e) {
      setStatus({ error: e.message || 'Erreur' })
      setLastChecked(new Date().toLocaleTimeString())
    } finally {
      setChecking(false)
    }
  }

  return (
    <div className="flex items-center space-x-3">
      <button
        onClick={runCheck}
        className={`px-3 py-2 rounded-lg font-semibold text-sm ${checking ? 'bg-gray-300 text-gray-700' : 'bg-indigo-600 text-white hover:bg-indigo-700'}`}
        disabled={checking}
      >
        {checking ? 'Vérification...' : 'Vérifier API'}
      </button>
      <div className="text-sm text-gray-700">
        {status ? (
          <div className="flex flex-col">
            <span className="font-medium">{status.status || (status.error ? 'erreur' : 'inconnu')}</span>
            <span className="text-xs text-gray-500">Dernière vérif: {lastChecked}</span>
          </div>
        ) : (
          <span className="text-xs text-gray-500">Aucune vérif</span>
        )}
      </div>
    </div>
  )
}
