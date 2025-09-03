import React, { useState, useEffect } from 'react'
import { dashboardAPI } from '../services/api'
import { Link } from 'react-router-dom'
import { 
  TrendingUp, 
  Truck, 
  Settings, 
  AlertTriangle,
  DollarSign,
  Package,
  Users,
  Zap,
  Activity,
  Play,
  ArrowRight,
  Sparkles,
  BarChart3,
  Bell
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

function StatCard({ title, value, change, icon: Icon, trend, accentColor = "from-blue-500 to-cyan-500" }) {
  const isPositive = trend === 'up'
  
  return (
    <div className="group relative overflow-hidden bg-white/90 backdrop-blur-sm rounded-3xl p-6 border border-gray-200/50 shadow-xl hover:shadow-2xl transition-all duration-500 hover:scale-[1.02] hover:-translate-y-1">
      {/* Subtle gradient background */}
      <div className={`absolute inset-0 bg-gradient-to-br ${accentColor} opacity-5 group-hover:opacity-10 transition-opacity duration-500`}></div>
      
      {/* Floating particles effect */}
      <div className="absolute top-2 right-2 w-12 h-12 bg-gradient-to-br from-gray-100/50 to-gray-200/30 rounded-full blur-md opacity-0 group-hover:opacity-100 transition-all duration-700 group-hover:animate-pulse"></div>
      <div className="absolute bottom-4 left-4 w-8 h-8 bg-gradient-to-br from-gray-100/30 to-gray-200/20 rounded-full blur-sm opacity-0 group-hover:opacity-100 transition-all duration-500 delay-200"></div>
      
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-6">
          <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${accentColor} p-3 shadow-lg group-hover:shadow-xl transition-all duration-300 group-hover:rotate-3`}>
            <Icon className="h-8 w-8 text-white" />
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-500 uppercase tracking-wide font-bold">{title}</p>
          </div>
        </div>
        
        <div className="mb-4">
          <p className="text-4xl font-black text-gray-900 bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
            {value}
          </p>
        </div>
        
        <div className={`flex items-center text-sm font-bold ${isPositive ? 'text-emerald-600' : 'text-red-500'}`}>
          <div className={`flex items-center justify-center w-6 h-6 rounded-full ${isPositive ? 'bg-emerald-100' : 'bg-red-100'} mr-2`}>
            <TrendingUp className={`h-3 w-3 ${isPositive ? '' : 'rotate-180'}`} />
          </div>
          <span>{change}</span>
          <span className="text-gray-500 ml-2 font-medium">vs dernier</span>
        </div>
      </div>
    </div>
  )
}

function Dashboard() {
  const [metrics, setMetrics] = useState({
    ordersPerHour: 65.4,
    avgOrderValue: 1250,
    transportEfficiency: 92.3,
    overallEfficiency: 94,
    predictiveMaintenance: 12,
    totalRevenue: 125690,
    alerts: []
  })
  const [loading, setLoading] = useState(false)
  const [apiConnected, setApiConnected] = useState(true)

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const kpis = await dashboardAPI.getKPIs('today')
        if (!mounted) return
        if (kpis) {
          setMetrics(prev => ({
            ...prev,
            // Accept either camelCase or snake_case keys from different backends
            ordersPerHour: kpis.ordersPerHour ?? kpis.orders_per_hour ?? prev.ordersPerHour,
            avgOrderValue: kpis.avgOrderValue ?? kpis.avg_order_value ?? prev.avgOrderValue,
            transportEfficiency: kpis.transportEfficiency ?? kpis.transport_efficiency ?? prev.transportEfficiency,
            overallEfficiency: kpis.overallEfficiency ?? kpis.overall_efficiency ?? prev.overallEfficiency,
            totalRevenue: kpis.totalRevenue ?? kpis.total_revenue ?? prev.totalRevenue,
            predictiveMaintenance: kpis.predictiveMaintenance ?? kpis.predictive_maintenance ?? prev.predictiveMaintenance,
            alerts: kpis.alerts ?? kpis.alerts_list ?? prev.alerts
          }))
        }
      } catch (err) {
        console.warn('Dashboard API unavailable, using mock metrics', err)
        setApiConnected(false)
      }
    })()
    return () => { mounted = false }
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-cyan-50">
        <div className="bg-white/80 backdrop-blur-xl rounded-3xl p-12 text-center shadow-2xl border border-gray-200">
          <div className="relative mb-6">
            <div className="animate-spin w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full mx-auto"></div>
            <div className="absolute inset-0 w-16 h-16 border-4 border-purple-200 border-t-purple-600 rounded-full mx-auto animate-spin" style={{animationDirection: 'reverse', animationDuration: '3s'}}></div>
          </div>
          <span className="text-3xl font-black text-gray-900 mb-2 block bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            SUPPLY CHAIN AI
          </span>
          <p className="text-gray-600 text-sm">
            Initialisation du système neuronal...
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8 p-6">
      {/* Enhanced KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <StatCard
            title="Flux Commandes"
            value={`${metrics.ordersPerHour}/h`}
            change="+12.5%"
            icon={Zap}
            trend="up"
            accentColor="from-purple-500 to-pink-500"
          />
          <StatCard
            title="Valeur Moyenne"
            value={`€${metrics.avgOrderValue}`}
            change="+8.2%"
            icon={DollarSign}
            trend="up"
            accentColor="from-emerald-500 to-teal-500"
          />
          <StatCard
            title="Efficacité Transport"
            value={`${metrics.transportEfficiency}%`}
            change="+5.1%"
            icon={Truck}
            trend="up"
            accentColor="from-blue-500 to-cyan-500"
          />
          <StatCard
            title="Performance Globale"
            value={`${metrics.overallEfficiency}%`}
            change="+3.7%"
            icon={BarChart3}
            trend="up"
            accentColor="from-orange-500 to-red-500"
          />
        </div>

        {/* Modern Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* AI Prediction Chart */}
          <div className="group relative overflow-hidden bg-white/70 backdrop-blur-xl rounded-3xl p-8 border border-gray-200/50 shadow-2xl hover:shadow-3xl transition-all duration-500">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
                    <Sparkles className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-gray-800">Intelligence Prédictive</h3>
                    <p className="text-gray-600 text-sm">Prédictions par réseau neuronal</p>
                  </div>
                </div>
                <div className="bg-purple-100 text-purple-700 px-4 py-2 rounded-full text-xs font-bold border border-purple-200">
                  IA EN DIRECT
                </div>
              </div>
              
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={demandData}>
                  <defs>
                    <linearGradient id="actualGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
                    </linearGradient>
                    <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '16px',
                      border: '1px solid rgba(226, 232, 240, 0.8)',
                      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.15)'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#8b5cf6" 
                    fill="url(#actualGradient)"
                    strokeWidth={3}
                    name="Données Réelles"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#06b6d4" 
                    strokeWidth={3}
                    strokeDasharray="10 10"
                    name="Prédiction IA"
                    dot={{ fill: '#06b6d4', strokeWidth: 2, r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Transport Analytics */}
          <div className="group relative overflow-hidden bg-white/70 backdrop-blur-xl rounded-3xl p-8 border border-gray-200/50 shadow-2xl hover:shadow-3xl transition-all duration-500">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-teal-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center space-x-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-xl flex items-center justify-center shadow-lg">
                    <Truck className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-gray-800">Optimisation Routes</h3>
                    <p className="text-gray-600 text-sm">Algorithm-Driven Logistics</p>
                  </div>
                </div>
                <div className="bg-emerald-100 text-emerald-700 px-4 py-2 rounded-full text-xs font-bold border border-emerald-200">
                  OPTIMISÉ
                </div>
              </div>
              
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={transportData}>
                  <defs>
                    <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={1}/>
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.8}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="route" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '16px',
                      border: '1px solid rgba(226, 232, 240, 0.8)',
                      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.15)'
                    }}
                  />
                  <Bar 
                    dataKey="efficiency" 
                    fill="url(#barGradient)"
                    name="Efficacité %" 
                    radius={[12, 12, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Status Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Equipment Status */}
          <div className="bg-white/70 backdrop-blur-xl rounded-3xl p-8 border border-gray-200/50 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-xl flex items-center justify-center shadow-lg">
                  <Settings className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-800">État Équipements</h3>
                  <p className="text-gray-600 text-sm">Surveillance IoT</p>
                </div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={250}>
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
                    border: '1px solid rgba(226, 232, 240, 0.8)',
                    boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.15)'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Real-time Alerts */}
          <div className="bg-white/70 backdrop-blur-xl rounded-3xl p-8 border border-gray-200/50 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
                  <Bell className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-800">Alertes Système</h3>
                  <p className="text-gray-600 text-sm">Notifications temps réel</p>
                </div>
              </div>
              <div className="bg-red-100 text-red-700 px-3 py-1 rounded-full text-xs font-bold border border-red-200">
                3 ACTIVES
              </div>
            </div>
            <div className="space-y-4">
              <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-2xl p-4 transition-all duration-300 hover:scale-[1.02] hover:shadow-lg">
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                    <AlertTriangle className="h-5 w-5 text-yellow-600" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-bold text-yellow-800">
                      Maintenance préventive requise
                    </p>
                    <p className="text-xs text-yellow-600">
                      Convoyeur B - dans 2 jours
                    </p>
                  </div>
                  <span className="bg-yellow-200 text-yellow-800 px-2 py-1 rounded-full text-xs font-bold">WARN</span>
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 rounded-2xl p-4 transition-all duration-300 hover:scale-[1.02] hover:shadow-lg">
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                    <TrendingUp className="h-5 w-5 text-blue-600" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-bold text-blue-800">
                      Pic de demande détecté
                    </p>
                    <p className="text-xs text-blue-600">
                      Produit A - augmentation 25%
                    </p>
                  </div>
                  <span className="bg-blue-200 text-blue-800 px-2 py-1 rounded-full text-xs font-bold">INFO</span>
                </div>
              </div>
              
              <div className="bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 rounded-2xl p-4 transition-all duration-300 hover:scale-[1.02] hover:shadow-lg">
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-emerald-100 rounded-lg flex items-center justify-center">
                    <Truck className="h-5 w-5 text-emerald-600" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-bold text-emerald-800">
                      Route optimisée avec succès
                    </p>
                    <p className="text-xs text-emerald-600">
                      Économie de 15% sur Route C
                    </p>
                  </div>
                  <span className="bg-emerald-200 text-emerald-800 px-2 py-1 rounded-full text-xs font-bold">OK</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Command Center */}
        <div className="relative overflow-hidden bg-white/60 backdrop-blur-xl rounded-3xl p-8 border border-gray-200/50 shadow-2xl">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-transparent to-purple-500/5"></div>
          
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h3 className="text-3xl font-black bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Centre de Commande Neural
                </h3>
                <p className="text-gray-600 text-lg">Interface de contrôle avancée • Gestion autonome</p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Link to="/demand" className="group relative overflow-hidden bg-gradient-to-br from-blue-500 to-blue-600 text-white p-6 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1">
                <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10 flex items-center justify-center space-x-3">
                  <TrendingUp className="h-6 w-6" />
                  <span className="font-bold text-lg">IA PRÉDICTIVE</span>
                </div>
                <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-all duration-300">
                  <ArrowRight className="w-4 h-4" />
                </div>
              </Link>
              
              <Link to="/transport" className="group relative overflow-hidden bg-gradient-to-br from-emerald-500 to-emerald-600 text-white p-6 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1">
                <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10 flex items-center justify-center space-x-3">
                  <Truck className="h-6 w-6" />
                  <span className="font-bold text-lg">ROUTES AUTO</span>
                </div>
                <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-all duration-300">
                  <ArrowRight className="w-4 h-4" />
                </div>
              </Link>
              
              <Link to="/maintenance" className="group relative overflow-hidden bg-gradient-to-br from-purple-500 to-purple-600 text-white p-6 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] hover:-translate-y-1">
                <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10 flex items-center justify-center space-x-3">
                  <Settings className="h-6 w-6" />
                  <span className="font-bold text-lg">MAINTENANCE</span>
                </div>
                <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-all duration-300">
                  <ArrowRight className="w-4 h-4" />
                </div>
              </Link>
            </div>
          </div>
        </div>
    </div>
  )
}

export default Dashboard