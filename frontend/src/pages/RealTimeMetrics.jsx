import React, { useState, useEffect } from 'react';
import { Activity, Zap, Users, Package, RefreshCw, BarChart3 } from 'lucide-react';
import { metricsAPI } from '../services/api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ComposedChart } from 'recharts';

const RealTimeMetrics = () => {
  const [realTimeData, setRealTimeData] = useState([]);
  const [isLive, setIsLive] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5);

  // Mock real-time data
  const generateRealTimeData = () => {
    const now = new Date();
    const data = [];
    
    for (let i = 29; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60000); // 1 minute intervals
      data.push({
        time: timestamp.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' }),
        throughput: Math.floor(Math.random() * 100) + 450,
        efficiency: Math.floor(Math.random() * 20) + 75,
        orders: Math.floor(Math.random() * 30) + 120,
        alerts: Math.floor(Math.random() * 5),
        temperature: Math.floor(Math.random() * 10) + 40,
        pressure: Math.floor(Math.random() * 20) + 110,
        powerConsumption: Math.floor(Math.random() * 50) + 200
      });
    }
    
    return data;
  };

  // Current KPIs
  const [currentKPIs, setCurrentKPIs] = useState({
    throughput: 487,
    efficiency: 92.3,
    activeOrders: 143,
    alertsCount: 2,
    avgResponseTime: 1.2,
    powerConsumption: 245,
    networkLatency: 45,
    uptimePercentage: 99.7
  });

  // System performance data
  const systemPerformanceData = [
    { name: 'CPU', value: 78, max: 100, color: '#3B82F6' },
    { name: 'Mémoire', value: 65, max: 100, color: '#10B981' },
    { name: 'Disque', value: 42, max: 100, color: '#F59E0B' },
    { name: 'Réseau', value: 88, max: 100, color: '#EF4444' }
  ];

  // Network traffic data
  const networkTrafficData = [
    { time: '14:30', incoming: 234, outgoing: 189 },
    { time: '14:35', incoming: 267, outgoing: 198 },
    { time: '14:40', incoming: 298, outgoing: 210 },
    { time: '14:45', incoming: 278, outgoing: 203 },
    { time: '14:50', incoming: 312, outgoing: 220 },
    { time: '14:55', incoming: 289, outgoing: 215 },
    { time: '15:00', incoming: 334, outgoing: 230 }
  ];

  // Activity feed data
  const [activityFeed, setActivityFeed] = useState([
    { id: 1, type: 'order', message: 'Nouvelle commande #12457 reçue', timestamp: '15:02', priority: 'normal' },
    { id: 2, type: 'alert', message: 'Alerte: Température élevée Entrepôt B', timestamp: '15:01', priority: 'high' },
    { id: 3, type: 'system', message: 'Synchronisation des données terminée', timestamp: '15:00', priority: 'low' },
    { id: 4, type: 'order', message: 'Commande #12456 expédiée', timestamp: '14:58', priority: 'normal' },
    { id: 5, type: 'maintenance', message: 'Maintenance préventive EQ003 programmée', timestamp: '14:55', priority: 'medium' }
  ]);

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const current = await metricsAPI.getCurrentMetrics()
        if (mounted && current) {
          // Backend returns a MetricsResponse shape; map fields safely
          setRealTimeData(current.recent || current.data || generateRealTimeData())
          setCurrentKPIs(prev => ({
            ...prev,
            throughput: current.orders_per_hour || (current.kpis && current.kpis.orders_processed) || prev.throughput,
            efficiency: current.overall_efficiency || (current.kpis && current.kpis.transport_efficiency) || prev.efficiency,
            activeOrders: current.active_alerts_count || prev.activeOrders,
            alertsCount: current.equipment_alerts || prev.alertsCount,
            avgResponseTime: prev.avgResponseTime
          }))
          return
        }
      } catch (err) {
        // fall back to local generator
      }
      if (mounted) setRealTimeData(generateRealTimeData())
    })()
    

    let interval;
    if (isLive) {
      interval = setInterval(() => {
        setRealTimeData(generateRealTimeData());
        
        // Update KPIs
        setCurrentKPIs(prev => ({
          ...prev,
          throughput: Math.floor(Math.random() * 100) + 450,
          efficiency: Math.floor(Math.random() * 20) + 75,
          activeOrders: Math.floor(Math.random() * 30) + 120,
          alertsCount: Math.floor(Math.random() * 5),
          avgResponseTime: (Math.random() * 2 + 0.5).toFixed(1),
          powerConsumption: Math.floor(Math.random() * 50) + 200,
          networkLatency: Math.floor(Math.random() * 20) + 35,
          uptimePercentage: (99 + Math.random()).toFixed(1)
        }));

        // Simulate new activity
        const newActivity = {
          id: Date.now(),
          type: ['order', 'alert', 'system', 'maintenance'][Math.floor(Math.random() * 4)],
          message: 'Nouvelle activité système détectée',
          timestamp: new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' }),
          priority: ['low', 'normal', 'medium', 'high'][Math.floor(Math.random() * 4)]
        };

        setActivityFeed(prev => [newActivity, ...prev.slice(0, 9)]);
      }, refreshInterval * 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isLive, refreshInterval]);

  const getActivityIcon = (type) => {
    switch (type) {
      case 'order': return <Package className="w-4 h-4 text-blue-600" />;
      case 'alert': return <Zap className="w-4 h-4 text-red-600" />;
      case 'system': return <Activity className="w-4 h-4 text-green-600" />;
      case 'maintenance': return <RefreshCw className="w-4 h-4 text-yellow-600" />;
      default: return <Activity className="w-4 h-4 text-gray-600" />;
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return 'border-l-red-500 bg-red-50';
      case 'medium': return 'border-l-yellow-500 bg-yellow-50';
      case 'low': return 'border-l-green-500 bg-green-50';
      default: return 'border-l-blue-500 bg-blue-50';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Métriques Temps Réel</h1>
          <p className="text-gray-600 mt-2">Surveillance en temps réel de vos opérations</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Actualisation:</label>
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
              className="input text-sm py-1"
            >
              <option value={1}>1s</option>
              <option value={5}>5s</option>
              <option value={10}>10s</option>
              <option value={30}>30s</option>
            </select>
          </div>
          <button
            onClick={() => setIsLive(!isLive)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border font-medium ${
              isLive 
                ? 'bg-green-100 text-green-700 border-green-300' 
                : 'bg-gray-100 text-gray-700 border-gray-300'
            }`}
          >
            <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            {isLive ? 'En Direct' : 'Arrêté'}
          </button>
        </div>
      </div>

      {/* Real-time KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-full">
              <BarChart3 className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Débit</p>
              <p className="text-2xl font-bold text-gray-900">{currentKPIs.throughput}</p>
              <p className="text-xs text-gray-500">unités/heure</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-green-100 p-3 rounded-full">
              <Activity className="w-6 h-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Efficacité</p>
              <p className="text-2xl font-bold text-gray-900">{currentKPIs.efficiency}%</p>
              <p className="text-xs text-green-600">+2.1% aujourd'hui</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-yellow-100 p-3 rounded-full">
              <Package className="w-6 h-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Commandes Actives</p>
              <p className="text-2xl font-bold text-gray-900">{currentKPIs.activeOrders}</p>
              <p className="text-xs text-blue-600">en cours</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-red-100 p-3 rounded-full">
              <Zap className="w-6 h-6 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Alertes</p>
              <p className="text-2xl font-bold text-gray-900">{currentKPIs.alertsCount}</p>
              <p className="text-xs text-red-600">nécessitent attention</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Real-time Chart */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Métriques en Temps Réel</h3>
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
              <span>Débit</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span>Efficacité</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
              <span>Commandes</span>
            </div>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={realTimeData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Area yAxisId="left" type="monotone" dataKey="throughput" fill="#3B82F6" fillOpacity={0.3} stroke="#3B82F6" strokeWidth={2} />
            <Line yAxisId="right" type="monotone" dataKey="efficiency" stroke="#10B981" strokeWidth={2} />
            <Bar yAxisId="right" dataKey="orders" fill="#F59E0B" fillOpacity={0.7} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* System Performance and Network Traffic */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Performance */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Système</h3>
          <div className="space-y-4">
            {systemPerformanceData.map((item, index) => (
              <div key={index}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-gray-700">{item.name}</span>
                  <span className="text-sm text-gray-600">{item.value}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${item.value}%`,
                      backgroundColor: item.color
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-600">Temps de Réponse</p>
              <p className="font-semibold">{currentKPIs.avgResponseTime}s</p>
            </div>
            <div>
              <p className="text-gray-600">Disponibilité</p>
              <p className="font-semibold text-green-600">{currentKPIs.uptimePercentage}%</p>
            </div>
            <div>
              <p className="text-gray-600">Consommation</p>
              <p className="font-semibold">{currentKPIs.powerConsumption}kW</p>
            </div>
            <div>
              <p className="text-gray-600">Latence Réseau</p>
              <p className="font-semibold">{currentKPIs.networkLatency}ms</p>
            </div>
          </div>
        </div>

        {/* Network Traffic */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Trafic Réseau</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={networkTrafficData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="incoming" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} name="Entrant (MB/s)" />
              <Area type="monotone" dataKey="outgoing" stackId="2" stroke="#10B981" fill="#10B981" fillOpacity={0.6} name="Sortant (MB/s)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Activity Feed and Additional Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Activity Feed */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Activité en Temps Réel</h3>
            <span className="text-sm text-gray-500">{activityFeed.length} événements</span>
          </div>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {activityFeed.map((activity) => (
              <div
                key={activity.id}
                className={`flex items-start gap-3 p-3 rounded-lg border-l-4 ${getPriorityColor(activity.priority)}`}
              >
                <div className="flex-shrink-0 mt-1">
                  {getActivityIcon(activity.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900">{activity.message}</p>
                  <p className="text-xs text-gray-500 mt-1">{activity.timestamp}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Temperature and Sensors */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Capteurs Environnementaux</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={realTimeData.slice(-10)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="temperature" stroke="#EF4444" strokeWidth={2} name="Température (°C)" />
              <Line type="monotone" dataKey="pressure" stroke="#3B82F6" strokeWidth={2} name="Pression (bar)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Real-time Alerts */}
      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold text-gray-900">Alertes Système en Temps Réel</h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center">
                <Zap className="w-5 h-5 text-red-600 mr-2" />
                <span className="font-medium text-red-800">Alertes Critiques</span>
              </div>
              <p className="text-2xl font-bold text-red-900 mt-2">1</p>
              <p className="text-sm text-red-600">Température élevée détectée</p>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-center">
                <Activity className="w-5 h-5 text-yellow-600 mr-2" />
                <span className="font-medium text-yellow-800">Avertissements</span>
              </div>
              <p className="text-2xl font-bold text-yellow-900 mt-2">3</p>
              <p className="text-sm text-yellow-600">Performance dégradée</p>
            </div>

            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center">
                <Users className="w-5 h-5 text-green-600 mr-2" />
                <span className="font-medium text-green-800">Système Sain</span>
              </div>
              <p className="text-2xl font-bold text-green-900 mt-2">12</p>
              <p className="text-sm text-green-600">Services opérationnels</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeMetrics;
