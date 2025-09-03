import React, { useState, useEffect } from 'react';
import { maintenanceAPI } from '../services/api';
import { Wrench, AlertTriangle, CheckCircle, Clock, TrendingUp, Settings } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell } from 'recharts';

const PredictiveMaintenance = () => {
  const [equipmentData, setEquipmentData] = useState([]);
  const [maintenanceAlerts, setMaintenanceAlerts] = useState([]);
  const [selectedEquipment, setSelectedEquipment] = useState('');

  // Mock data for demonstration
  const equipmentList = [
    { 
      id: 'EQ001', 
      name: 'Convoyeur Principal', 
      status: 'good', 
      healthScore: 85, 
      nextMaintenance: '2024-02-15',
      lastMaintenance: '2024-01-01',
      efficiency: 92,
      temperature: 45,
      vibration: 2.3,
      pressure: 120
    },
    { 
      id: 'EQ002', 
      name: 'Robot de Tri A', 
      status: 'warning', 
      healthScore: 68, 
      nextMaintenance: '2024-01-20',
      lastMaintenance: '2023-12-15',
      efficiency: 78,
      temperature: 62,
      vibration: 4.1,
      pressure: 95
    },
    { 
      id: 'EQ003', 
      name: 'Système de Refroidissement', 
      status: 'critical', 
      healthScore: 45, 
      nextMaintenance: '2024-01-18',
      lastMaintenance: '2023-11-28',
      efficiency: 65,
      temperature: 78,
      vibration: 6.2,
      pressure: 85
    },
    { 
      id: 'EQ004', 
      name: 'Chariot Élévateur 1', 
      status: 'good', 
      healthScore: 91, 
      nextMaintenance: '2024-03-01',
      lastMaintenance: '2024-01-10',
      efficiency: 95,
      temperature: 38,
      vibration: 1.8,
      pressure: 140
    }
  ];

  const healthTrendData = [
      { date: '05/01', EQ001: 87, EQ002: 72, EQ003: 55, EQ004: 90 },
      { date: '10/01', EQ001: 86, EQ002: 70, EQ003: 50, EQ004: 91 },
      { date: '15/01', EQ001: 85, EQ002: 68, EQ003: 45, EQ004: 91 },
      { date: '20/01', EQ001: 85, EQ002: 68, EQ003: 45, EQ004: 91 }
    ];

  const maintenanceCostsData = [
    { month: 'Oct', preventive: 12000, corrective: 8500, total: 20500 },
    { month: 'Nov', preventive: 15000, corrective: 6200, total: 21200 },
    { month: 'Déc', preventive: 18000, corrective: 4800, total: 22800 },
    { month: 'Jan', preventive: 22000, corrective: 3200, total: 25200 }
  ];

  const alertsData = [
    { 
      id: 1, 
      equipment: 'EQ003', 
      name: 'Système de Refroidissement',
      type: 'critical', 
      message: 'Température critique détectée',
      priority: 'high',
      estimatedFailure: '2024-01-18',
      recommendedAction: 'Maintenance immédiate requise'
    },
    { 
      id: 2, 
      equipment: 'EQ002', 
      name: 'Robot de Tri A',
      type: 'warning', 
      message: 'Vibrations anormales détectées',
      priority: 'medium',
      estimatedFailure: '2024-01-25',
      recommendedAction: 'Inspection programmée recommandée'
    },
    { 
      id: 3, 
      equipment: 'EQ001', 
      name: 'Convoyeur Principal',
      type: 'info', 
      message: 'Maintenance préventive due',
      priority: 'low',
      estimatedFailure: '2024-02-15',
      recommendedAction: 'Planifier maintenance de routine'
    }
  ];

  const sensorData = [
    { time: '08:00', temperature: 45, vibration: 2.1, pressure: 118 },
    { time: '10:00', temperature: 47, vibration: 2.3, pressure: 120 },
    { time: '12:00', temperature: 49, vibration: 2.5, pressure: 122 },
    { time: '14:00', temperature: 52, vibration: 2.8, pressure: 119 },
    { time: '16:00', temperature: 48, vibration: 2.4, pressure: 121 },
    { time: '18:00', temperature: 46, vibration: 2.2, pressure: 120 }
  ];

  const downtimeData = [
    { name: 'Planifiée', value: 15, color: '#10B981' },
    { name: 'Non planifiée', value: 8, color: '#EF4444' },
    { name: 'Urgence', value: 3, color: '#F59E0B' },
    { name: 'Préventive', value: 12, color: '#3B82F6' }
  ];

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const equipment = await maintenanceAPI.getEquipmentHealth()
        if (mounted && equipment) {
          // Backend returns { equipment: [...], total: N }
          setEquipmentData(equipment.equipment || equipment.items || equipmentList)
          // Backend does not return alerts from this endpoint; keep fallback
          setMaintenanceAlerts(equipment.alerts || alertsData)
          return
        }
      } catch (err) {
        // fall back to mock
      }

      if (mounted) {
        setEquipmentData(equipmentList)
        setMaintenanceAlerts(alertsData)
      }
    })()

    return () => { mounted = false }
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'good': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getHealthScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Maintenance Prédictive</h1>
          <p className="text-gray-600 mt-2">Surveillez l'état de vos équipements en temps réel</p>
        </div>
        <button className="btn btn-primary flex items-center gap-2">
          <Settings className="w-4 h-4" />
          Configurer Alertes
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-green-100 p-3 rounded-full">
              <CheckCircle className="w-6 h-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Équipements Opérationnels</p>
              <p className="text-2xl font-bold text-gray-900">
                {equipmentData.filter(eq => eq.status === 'good').length}/{equipmentData.length}
              </p>
              <p className="text-xs text-green-600">75% du parc</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-yellow-100 p-3 rounded-full">
              <AlertTriangle className="w-6 h-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Alertes Actives</p>
              <p className="text-2xl font-bold text-gray-900">{maintenanceAlerts.length}</p>
              <p className="text-xs text-yellow-600">1 critique</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-full">
              <Clock className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Temps d'Arrêt Évité</p>
              <p className="text-2xl font-bold text-gray-900">168h</p>
              <p className="text-xs text-green-600">-15% ce mois</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-purple-100 p-3 rounded-full">
              <TrendingUp className="w-6 h-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Économies Réalisées</p>
              <p className="text-2xl font-bold text-gray-900">45,600€</p>
              <p className="text-xs text-green-600">+23% vs prévu</p>
            </div>
          </div>
        </div>
      </div>

      {/* Equipment Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {equipmentData.map((equipment) => (
          <div key={equipment.id} className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <Wrench className="w-5 h-5 text-gray-400 mr-2" />
                <span className="font-medium text-gray-900">{equipment.name}</span>
              </div>
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(equipment.status)}`}>
                {equipment.status === 'good' ? 'Bon' : 
                 equipment.status === 'warning' ? 'Attention' : 'Critique'}
              </span>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Score de Santé</span>
                <span className={`font-bold ${getHealthScoreColor(equipment.healthScore)}`}>
                  {equipment.healthScore}%
                </span>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    equipment.healthScore >= 80 ? 'bg-green-500' :
                    equipment.healthScore >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${equipment.healthScore}%` }}
                />
                <div className="flex justify-between mt-1">
                  <span>Vibration: {equipment.vibration}</span>
                  <span>Pression: {equipment.pressure} bar</span>
                </div>
              </div>
              
              <div className="pt-2 border-t text-xs text-gray-500">
                Prochaine maintenance: {new Date(equipment.nextMaintenance).toLocaleDateString('fr-FR')}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Health Trend Chart */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Évolution de la Santé des Équipements</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={healthTrendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="EQ001" stroke="#10B981" strokeWidth={2} name="Convoyeur Principal" />
              <Line type="monotone" dataKey="EQ002" stroke="#F59E0B" strokeWidth={2} name="Robot de Tri A" />
              <Line type="monotone" dataKey="EQ003" stroke="#EF4444" strokeWidth={2} name="Système Refroidissement" />
              <Line type="monotone" dataKey="EQ004" stroke="#3B82F6" strokeWidth={2} name="Chariot Élévateur 1" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Maintenance Costs */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Coûts de Maintenance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={maintenanceCostsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="preventive" fill="#10B981" name="Préventive" />
              <Bar dataKey="corrective" fill="#EF4444" name="Corrective" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sensor Data and Downtime */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-time Sensor Data */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Données Capteurs en Temps Réel</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={sensorData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="temperature" stroke="#EF4444" strokeWidth={2} name="Température (°C)" />
              <Line type="monotone" dataKey="vibration" stroke="#F59E0B" strokeWidth={2} name="Vibration" />
              <Line type="monotone" dataKey="pressure" stroke="#3B82F6" strokeWidth={2} name="Pression (bar)" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Downtime Analysis */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Analyse des Temps d'Arrêt</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={downtimeData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {downtimeData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Alerts Table */}
      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold text-gray-900">Alertes de Maintenance</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Équipement</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Message</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Priorité</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Échéance Estimée</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action Recommandée</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {maintenanceAlerts.map((alert) => (
                <tr key={alert.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div>
                      <div className="text-sm font-medium text-gray-900">{alert.name}</div>
                      <div className="text-sm text-gray-500">{alert.equipment}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900">{alert.message}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getPriorityColor(alert.priority)}`}>
                      {alert.priority === 'high' ? 'Élevée' : 
                       alert.priority === 'medium' ? 'Moyenne' : 'Faible'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {new Date(alert.estimatedFailure).toLocaleDateString('fr-FR')}
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900">{alert.recommendedAction}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PredictiveMaintenance;
