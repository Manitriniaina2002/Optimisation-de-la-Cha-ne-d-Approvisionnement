import React, { useState, useEffect } from 'react';
import { Truck, MapPin, Calculator, Clock, DollarSign, Navigation } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { transportAPI } from '../services/api';

const TransportOptimization = () => {
  const [optimizationData, setOptimizationData] = useState({
    routes: [],
    costs: { current: 0, optimized: 0, savings: 0 },
    performance: { onTime: 0, totalDeliveries: 0, efficiency: 0 }
  });
  
  const [formData, setFormData] = useState({
    startLocation: '',
    endLocation: '',
    vehicleType: 'truck',
    maxCapacity: 1000,
    priorityLevel: 'standard'
  });

  // Mock data for demonstration
  const routesData = [
    { id: 1, from: 'Entrepôt A', to: 'Client 1', distance: 120, cost: 450, time: 2.5, status: 'optimal' },
    { id: 2, from: 'Entrepôt A', to: 'Client 2', distance: 85, cost: 320, time: 1.8, status: 'optimal' },
    { id: 3, from: 'Entrepôt B', to: 'Client 3', distance: 200, cost: 750, time: 4.2, status: 'alternative' },
    { id: 4, from: 'Entrepôt B', to: 'Client 4', distance: 150, cost: 580, time: 3.1, status: 'optimal' }
  ];

  const costComparisonData = [
    { name: 'Lun', current: 2400, optimized: 1800 },
    { name: 'Mar', current: 3200, optimized: 2400 },
    { name: 'Mer', current: 2800, optimized: 2100 },
    { name: 'Jeu', current: 3500, optimized: 2600 },
    { name: 'Ven', current: 4100, optimized: 3000 },
    { name: 'Sam', current: 1800, optimized: 1400 },
    { name: 'Dim', current: 1200, optimized: 900 }
  ];

  const vehicleDistributionData = [
    { name: 'Camions', value: 35, color: '#3B82F6' },
    { name: 'Fourgons', value: 25, color: '#10B981' },
    { name: 'Motos', value: 20, color: '#F59E0B' },
    { name: 'Drones', value: 20, color: '#EF4444' }
  ];

  const performanceData = [
    { day: 'Lun', deliveries: 45, onTime: 42 },
    { day: 'Mar', deliveries: 52, onTime: 48 },
    { day: 'Mer', deliveries: 38, onTime: 36 },
    { day: 'Jeu', deliveries: 61, onTime: 55 },
    { day: 'Ven', deliveries: 48, onTime: 44 },
    { day: 'Sam', deliveries: 35, onTime: 33 },
    { day: 'Dim', deliveries: 28, onTime: 27 }
  ];

  useEffect(() => {
    // Try fetching real analytics from API, otherwise fall back to mock data
    let mounted = true
    ;(async () => {
      try {
        const data = await transportAPI.getAnalytics('week')
        if (mounted && data) {
          // If backend returned KPIs, map plausible fields into the optimization data
          const mappedCosts = data.summary || data.costs || { current: 18500, optimized: 13800, savings: 4700 }
          const mappedPerformance = data.performance || {
            onTime: data.kpis ? data.kpis.delivery_performance_percent : 285,
            totalDeliveries: data.kpis ? data.kpis.orders_processed : 307,
            efficiency: data.kpis ? data.kpis.transport_efficiency : 92.8
          }
          setOptimizationData(prev => ({
            ...prev,
            routes: data.routes || routesData,
            costs: mappedCosts,
            performance: mappedPerformance
          }))
          return
        }
      } catch (err) {
        // ignore and fall back
      }

      if (mounted) {
        setOptimizationData({
          routes: routesData,
          costs: { current: 18500, optimized: 13800, savings: 4700 },
          performance: { onTime: 285, totalDeliveries: 307, efficiency: 92.8 }
        })
      }
    })()
    return () => { mounted = false }
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleOptimizeRoute = () => {
    // Call backend optimization endpoint
    (async () => {
      try {
        const payload = {
          delivery_points: [{ start: formData.startLocation, end: formData.endLocation }],
          vehicles: [{ type: formData.vehicleType, capacity: formData.maxCapacity }],
          optimization_objectives: { priority: formData.priorityLevel }
        }
        const result = await transportAPI.optimizeRoutes(payload.delivery_points, payload.vehicles, payload.optimization_objectives)
        if (result) {
          // try to apply returned shape
          setOptimizationData({
            routes: result.routes || routesData,
            costs: result.costs || optimizationData.costs,
            performance: result.performance || optimizationData.performance
          })
        }
      } catch (err) {
        console.error('Optimization API failed, using local simulation', err)
      }
    })()
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'optimal': return 'bg-green-100 text-green-800';
      case 'alternative': return 'bg-yellow-100 text-yellow-800';
      case 'inefficient': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Optimisation du Transport</h1>
          <p className="text-gray-600 mt-2">Optimisez vos routes et réduisez vos coûts de transport</p>
        </div>
        <button 
          onClick={handleOptimizeRoute}
          className="btn btn-primary flex items-center gap-2"
        >
          <Calculator className="w-4 h-4" />
          Optimiser les Routes
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-full">
              <DollarSign className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Économies Réalisées</p>
              <p className="text-2xl font-bold text-gray-900">{optimizationData.costs.savings.toLocaleString()}€</p>
              <p className="text-xs text-green-600">+25% ce mois</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-green-100 p-3 rounded-full">
              <Clock className="w-6 h-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Taux de Ponctualité</p>
              <p className="text-2xl font-bold text-gray-900">{optimizationData.performance.efficiency}%</p>
              <p className="text-xs text-green-600">+3.2% cette semaine</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-yellow-100 p-3 rounded-full">
              <Truck className="w-6 h-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Livraisons</p>
              <p className="text-2xl font-bold text-gray-900">{optimizationData.performance.totalDeliveries}</p>
              <p className="text-xs text-blue-600">Cette semaine</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-purple-100 p-3 rounded-full">
              <Navigation className="w-6 h-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Routes Optimisées</p>
              <p className="text-2xl font-bold text-gray-900">156</p>
              <p className="text-xs text-green-600">+12 aujourd'hui</p>
            </div>
          </div>
        </div>
      </div>

      {/* Route Configuration Form */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Nouvelle Optimisation de Route</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Point de Départ</label>
            <input
              type="text"
              name="startLocation"
              value={formData.startLocation}
              onChange={handleInputChange}
              placeholder="Entrepôt A"
              className="input"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Destination</label>
            <input
              type="text"
              name="endLocation"
              value={formData.endLocation}
              onChange={handleInputChange}
              placeholder="Zone de livraison"
              className="input"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Type de Véhicule</label>
            <select
              name="vehicleType"
              value={formData.vehicleType}
              onChange={handleInputChange}
              className="input"
            >
              <option value="truck">Camion</option>
              <option value="van">Fourgon</option>
              <option value="motorcycle">Moto</option>
              <option value="drone">Drone</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Capacité Max (kg)</label>
            <input
              type="number"
              name="maxCapacity"
              value={formData.maxCapacity}
              onChange={handleInputChange}
              className="input"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Priorité</label>
            <select
              name="priorityLevel"
              value={formData.priorityLevel}
              onChange={handleInputChange}
              className="input"
            >
              <option value="urgent">Urgent</option>
              <option value="standard">Standard</option>
              <option value="economy">Économique</option>
            </select>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost Comparison Chart */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Comparaison des Coûts</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={costComparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="current" fill="#EF4444" name="Coût Actuel" />
              <Bar dataKey="optimized" fill="#10B981" name="Coût Optimisé" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Vehicle Distribution */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Répartition des Véhicules</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={vehicleDistributionData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {vehicleDistributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance des Livraisons</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="deliveries" stroke="#3B82F6" name="Total Livraisons" strokeWidth={2} />
            <Line type="monotone" dataKey="onTime" stroke="#10B981" name="À l'heure" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Routes Table */}
      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold text-gray-900">Routes Optimisées</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Route</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Distance</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Coût</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Temps</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Statut</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {routesData.map((route) => (
                <tr key={route.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <MapPin className="w-4 h-4 text-gray-400 mr-2" />
                      <div>
                        <div className="text-sm font-medium text-gray-900">{route.from}</div>
                        <div className="text-sm text-gray-500">→ {route.to}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{route.distance} km</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{route.cost}€</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{route.time}h</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(route.status)}`}>
                      {route.status === 'optimal' ? 'Optimal' : 
                       route.status === 'alternative' ? 'Alternative' : 'Inefficace'}
                    </span>
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

export default TransportOptimization;
