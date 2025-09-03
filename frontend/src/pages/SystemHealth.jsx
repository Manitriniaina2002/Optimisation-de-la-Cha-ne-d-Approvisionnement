import React, { useState, useEffect } from 'react';
import { Server, Database, Wifi, Shield, AlertCircle, CheckCircle, Clock, Cpu } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const SystemHealth = () => {
  const [systemStatus, setSystemStatus] = useState('healthy');
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // System components status
  const [components, setComponents] = useState([
    {
      id: 'api-gateway',
      name: 'API Gateway',
      status: 'healthy',
      uptime: 99.8,
      responseTime: 45,
      requestsPerMin: 1250,
      errors: 2,
      lastCheck: new Date()
    },
    {
      id: 'database',
      name: 'Base de Données',
      status: 'healthy',
      uptime: 99.9,
      responseTime: 12,
      requestsPerMin: 890,
      errors: 0,
      lastCheck: new Date()
    },
    {
      id: 'cache-redis',
      name: 'Cache Redis',
      status: 'warning',
      uptime: 98.5,
      responseTime: 8,
      requestsPerMin: 2100,
      errors: 15,
      lastCheck: new Date()
    },
    {
      id: 'message-queue',
      name: 'File de Messages',
      status: 'healthy',
      uptime: 99.7,
      responseTime: 25,
      requestsPerMin: 750,
      errors: 1,
      lastCheck: new Date()
    },
    {
      id: 'auth-service',
      name: 'Service d\'Authentification',
      status: 'critical',
      uptime: 97.2,
      responseTime: 150,
      requestsPerMin: 340,
      errors: 28,
      lastCheck: new Date()
    },
    {
      id: 'file-storage',
      name: 'Stockage de Fichiers',
      status: 'healthy',
      uptime: 99.6,
      responseTime: 85,
      requestsPerMin: 180,
      errors: 3,
      lastCheck: new Date()
    }
  ]);

  // Performance metrics over time
  const performanceData = [
    { time: '09:00', cpu: 45, memory: 62, network: 34, disk: 28 },
    { time: '10:00', cpu: 52, memory: 68, network: 41, disk: 32 },
    { time: '11:00', cpu: 48, memory: 65, network: 38, disk: 29 },
    { time: '12:00', cpu: 58, memory: 72, network: 45, disk: 35 },
    { time: '13:00', cpu: 62, memory: 75, network: 48, disk: 38 },
    { time: '14:00', cpu: 55, memory: 70, network: 42, disk: 33 },
    { time: '15:00', cpu: 49, memory: 67, network: 39, disk: 31 }
  ];

  // Error distribution
  const errorData = [
    { name: '4xx Client', value: 45, color: '#F59E0B' },
    { name: '5xx Server', value: 23, color: '#EF4444' },
    { name: 'Timeout', value: 18, color: '#8B5CF6' },
    { name: 'Network', value: 14, color: '#3B82F6' }
  ];

  // Uptime data
  const uptimeData = [
    { date: '01/01', uptime: 99.8 },
    { date: '02/01', uptime: 99.5 },
    { date: '03/01', uptime: 99.9 },
    { date: '04/01', uptime: 98.2 },
    { date: '05/01', uptime: 99.7 },
    { date: '06/01', uptime: 99.6 },
    { date: '07/01', uptime: 99.4 }
  ];

  // Security events
  const [securityEvents, setSecurityEvents] = useState([
    {
      id: 1,
      type: 'login_failure',
      severity: 'medium',
      message: 'Tentatives de connexion échouées multiples',
      source: '192.168.1.100',
      timestamp: new Date(Date.now() - 300000),
      status: 'investigating'
    },
    {
      id: 2,
      type: 'suspicious_traffic',
      severity: 'high',
      message: 'Trafic suspect détecté',
      source: '10.0.0.25',
      timestamp: new Date(Date.now() - 600000),
      status: 'resolved'
    },
    {
      id: 3,
      type: 'rate_limit',
      severity: 'low',
      message: 'Limite de taux dépassée',
      source: '172.16.0.5',
      timestamp: new Date(Date.now() - 900000),
      status: 'monitoring'
    }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
      
      // Simulate status updates
      setComponents(prev => prev.map(comp => ({
        ...comp,
        responseTime: comp.responseTime + Math.floor(Math.random() * 10 - 5),
        requestsPerMin: comp.requestsPerMin + Math.floor(Math.random() * 100 - 50),
        lastCheck: new Date()
      })));
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'warning': return <AlertCircle className="w-5 h-5 text-yellow-600" />;
      case 'critical': return <AlertCircle className="w-5 h-5 text-red-600" />;
      default: return <Clock className="w-5 h-5 text-gray-600" />;
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getOverallHealth = () => {
    const healthyCount = components.filter(c => c.status === 'healthy').length;
    const warningCount = components.filter(c => c.status === 'warning').length;
    const criticalCount = components.filter(c => c.status === 'critical').length;

    if (criticalCount > 0) return 'critical';
    if (warningCount > 0) return 'warning';
    return 'healthy';
  };

  return (
    <Layout>
      <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Santé du Système</h1>
          <p className="text-gray-600 mt-2">Surveillance complète de l'infrastructure</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              getOverallHealth() === 'healthy' ? 'bg-green-500' :
              getOverallHealth() === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
            <span className="text-sm font-medium">
              Statut: {getOverallHealth() === 'healthy' ? 'Sain' : 
                      getOverallHealth() === 'warning' ? 'Attention' : 'Critique'}
            </span>
          </div>
          <span className="text-sm text-gray-500">
            Dernière mise à jour: {lastUpdate.toLocaleTimeString('fr-FR')}
          </span>
        </div>
      </div>

      {/* Overall System Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-full">
              <Server className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Services Actifs</p>
              <p className="text-2xl font-bold text-gray-900">
                {components.filter(c => c.status !== 'critical').length}/{components.length}
              </p>
              <p className="text-xs text-green-600">Opérationnels</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-green-100 p-3 rounded-full">
              <CheckCircle className="w-6 h-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Disponibilité</p>
              <p className="text-2xl font-bold text-gray-900">
                {(components.reduce((acc, c) => acc + c.uptime, 0) / components.length).toFixed(1)}%
              </p>
              <p className="text-xs text-green-600">Moyenne 7j</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-yellow-100 p-3 rounded-full">
              <Cpu className="w-6 h-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Charge CPU</p>
              <p className="text-2xl font-bold text-gray-900">
                {performanceData[performanceData.length - 1]?.cpu}%
              </p>
              <p className="text-xs text-blue-600">Temps réel</p>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="bg-red-100 p-3 rounded-full">
              <AlertCircle className="w-6 h-6 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Incidents</p>
              <p className="text-2xl font-bold text-gray-900">
                {components.reduce((acc, c) => acc + c.errors, 0)}
              </p>
              <p className="text-xs text-red-600">Dernière heure</p>
            </div>
          </div>
        </div>
      </div>

      {/* System Components Status */}
      <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold text-gray-900">État des Composants</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Service</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Statut</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Disponibilité</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Temps de Réponse</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trafic</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Erreurs</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Dernière Vérification</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {components.map((component) => (
                <tr key={component.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {component.id.includes('database') ? <Database className="w-5 h-5 text-gray-400 mr-3" /> :
                       component.id.includes('cache') || component.id.includes('queue') ? <Server className="w-5 h-5 text-gray-400 mr-3" /> :
                       component.id.includes('auth') ? <Shield className="w-5 h-5 text-gray-400 mr-3" /> :
                       <Wifi className="w-5 h-5 text-gray-400 mr-3" />}
                      <div>
                        <div className="text-sm font-medium text-gray-900">{component.name}</div>
                        <div className="text-sm text-gray-500">{component.id}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      {getStatusIcon(component.status)}
                      <span className={`ml-2 px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(component.status)}`}>
                        {component.status === 'healthy' ? 'Sain' :
                         component.status === 'warning' ? 'Attention' : 'Critique'}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{component.uptime}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{component.responseTime}ms</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{component.requestsPerMin}/min</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-medium ${component.errors > 10 ? 'text-red-600' : component.errors > 0 ? 'text-yellow-600' : 'text-green-600'}`}>
                      {component.errors}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {component.lastCheck.toLocaleTimeString('fr-FR')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Performance */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Système</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="cpu" stackId="1" stroke="#EF4444" fill="#EF4444" fillOpacity={0.6} name="CPU %" />
              <Area type="monotone" dataKey="memory" stackId="2" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} name="Mémoire %" />
              <Area type="monotone" dataKey="network" stackId="3" stroke="#10B981" fill="#10B981" fillOpacity={0.6} name="Réseau %" />
              <Area type="monotone" dataKey="disk" stackId="4" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.6} name="Disque %" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Error Distribution */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Distribution des Erreurs</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={errorData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {errorData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Uptime and Security */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Uptime Trend */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Tendance de Disponibilité</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={uptimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[95, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="uptime" stroke="#10B981" strokeWidth={3} name="Disponibilité %" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Security Events */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Événements de Sécurité</h3>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {securityEvents.map((event) => (
              <div key={event.id} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                <Shield className="w-5 h-5 text-gray-400 mt-0.5" />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-900">{event.message}</span>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getSeverityColor(event.severity)}`}>
                      {event.severity === 'high' ? 'Élevé' :
                       event.severity === 'medium' ? 'Moyen' : 'Faible'}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Source: {event.source} • {event.timestamp.toLocaleString('fr-FR')}
                  </div>
                  <div className="text-xs mt-1">
                    <span className={`px-2 py-0.5 rounded-full ${
                      event.status === 'resolved' ? 'bg-green-100 text-green-800' :
                      event.status === 'investigating' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-blue-100 text-blue-800'
                    }`}>
                      {event.status === 'resolved' ? 'Résolu' :
                       event.status === 'investigating' ? 'En Investigation' : 'Surveillance'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System Recommendations */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommandations Système</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-yellow-600 mr-2 mt-0.5" />
              <div>
                <h4 className="font-medium text-yellow-800">Cache Redis</h4>
                <p className="text-sm text-yellow-700 mt-1">
                  Performance dégradée détectée. Vérifiez la configuration de la mémoire.
                </p>
                <button className="text-xs text-yellow-800 underline mt-2">Voir détails</button>
              </div>
            </div>
          </div>

          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-red-600 mr-2 mt-0.5" />
              <div>
                <h4 className="font-medium text-red-800">Service d'Authentification</h4>
                <p className="text-sm text-red-700 mt-1">
                  Temps de réponse élevé. Intervention immédiate recommandée.
                </p>
                <button className="text-xs text-red-800 underline mt-2">Action requise</button>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start">
              <CheckCircle className="w-5 h-5 text-blue-600 mr-2 mt-0.5" />
              <div>
                <h4 className="font-medium text-blue-800">Optimisation</h4>
                <p className="text-sm text-blue-700 mt-1">
                  Système globalement stable. Considérez une mise à l'échelle préventive.
                </p>
                <button className="text-xs text-blue-800 underline mt-2">Planifier</button>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>
    </Layout>
  );
};

export default SystemHealth;
