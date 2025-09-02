import axios from 'axios';

// API base configuration (readable via Vite env: VITE_API_BASE)
const API_BASE_URL = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication if needed
api.interceptors.request.use(
  (config) => {
    // Add demo auth token for API access
    config.headers.Authorization = 'Bearer demo-token';
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// ===================
// DEMAND FORECASTING
// ===================
export const demandForecastingAPI = {
  // Get demand forecast for a product
  getForecast: async (productId, forecastDays = 30, includeConfidenceIntervals = true) => {
    try {
      const response = await api.post('/demand/forecast', {
        product_id: productId,
        forecast_days: forecastDays,
        include_confidence_intervals: includeConfidenceIntervals
      });
      return response.data;
    } catch (error) {
      console.error('Error getting demand forecast:', error);
      throw error;
    }
  },

  // Get demand analytics
  getAnalytics: async (productId, period = 'week') => {
    try {
      const response = await api.get(`/demand/analytics`, {
        params: { product_id: productId, period }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting demand analytics:', error);
      throw error;
    }
  },
};

// ===================
// TRANSPORT OPTIMIZATION
// ===================
export const transportAPI = {
  // Optimize transport routes
  optimizeRoutes: async (deliveryPoints, vehicles, objectives) => {
    try {
      const response = await api.post('/transport/optimize', {
        delivery_points: deliveryPoints,
        vehicles: vehicles,
        optimization_objectives: objectives
      });
      return response.data;
    } catch (error) {
      console.error('Error optimizing routes:', error);
      throw error;
    }
  },

  // Get transport analytics
  getAnalytics: async (period = 'week') => {
    try {
      const response = await api.get('/transport/analytics', {
        params: { period }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting transport analytics:', error);
      throw error;
    }
  },

  // Calculate route costs
  calculateCosts: async (routes) => {
    try {
      const response = await api.post('/transport/calculate-costs', { routes });
      return response.data;
    } catch (error) {
      console.error('Error calculating route costs:', error);
      throw error;
    }
  },
};

// ===================
// PREDICTIVE MAINTENANCE
// ===================
export const maintenanceAPI = {
  // Analyze equipment maintenance needs
  analyzeEquipment: async (equipmentData, sensorReadings) => {
    try {
      const response = await api.post('/maintenance/analyze', {
        equipment: equipmentData,
        sensor_readings: sensorReadings
      });
      return response.data;
    } catch (error) {
      console.error('Error analyzing equipment:', error);
      throw error;
    }
  },

  // Get maintenance schedule
  getSchedule: async (equipmentId) => {
    try {
      const response = await api.get(`/maintenance/schedule/${equipmentId}`);
      return response.data;
    } catch (error) {
      console.error('Error getting maintenance schedule:', error);
      throw error;
    }
  },

  // Get equipment health
  getEquipmentHealth: async () => {
    try {
      const response = await api.get('/maintenance/health');
      return response.data;
    } catch (error) {
      console.error('Error getting equipment health:', error);
      throw error;
    }
  },
};

// ===================
// REAL-TIME METRICS
// ===================
export const metricsAPI = {
  // Get current system metrics
  getCurrentMetrics: async () => {
    try {
      const response = await api.get('/metrics/current');
      return response.data;
    } catch (error) {
      console.error('Error getting current metrics:', error);
      throw error;
    }
  },

  // Get historical metrics
  getHistoricalMetrics: async (period = '24h') => {
    try {
      const response = await api.get('/metrics/historical', {
        params: { period }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting historical metrics:', error);
      throw error;
    }
  },

  // Get system health
  getSystemHealth: async () => {
    try {
      const response = await api.get('/metrics/health');
      return response.data;
    } catch (error) {
      console.error('Error getting system health:', error);
      throw error;
    }
  },
};

// ===================
// DASHBOARD
// ===================
export const dashboardAPI = {
  // Get dashboard overview
  getOverview: async () => {
    try {
      const response = await api.get('/dashboard/overview');
      return response.data;
    } catch (error) {
      console.error('Error getting dashboard overview:', error);
      throw error;
    }
  },

  // Get KPIs
  getKPIs: async (period = 'today') => {
    try {
      const response = await api.get('/dashboard/kpis', {
        params: { period }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting KPIs:', error);
      throw error;
    }
  },

  // Get alerts
  getAlerts: async () => {
    try {
      const response = await api.get('/dashboard/alerts');
      return response.data;
    } catch (error) {
      console.error('Error getting alerts:', error);
      throw error;
    }
  },
};

// ===================
// GENERAL API
// ===================
export const generalAPI = {
  // Check API health
  checkHealth: async () => {
    try {
      const response = await api.get('/');
      return response.data;
    } catch (error) {
      console.error('Error checking API health:', error);
      throw error;
    }
  },

  // Test API connection
  testConnection: async () => {
    try {
      // Try the API root first
      const response = await axios.get('http://localhost:8000/');
      return { connected: true, data: response.data };
    } catch (error) {
      return { connected: false, error: error.message };
    }
  },
};

export default api;
