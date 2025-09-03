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
  // Backend provides a products list endpoint; if a productId is provided
  // use it as a category filter for demo purposes.
  const params = {};
  if (productId) params.category = productId;
  const response = await api.get(`/demand/products`, { params });
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
      // No dedicated transport analytics endpoint on backend; reuse metrics KPIs
      // which include transport-related KPIs for the requested period.
      const response = await api.get('/metrics/kpis', {
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
  // Backend does not expose a calculate-costs endpoint. For now attempt
  // to call the optimize endpoint if routes are provided as delivery points
  // (best-effort mapping). If the payload doesn't match, the caller will
  // receive the backend error which should be handled upstream.
  const payload = { routes };
  const response = await api.post('/transport/optimize', payload);
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
  getSchedule: async (days = 30) => {
    try {
      // Backend exposes /maintenance/schedule with a 'days' query parameter
      const response = await api.get(`/maintenance/schedule`, { params: { days } });
      return response.data;
    } catch (error) {
      console.error('Error getting maintenance schedule:', error);
      throw error;
    }
  },

  // Get equipment health
  getEquipmentHealth: async () => {
    try {
  // Backend exposes /maintenance/equipment for equipment list/health
  const response = await api.get('/maintenance/equipment');
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
  // Backend provides a real-time metrics endpoint
  const response = await api.get('/metrics/real-time');
      return response.data;
    } catch (error) {
      console.error('Error getting current metrics:', error);
      throw error;
    }
  },

  // Get historical metrics
  getHistoricalMetrics: async (period = '24h') => {
    try {
      // Map common period formats to the backend 'period' query parameter
      let p = 'today';
      if (typeof period === 'string') {
        const lp = period.toLowerCase();
        if (lp.includes('24') || lp === 'today') p = 'today';
        else if (lp.includes('7') || lp.includes('week')) p = 'week';
        else if (lp.includes('30') || lp.includes('month')) p = 'month';
      }
      const response = await api.get('/metrics/kpis', { params: { period: p } });
      return response.data;
    } catch (error) {
      console.error('Error getting historical metrics:', error);
      throw error;
    }
  },

  // Get system health
  getSystemHealth: async () => {
    try {
  // Backend exposes system health under /system/health
  const response = await api.get('/system/health');
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
