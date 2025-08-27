import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import DemandForecasting from './pages/DemandForecasting'
import TransportOptimization from './pages/TransportOptimization'
import PredictiveMaintenance from './pages/PredictiveMaintenance'
import RealTimeMetrics from './pages/RealTimeMetrics'
import SystemHealth from './pages/SystemHealth'

function App() {
  return (
    <Router 
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/demand" element={<DemandForecasting />} />
          <Route path="/transport" element={<TransportOptimization />} />
          <Route path="/maintenance" element={<PredictiveMaintenance />} />
          <Route path="/metrics" element={<RealTimeMetrics />} />
          <Route path="/system" element={<SystemHealth />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App
