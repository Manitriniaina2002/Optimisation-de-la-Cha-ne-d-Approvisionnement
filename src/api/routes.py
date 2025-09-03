"""
API REST pour l'application Big Data Supply Chain
Endpoints pour toutes les fonctionnalités d'optimisation
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
import math

# Configuration
from src.config.settings import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

api_router = APIRouter()
security = HTTPBearer()

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)):
        """Authentification simple (à remplacer par une vraie auth)

        Notes:
        - HTTPBearer dependency returns an HTTPAuthorizationCredentials object with
            attributes `.scheme` and `.credentials`.
        - For the demo environment we accept the demo token or any token that
            begins with 'demo'. This helper reads `.credentials` safely to avoid
            AttributeError when called by FastAPI.
        """
        try:
                creds = token.credentials if hasattr(token, 'credentials') else token
                # Accept demo token or any bearer token that starts with 'demo'
                if isinstance(creds, str) and (creds == "demo-token" or creds.startswith("demo")):
                        return {"user_id": "demo_user"}
        except Exception:
                # In case of unexpected token shapes, fall back to demo user for local dev
                return {"user_id": "demo_user"}

        # For demo purposes, accept any token (keep permissive for development)
        return {"user_id": "demo_user"}

async def get_optional_user(authorization: str = None):
    """Authentification optionnelle pour demo"""
    if authorization:
        return {"user_id": "demo_user"}
    return None

# ==========================
#        DATA MODELS
# ==========================
class DemandForecastRequest(BaseModel):
    product_id: str
    forecast_days: int = Field(default=30, ge=1, le=365)
    include_confidence_intervals: bool = True

class DemandForecastResponse(BaseModel):
    product_id: str
    forecast_dates: List[str]
    predictions: Dict[str, List[float]]
    confidence_interval: float
    upper_bound: List[float]
    lower_bound: List[float]
    model_accuracy: Dict[str, float]
    generated_at: str

class DeliveryPointRequest(BaseModel):
    id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    demand: float = Field(..., ge=0)
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    service_time: int = Field(0, ge=0, description="Temps de service (min)")
    priority: int = Field(1, ge=1, le=4)

class VehicleRequest(BaseModel):
    id: str
    capacity: float = Field(..., gt=0)
    start_latitude: float = Field(..., ge=-90, le=90)
    start_longitude: float = Field(..., ge=-180, le=180)
    end_latitude: float = Field(..., ge=-90, le=90)
    end_longitude: float = Field(..., ge=-180, le=180)
    available_start: Optional[str] = None
    available_end: Optional[str] = None
    cost_per_km: float = Field(default=0.15, ge=0)
    co2_per_km: float = Field(default=0.25, ge=0)

class TransportOptimizationRequest(BaseModel):
    delivery_points: List[DeliveryPointRequest]
    vehicles: List[VehicleRequest]
    optimization_objectives: Dict[str, float] = Field(
        default={'cost': 0.4, 'time': 0.3, 'co2': 0.2, 'service_quality': 0.1}
    )

class RouteResponse(BaseModel):
    vehicle_id: str
    stops: List[Dict[str, Any]]
    total_distance: float
    total_time: int
    total_cost: float
    co2_emissions: float
    load_factor: float

class TransportOptimizationResponse(BaseModel):
    routes: List[RouteResponse]
    summary: Dict[str, float]
    optimization_time: float
    generated_at: str

class EquipmentRequest(BaseModel):
    id: str
    name: str
    type: str
    location: Optional[str] = None
    installation_date: Optional[str] = None
    last_maintenance: Optional[str] = None
    criticality_level: int = Field(..., ge=1, le=4)
    operating_hours: float = Field(..., ge=0)

class SensorReadingRequest(BaseModel):
    equipment_id: str
    timestamp: str
    sensor_type: str
    value: float
    unit: str
    quality: float = Field(default=1.0, ge=0, le=1)

class MaintenanceAnalysisRequest(BaseModel):
    equipment: EquipmentRequest
    sensor_readings: List[SensorReadingRequest]

class MaintenanceAnalysisResponse(BaseModel):
    equipment_id: str
    status: str
    failure_probability: float
    time_to_failure_hours: Optional[float]
    recommendations: List[Dict[str, Any]]
    alerts: List[Dict[str, str]]
    generated_at: str

class MetricsResponse(BaseModel):
    orders_per_hour: float
    avg_order_value: float
    equipment_alerts: int
    transport_efficiency: float
    overall_efficiency: float
    active_alerts_count: int
    timestamp: str

# Helper
def _mean(vals: list[float], default: float = 0.0) -> float:
    return sum(vals) / len(vals) if vals else default

# ==========================
#         GENERAL
# ==========================
@api_router.get("/", tags=["General"])
async def api_root():
    return {
        "name": "Big Data Supply Chain Optimization API",
        "version": "1.0.0",
        "description": "API pour l'optimisation de la chaîne d'approvisionnement",
        "endpoints": {
            "demand": "/demand",
            "transport": "/transport",
            "maintenance": "/maintenance",
            "metrics": "/metrics"
        }
    }

# ==========================
#   DEMAND FORECASTING
# ==========================
@api_router.post("/demand/forecast", response_model=DemandForecastResponse, tags=["Demand Forecasting"])
async def forecast_demand(
    request: DemandForecastRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    try:
        forecast_dates = [
            (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(1, request.forecast_days + 1)
        ]
        base_demand = 1000.0
        predictions = {
            'ensemble': [
                base_demand + random.gauss(0, 50) + 100 * math.sin(2 * math.pi * i / 30)
                for i in range(request.forecast_days)
            ],
            'prophet': [
                base_demand + random.gauss(0, 40) + 80 * math.sin(2 * math.pi * i / 30)
                for i in range(request.forecast_days)
            ],
            'xgboost': [
                base_demand + random.gauss(0, 60) + 120 * math.sin(2 * math.pi * i / 30)
                for i in range(request.forecast_days)
            ]
        }
        ensemble_pred = predictions['ensemble']
        confidence_interval = 50.0
        response = DemandForecastResponse(
            product_id=request.product_id,
            forecast_dates=forecast_dates,
            predictions=predictions,
            confidence_interval=confidence_interval,
            upper_bound=[p + confidence_interval for p in ensemble_pred],
            lower_bound=[max(0.0, p - confidence_interval) for p in ensemble_pred],
            model_accuracy={'mae': 45.2, 'mape': 5.8, 'r2': 0.94},
            generated_at=datetime.now().isoformat()
        )
        logger.info(f"Prévision générée pour {request.product_id}: {request.forecast_days} jours")
        return response
    except Exception as e:
        logger.error(f"Erreur prévision demande: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prévision: {str(e)}")

@api_router.get("/demand/products", tags=["Demand Forecasting"])
async def get_products(
    category: Optional[str] = Query(None, description="Filtrer par catégorie"),
    user: dict = Depends(get_current_user)
):
    products = [
        {"id": "PROD_A", "name": "Produit A", "category": "Electronics"},
        {"id": "PROD_B", "name": "Produit B", "category": "Clothing"},
        {"id": "PROD_C", "name": "Produit C", "category": "Food"},
        {"id": "PROD_D", "name": "Produit D", "category": "Home"}
    ]
    if category:
        products = [p for p in products if p['category'].lower() == category.lower()]
    return {"products": products, "total": len(products)}

# ==========================
#  TRANSPORT OPTIMIZATION
# ==========================
@api_router.post("/transport/optimize", response_model=TransportOptimizationResponse, tags=["Transport Optimization"])
async def optimize_routes(
    request: TransportOptimizationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    try:
        import time as _time
        start_time = _time.time()
        routes: List[RouteResponse] = []
        total_distance = 0.0
        total_cost = 0.0
        total_co2 = 0.0
        for i, vehicle in enumerate(request.vehicles):
            assigned_stops = request.delivery_points[i::max(1, len(request.vehicles))]
            if assigned_stops:
                distance = random.uniform(50, 200)
                time_hours = distance / 50.0
                cost = distance * float(vehicle.cost_per_km)
                co2 = distance * float(vehicle.co2_per_km)
                load = sum(stop.demand for stop in assigned_stops)
                route = RouteResponse(
                    vehicle_id=vehicle.id,
                    stops=[
                        {
                            "id": stop.id,
                            "latitude": stop.latitude,
                            "longitude": stop.longitude,
                            "demand": stop.demand,
                            "service_time": stop.service_time
                        }
                        for stop in assigned_stops
                    ],
                    total_distance=distance,
                    total_time=int(time_hours * 60),
                    total_cost=cost,
                    co2_emissions=co2,
                    load_factor=min(1.0, load / float(vehicle.capacity))
                )
                routes.append(route)
                total_distance += distance
                total_cost += cost
                total_co2 += co2
        optimization_time = _time.time() - start_time
        response = TransportOptimizationResponse(
            routes=routes,
            summary={
                "total_distance_km": total_distance,
                "total_cost_euros": total_cost,
                "total_co2_kg": total_co2,
                "vehicles_used": len(routes),
                "deliveries_count": len(request.delivery_points),
                "average_load_factor": _mean([r.load_factor for r in routes], 0.0) if routes else 0.0,
                "optimization_score": random.uniform(0.85, 0.95)
            },
            optimization_time=optimization_time,
            generated_at=datetime.now().isoformat()
        )
        logger.info(f"Optimisation transport: {len(routes)} routes, {total_distance:.1f}km")
        return response
    except Exception as e:
        logger.error(f"Erreur optimisation transport: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'optimisation: {str(e)}")

@api_router.get("/transport/vehicles", tags=["Transport Optimization"])
async def get_vehicles(
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    user: dict = Depends(get_current_user)
):
    vehicles = [
        {"id": "VEH_001", "type": "Truck", "capacity": 1000, "status": "available", "location": {"lat": 45.764, "lon": 4.835}, "fuel_level": 85.5, "last_maintenance": "2024-01-15"},
        {"id": "VEH_002", "type": "Van",   "capacity":  500, "status": "in_transit", "location": {"lat": 45.780, "lon": 4.850}, "fuel_level": 62.3, "last_maintenance": "2024-01-20"},
        {"id": "VEH_003", "type": "Truck", "capacity": 1200, "status": "maintenance", "location": {"lat": 45.750, "lon": 4.820}, "fuel_level": 95.0, "last_maintenance": "2024-02-01"}
    ]
    if status:
        vehicles = [v for v in vehicles if v['status'] == status]
    return {"vehicles": vehicles, "total": len(vehicles)}

# ==========================
#   PREDICTIVE MAINTENANCE
# ==========================
@api_router.post("/maintenance/analyze", response_model=MaintenanceAnalysisResponse, tags=["Predictive Maintenance"])
async def analyze_equipment(
    request: MaintenanceAnalysisRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    try:
        equipment_id = request.equipment.id
        temp_readings = [r.value for r in request.sensor_readings if r.sensor_type == 'temperature']
        vibration_readings = [r.value for r in request.sensor_readings if r.sensor_type == 'vibration']
        avg_temp = _mean(temp_readings, 60.0)
        avg_vibration = _mean(vibration_readings, 2.0)
        temp_risk = max(0.0, (avg_temp - 70.0) / 20.0)
        vibration_risk = max(0.0, (avg_vibration - 3.0) / 2.0)
        failure_probability = min(0.95, (temp_risk + vibration_risk) / 2.0)
        if failure_probability > 0.8:
            status = "critical"; time_to_failure = 24.0
        elif failure_probability > 0.6:
            status = "warning"; time_to_failure = 168.0
        else:
            status = "healthy"; time_to_failure = None
        recommendations: List[Dict[str, Any]] = []
        alerts: List[Dict[str, str]] = []
        if status == "critical":
            recommendations.append({
                "priority": "urgent",
                "action": "Arrêt immédiat et inspection",
                "estimated_cost": 5000,
                "deadline": (datetime.now() + timedelta(hours=4)).isoformat()
            })
            alerts.append({"type": "critical_failure_risk", "message": f"Risque de panne critique sur {equipment_id}"})
        elif status == "warning":
            recommendations.append({
                "priority": "high",
                "action": "Maintenance préventive programmée",
                "estimated_cost": 2000,
                "deadline": (datetime.now() + timedelta(days=3)).isoformat()
            })
            alerts.append({"type": "maintenance_required", "message": f"Maintenance recommandée pour {equipment_id}"})
        return MaintenanceAnalysisResponse(
            equipment_id=equipment_id,
            status=status,
            failure_probability=failure_probability,
            time_to_failure_hours=time_to_failure,
            recommendations=recommendations,
            alerts=alerts,
            generated_at=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Erreur analyse maintenance: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {str(e)}")

@api_router.get("/maintenance/equipment", tags=["Predictive Maintenance"])
async def get_equipment(
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    criticality: Optional[int] = Query(None, ge=1, le=4, description="Filtrer par criticité"),
    user: dict = Depends(get_current_user)
):
    equipment_list = [
        {"id": "PUMP_001", "name": "Pompe Principale A", "type": "pump", "status": "healthy", "criticality": 3, "last_maintenance": "2024-01-15", "operating_hours": 8760, "failure_risk": 0.15},
        {"id": "CONV_002", "name": "Convoyeur B", "type": "conveyor", "status": "warning", "criticality": 2, "last_maintenance": "2023-11-20", "operating_hours": 12500, "failure_risk": 0.65},
        {"id": "ROBOT_003", "name": "Robot de Palettisation", "type": "robot", "status": "critical", "criticality": 4, "last_maintenance": "2023-09-10", "operating_hours": 15800, "failure_risk": 0.85}
    ]
    if status:
        equipment_list = [e for e in equipment_list if e['status'] == status]
    if criticality:
        equipment_list = [e for e in equipment_list if e['criticality'] == criticality]
    return {"equipment": equipment_list, "total": len(equipment_list)}

@api_router.get("/maintenance/schedule", tags=["Predictive Maintenance"])
async def get_maintenance_schedule(
    days: int = Query(30, ge=1, le=365, description="Horizon en jours"),
    user: dict = Depends(get_current_user)
):
    schedule: List[Dict[str, Any]] = []
    for i in range(random.randint(5, 14)):
        equipment_id = f"EQUIP_{random.randint(1, 99):03d}"
        scheduled_date = datetime.now() + timedelta(days=random.randint(1, days))
        schedule.append({
            "equipment_id": equipment_id,
            "type": random.choice(["preventive", "corrective", "inspection"]),
            "priority": random.choice(["low", "medium", "high", "urgent"]),
            "scheduled_date": scheduled_date.isoformat(),
            "estimated_duration": random.randint(2, 7),
            "estimated_cost": random.randint(500, 4999),
            "technician_required": random.choice(["Level_1", "Level_2", "Specialist"])
        })
    schedule.sort(key=lambda x: x['scheduled_date'])
    return {
        "schedule": schedule,
        "summary": {
            "total_interventions": len(schedule),
            "total_cost": sum(item['estimated_cost'] for item in schedule),
            "urgent_count": len([item for item in schedule if item['priority'] == 'urgent'])
        }
    }

# ==========================
#      REAL-TIME METRICS
# ==========================
@api_router.get("/metrics/real-time", response_model=MetricsResponse, tags=["Real-time Metrics"])
async def get_real_time_metrics(user: dict = Depends(get_current_user)):
    try:
        hour = datetime.now().hour
        base_orders = 60 if 8 <= hour <= 18 else 20
        base_efficiency = 85 if 8 <= hour <= 18 else 70
        metrics = MetricsResponse(
            orders_per_hour=base_orders + random.gauss(0, 10),
            avg_order_value=random.uniform(150, 350),
            equipment_alerts=random.randint(0, 4),
            transport_efficiency=base_efficiency + random.gauss(0, 5),
            overall_efficiency=random.uniform(80, 95),
            active_alerts_count=random.randint(0, 7),
            timestamp=datetime.now().isoformat()
        )
        return metrics
    except Exception as e:
        logger.error(f"Erreur métriques temps réel: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération: {str(e)}")

@api_router.get("/metrics/kpis", tags=["Real-time Metrics"])
async def get_kpis(
    period: str = Query("today", description="Période: today, week, month"),
    user: dict = Depends(get_current_user)
):
    if period == "today":
        kpis = {
            "cost_reduction_percent": random.uniform(15, 25),
            "delivery_performance_percent": random.uniform(92, 98),
            "inventory_turnover": random.uniform(10, 14),
            "customer_satisfaction_percent": random.uniform(85, 92),
            "sustainability_score": random.uniform(68, 78),
            "orders_processed": random.randint(800, 1500),
            "avg_delivery_time_hours": random.uniform(20, 48),
            "transport_cost_per_km": random.uniform(0.12, 0.18)
        }
    else:
        multiplier = 7 if period == "week" else 30
        kpis = {
            "cost_reduction_percent": random.uniform(18, 22),
            "delivery_performance_percent": random.uniform(94, 96),
            "inventory_turnover": random.uniform(11, 13),
            "customer_satisfaction_percent": random.uniform(87, 90),
            "sustainability_score": random.uniform(70, 75),
            "orders_processed": random.randint(800, 1500) * multiplier,
            "avg_delivery_time_hours": random.uniform(22, 36),
            "transport_cost_per_km": random.uniform(0.13, 0.16)
        }
    return {
        "period": period,
        "kpis": kpis,
        "generated_at": datetime.now().isoformat(),
        "targets": {
            "cost_reduction_percent": 20,
            "delivery_performance_percent": 95,
            "inventory_turnover": 12,
            "customer_satisfaction_percent": 90,
            "sustainability_score": 80
        }
    }

@api_router.get("/metrics/alerts", tags=["Real-time Metrics"])
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filtrer par sévérité"),
    limit: int = Query(50, ge=1, le=200, description="Nombre max d'alertes"),
    user: dict = Depends(get_current_user)
):
    alert_types = ["equipment_anomaly", "low_inventory", "transport_delay", "demand_spike", "maintenance_required", "route_inefficient"]
    severities = ["low", "medium", "high", "critical"]
    alerts: List[Dict[str, Any]] = []
    for i in range(random.randint(3, 14)):
        alert_severity = random.choice(severities)
        if severity and alert_severity != severity:
            continue
        alerts.append({
            "id": f"alert_{i:04d}",
            "type": random.choice(alert_types),
            "severity": alert_severity,
            "message": f"Alerte simulée {i}",
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat(),
            "source": random.choice(["IoT", "ERP", "Transport", "Manual"]),
            "acknowledged": random.choice([True, False]),
            "actions": ["Vérifier", "Analyser", "Corriger"]
        })
    alerts.sort(key=lambda x: x['timestamp'], reverse=True)
    return {
        "alerts": alerts[:limit],
        "total": len(alerts),
        "summary": {
            "critical": len([a for a in alerts if a['severity'] == 'critical']),
            "high": len([a for a in alerts if a['severity'] == 'high']),
            "medium": len([a for a in alerts if a['severity'] == 'medium']),
            "low": len([a for a in alerts if a['severity'] == 'low'])
        }
    }

# ==========================
#          SYSTEM
# ==========================
@api_router.get("/system/health", tags=["System"])
async def system_health():
    try:
        import psutil
        import time as _time
        health_checks = {
            "api_status": "healthy",
            "database_connection": "healthy",
            "kafka_connection": "healthy",
            "ml_models_loaded": "healthy",
            "cpu_usage_percent": float(psutil.cpu_percent()),
            "memory_usage_percent": float(psutil.virtual_memory().percent),
            "disk_usage_percent": float(psutil.disk_usage('/').percent),
            "uptime_seconds": float(_time.time() - psutil.boot_time()),
            "timestamp": datetime.now().isoformat()
        }
        cpu_ok = health_checks["cpu_usage_percent"] < 80
        memory_ok = health_checks["memory_usage_percent"] < 80
        disk_ok = health_checks["disk_usage_percent"] < 90
        overall_status = "healthy" if all([cpu_ok, memory_ok, disk_ok]) else "degraded"
        return {
            "status": overall_status,
            "checks": health_checks,
            "recommendations": [
                "Système fonctionnel",
                "Tous les services opérationnels"
            ] if overall_status == "healthy" else [
                "Surveiller utilisation ressources",
                "Considérer scaling si nécessaire"
            ]
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

@api_router.get("/system/stats", tags=["System"])
async def system_statistics(user: dict = Depends(get_current_user)):
    stats = {
        "requests_processed_today": random.randint(5000, 15000),
        "forecasts_generated": random.randint(500, 1500),
        "routes_optimized": random.randint(200, 800),
        "maintenance_analyses": random.randint(100, 400),
        "alerts_generated": random.randint(50, 200),
        "data_points_processed": random.randint(50000, 200000),
        "average_response_time_ms": random.uniform(50, 200),
        "system_efficiency_percent": random.uniform(85, 95)
    }
    return {"statistics": stats, "period": "last_24_hours", "generated_at": datetime.now().isoformat()}

# ==========================
#        DASHBOARD
# ==========================
@api_router.get("/dashboard/overview", tags=["Dashboard"])
async def get_dashboard_overview():
    """Get dashboard overview with key metrics"""
    try:
        overview = {
            "total_orders": random.randint(1000, 5000),
            "active_shipments": random.randint(50, 200),
            "equipment_health": random.uniform(85, 95),
            "system_efficiency": random.uniform(88, 96),
            "cost_savings": random.uniform(15000, 50000),
            "performance_trends": {
                "orders": [random.randint(50, 150) for _ in range(7)],
                "efficiency": [random.uniform(85, 95) for _ in range(7)],
                "costs": [random.uniform(1000, 3000) for _ in range(7)]
            },
            "recent_activities": [
                {
                    "id": i,
                    "type": "order",
                    "message": f"Nouvelle commande #{12000 + i} reçue",
                    "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                    "priority": "normal"
                } for i in range(5)
            ]
        }
        return overview
    except Exception as e:
        logger.error(f"Erreur dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur dashboard: {str(e)}")

@api_router.get("/dashboard/kpis", tags=["Dashboard"])
async def get_dashboard_kpis(
    period: str = Query("today", description="Période: today, week, month")
):
    """Get dashboard KPIs for specified period"""
    try:
        kpis = {
            "orders_per_hour": random.uniform(50, 100),
            "avg_order_value": random.uniform(250, 350),
            "transport_efficiency": random.uniform(88, 96),
            "overall_efficiency": random.uniform(85, 94),
            "equipment_alerts": random.randint(0, 15),
            "active_alerts_count": random.randint(1, 8),
            "cost_per_order": random.uniform(15, 35),
            "delivery_success_rate": random.uniform(92, 99),
            "period": period,
            "last_updated": datetime.now().isoformat()
        }
        return kpis
    except Exception as e:
        logger.error(f"Erreur dashboard KPIs: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur KPIs: {str(e)}")

@api_router.get("/dashboard/alerts", tags=["Dashboard"])
async def get_dashboard_alerts():
    """Get current system alerts for dashboard"""
    try:
        alerts = [
            {
                "id": 1,
                "type": "warning",
                "title": "Température élevée",
                "message": "Température élevée détectée dans l'entrepôt B",
                "severity": "medium",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "status": "active"
            },
            {
                "id": 2,
                "type": "info",
                "title": "Maintenance programmée",
                "message": "Maintenance préventive équipement EQ001 dans 2h",
                "severity": "low",
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "status": "scheduled"
            },
            {
                "id": 3,
                "type": "success",
                "title": "Optimisation réussie",
                "message": "Route optimisée avec 25% d'économies",
                "severity": "low",
                "timestamp": (datetime.now() - timedelta(minutes=45)).isoformat(),
                "status": "resolved"
            }
        ]
        return {"alerts": alerts, "total_count": len(alerts)}
    except Exception as e:
        logger.error(f"Erreur dashboard alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur alerts: {str(e)}")
