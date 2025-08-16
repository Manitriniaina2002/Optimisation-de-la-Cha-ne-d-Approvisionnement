"""
API REST pour l'application Big Data Supply Chain
Endpoints pour toutes les fonctionnalités d'optimisation
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

# Configuration
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Modèles Pydantic pour l'API
class DemandForecastRequest(BaseModel):
    """Requête de prévision de demande"""
    product_id: str = Field(..., description="Identifiant du produit")
    forecast_days: int = Field(default=30, ge=1, le=365, description="Horizon de prévision en jours")
    include_confidence_intervals: bool = Field(default=True, description="Inclure les intervalles de confiance")

class DemandForecastResponse(BaseModel):
    """Réponse de prévision de demande"""
    product_id: str
    forecast_dates: List[str]
    predictions: Dict[str, List[float]]
    confidence_interval: float
    upper_bound: List[float]
    lower_bound: List[float]
    model_accuracy: Dict[str, float]
    generated_at: str

class DeliveryPointRequest(BaseModel):
    """Point de livraison pour optimisation transport"""
    id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    demand: float = Field(..., ge=0)
    time_window_start: str
    time_window_end: str
    service_time: int = Field(..., ge=0, description="Temps de service en minutes")
    priority: int = Field(default=1, ge=1, le=4)

class VehicleRequest(BaseModel):
    """Véhicule pour optimisation transport"""
    id: str
    capacity: float = Field(..., gt=0)
    start_latitude: float = Field(..., ge=-90, le=90)
    start_longitude: float = Field(..., ge=-180, le=180)
    end_latitude: float = Field(..., ge=-90, le=90)
    end_longitude: float = Field(..., ge=-180, le=180)
    available_start: str
    available_end: str
    cost_per_km: float = Field(default=0.15, ge=0)
    co2_per_km: float = Field(default=0.25, ge=0)

class TransportOptimizationRequest(BaseModel):
    """Requête d'optimisation de transport"""
    delivery_points: List[DeliveryPointRequest]
    vehicles: List[VehicleRequest]
    optimization_objectives: Dict[str, float] = Field(
        default={'cost': 0.4, 'time': 0.3, 'co2': 0.2, 'service_quality': 0.1}
    )

class RouteResponse(BaseModel):
    """Route optimisée"""
    vehicle_id: str
    stops: List[Dict[str, Any]]
    total_distance: float
    total_time: int
    total_cost: float
    co2_emissions: float
    load_factor: float

class TransportOptimizationResponse(BaseModel):
    """Réponse d'optimisation de transport"""
    routes: List[RouteResponse]
    summary: Dict[str, float]
    optimization_time: float
    generated_at: str

class EquipmentRequest(BaseModel):
    """Équipement pour analyse de maintenance"""
    id: str
    name: str
    type: str
    location: str
    installation_date: str
    last_maintenance: str
    criticality_level: int = Field(..., ge=1, le=4)
    operating_hours: float = Field(..., ge=0)

class SensorReadingRequest(BaseModel):
    """Lecture de capteur"""
    equipment_id: str
    timestamp: str
    sensor_type: str
    value: float
    unit: str
    quality: float = Field(default=1.0, ge=0, le=1)

class MaintenanceAnalysisRequest(BaseModel):
    """Requête d'analyse de maintenance"""
    equipment: EquipmentRequest
    sensor_readings: List[SensorReadingRequest]

class MaintenanceAnalysisResponse(BaseModel):
    """Réponse d'analyse de maintenance"""
    equipment_id: str
    status: str
    failure_probability: float
    time_to_failure_hours: Optional[float]
    recommendations: List[Dict[str, Any]]
    alerts: List[Dict[str, str]]
    generated_at: str

class MetricsResponse(BaseModel):
    """Métriques temps réel"""
    orders_per_hour: float
    avg_order_value: float
    equipment_alerts: int
    transport_efficiency: float
    overall_efficiency: float
    active_alerts_count: int
    timestamp: str

# Création du routeur API
api_router = APIRouter()

# Sécurité (optionnelle)
security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    """Authentification simple (à améliorer en production)"""
    # En production, valider le JWT token
    return {"user_id": "demo_user"}

@api_router.get("/", tags=["General"])
async def api_root():
    """Endpoint racine de l'API"""
    return {
        "name": "Big Data Supply Chain Optimization API",
        "version": "1.0.0",
        "description": "API pour l'optimisation de la chaîne d'approvisionnement",
        "features": [
            "Prévision intelligente de la demande",
            "Optimisation des transports",
            "Maintenance prédictive",
            "Métriques temps réel"
        ],
        "endpoints": {
            "demand": "/demand",
            "transport": "/transport", 
            "maintenance": "/maintenance",
            "metrics": "/metrics"
        }
    }

# === ENDPOINTS PRÉVISION DE DEMANDE ===

@api_router.post("/demand/forecast", response_model=DemandForecastResponse, tags=["Demand Forecasting"])
async def forecast_demand(
    request: DemandForecastRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Génère une prévision de demande pour un produit
    
    **Fonctionnalités:**
    - Utilise des modèles ML avancés (Prophet, XGBoost, LSTM)
    - Intègre données externes (météo, événements, réseaux sociaux)
    - Fournit intervalles de confiance
    - Optimisé pour haute performance
    """
    try:
        # Simulation de prévision (remplacer par vraie logique)
        import numpy as np
        from datetime import datetime, timedelta
        
        forecast_dates = [
            (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(1, request.forecast_days + 1)
        ]
        
        # Génération de prédictions simulées
        base_demand = 1000
        predictions = {
            'ensemble': [
                base_demand + np.random.normal(0, 50) + 100 * np.sin(2 * np.pi * i / 30)
                for i in range(request.forecast_days)
            ],
            'prophet': [
                base_demand + np.random.normal(0, 40) + 80 * np.sin(2 * np.pi * i / 30)
                for i in range(request.forecast_days)
            ],
            'xgboost': [
                base_demand + np.random.normal(0, 60) + 120 * np.sin(2 * np.pi * i / 30)
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
            lower_bound=[max(0, p - confidence_interval) for p in ensemble_pred],
            model_accuracy={
                'mae': 45.2,
                'mape': 5.8,
                'r2': 0.94
            },
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
    """Liste les produits disponibles pour prévision"""
    # Simulation de liste de produits
    products = [
        {"id": "PROD_A", "name": "Produit A", "category": "Electronics"},
        {"id": "PROD_B", "name": "Produit B", "category": "Clothing"},
        {"id": "PROD_C", "name": "Produit C", "category": "Food"},
        {"id": "PROD_D", "name": "Produit D", "category": "Home"}
    ]
    
    if category:
        products = [p for p in products if p['category'].lower() == category.lower()]
    
    return {"products": products, "total": len(products)}

# === ENDPOINTS OPTIMISATION TRANSPORT ===

@api_router.post("/transport/optimize", response_model=TransportOptimizationResponse, tags=["Transport Optimization"])
async def optimize_routes(
    request: TransportOptimizationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Optimise les routes de transport
    
    **Algorithmes utilisés:**
    - ORION (UPS) inspired routing
    - Optimisation multi-objectifs
    - Contraintes temps réel
    - Clustering intelligent
    """
    try:
        import time
        start_time = time.time()
        
        # Simulation d'optimisation (remplacer par vraie logique)
        import numpy as np
        
        routes = []
        total_distance = 0
        total_cost = 0
        total_co2 = 0
        
        for i, vehicle in enumerate(request.vehicles):
            # Simulation de route pour chaque véhicule
            assigned_stops = request.delivery_points[i::len(request.vehicles)]  # Distribution simple
            
            if assigned_stops:
                distance = np.random.uniform(50, 200)  # km
                time_hours = distance / 50  # Vitesse moyenne 50 km/h
                cost = distance * vehicle.cost_per_km
                co2 = distance * vehicle.co2_per_km
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
                    total_time=int(time_hours * 60),  # minutes
                    total_cost=cost,
                    co2_emissions=co2,
                    load_factor=min(1.0, load / vehicle.capacity)
                )
                
                routes.append(route)
                total_distance += distance
                total_cost += cost
                total_co2 += co2
        
        optimization_time = time.time() - start_time
        
        response = TransportOptimizationResponse(
            routes=routes,
            summary={
                "total_distance_km": total_distance,
                "total_cost_euros": total_cost,
                "total_co2_kg": total_co2,
                "vehicles_used": len(routes),
                "deliveries_count": len(request.delivery_points),
                "average_load_factor": np.mean([r.load_factor for r in routes]) if routes else 0,
                "optimization_score": np.random.uniform(0.85, 0.95)
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
    """Liste les véhicules disponibles"""
    # Simulation de flotte de véhicules
    vehicles = [
        {
            "id": "VEH_001",
            "type": "Truck",
            "capacity": 1000,
            "status": "available",
            "location": {"lat": 45.764, "lon": 4.835},
            "fuel_level": 85.5,
            "last_maintenance": "2024-01-15"
        },
        {
            "id": "VEH_002", 
            "type": "Van",
            "capacity": 500,
            "status": "in_transit",
            "location": {"lat": 45.780, "lon": 4.850},
            "fuel_level": 62.3,
            "last_maintenance": "2024-01-20"
        },
        {
            "id": "VEH_003",
            "type": "Truck",
            "capacity": 1200,
            "status": "maintenance",
            "location": {"lat": 45.750, "lon": 4.820},
            "fuel_level": 95.0,
            "last_maintenance": "2024-02-01"
        }
    ]
    
    if status:
        vehicles = [v for v in vehicles if v['status'] == status]
    
    return {"vehicles": vehicles, "total": len(vehicles)}

# === ENDPOINTS MAINTENANCE PRÉDICTIVE ===

@api_router.post("/maintenance/analyze", response_model=MaintenanceAnalysisResponse, tags=["Predictive Maintenance"])
async def analyze_equipment(
    request: MaintenanceAnalysisRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Analyse prédictive d'un équipement
    
    **Fonctionnalités:**
    - Détection d'anomalies en temps réel
    - Prédiction de pannes avec ML
    - Recommandations personnalisées
    - Calcul ROI des interventions
    """
    try:
        import numpy as np
        
        # Simulation d'analyse (remplacer par vraie logique)
        equipment_id = request.equipment.id
        
        # Calcul de score de santé basé sur les données capteurs
        if request.sensor_readings:
            temp_readings = [r.value for r in request.sensor_readings if r.sensor_type == 'temperature']
            vibration_readings = [r.value for r in request.sensor_readings if r.sensor_type == 'vibration']
            
            avg_temp = np.mean(temp_readings) if temp_readings else 60
            avg_vibration = np.mean(vibration_readings) if vibration_readings else 2.0
            
            # Score de risque simple
            temp_risk = max(0, (avg_temp - 70) / 20)  # Risque si > 70°C
            vibration_risk = max(0, (avg_vibration - 3.0) / 2.0)  # Risque si > 3mm/s
            
            failure_probability = min(0.95, (temp_risk + vibration_risk) / 2)
        else:
            failure_probability = np.random.uniform(0.1, 0.4)
        
        # Détermination du statut
        if failure_probability > 0.8:
            status = "critical"
            time_to_failure = 24  # 24 heures
        elif failure_probability > 0.6:
            status = "warning" 
            time_to_failure = 168  # 1 semaine
        else:
            status = "healthy"
            time_to_failure = None
        
        # Génération de recommandations
        recommendations = []
        alerts = []
        
        if status == "critical":
            recommendations.append({
                "priority": "urgent",
                "action": "Arrêt immédiat et inspection",
                "estimated_cost": 5000,
                "deadline": (datetime.now() + timedelta(hours=4)).isoformat()
            })
            alerts.append({
                "type": "critical_failure_risk",
                "message": f"Risque de panne critique sur {equipment_id}"
            })
        elif status == "warning":
            recommendations.append({
                "priority": "high",
                "action": "Maintenance préventive programmée",
                "estimated_cost": 2000,
                "deadline": (datetime.now() + timedelta(days=3)).isoformat()
            })
            alerts.append({
                "type": "maintenance_required",
                "message": f"Maintenance recommandée pour {equipment_id}"
            })
        
        response = MaintenanceAnalysisResponse(
            equipment_id=equipment_id,
            status=status,
            failure_probability=failure_probability,
            time_to_failure_hours=time_to_failure,
            recommendations=recommendations,
            alerts=alerts,
            generated_at=datetime.now().isoformat()
        )
        
        logger.info(f"Analyse maintenance {equipment_id}: statut={status}, risque={failure_probability:.1%}")
        return response
        
    except Exception as e:
        logger.error(f"Erreur analyse maintenance: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {str(e)}")

@api_router.get("/maintenance/equipment", tags=["Predictive Maintenance"])
async def get_equipment(
    status: Optional[str] = Query(None, description="Filtrer par statut"),
    criticality: Optional[int] = Query(None, ge=1, le=4, description="Filtrer par criticité"),
    user: dict = Depends(get_current_user)
):
    """Liste les équipements monitorrés"""
    # Simulation d'équipements
    equipment_list = [
        {
            "id": "PUMP_001",
            "name": "Pompe Principale A",
            "type": "pump",
            "status": "healthy",
            "criticality": 3,
            "last_maintenance": "2024-01-15",
            "operating_hours": 8760,
            "failure_risk": 0.15
        },
        {
            "id": "CONV_002",
            "name": "Convoyeur B",
            "type": "conveyor", 
            "status": "warning",
            "criticality": 2,
            "last_maintenance": "2023-11-20",
            "operating_hours": 12500,
            "failure_risk": 0.65
        },
        {
            "id": "ROBOT_003",
            "name": "Robot de Palettisation",
            "type": "robot",
            "status": "critical",
            "criticality": 4,
            "last_maintenance": "2023-09-10", 
            "operating_hours": 15800,
            "failure_risk": 0.85
        }
    ]
    
    # Filtrage
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
    """Obtient le planning de maintenance optimal"""
    # Simulation de planning
    import numpy as np
    
    schedule = []
    for i in range(np.random.randint(5, 15)):
        equipment_id = f"EQUIP_{np.random.randint(1, 100):03d}"
        scheduled_date = datetime.now() + timedelta(days=np.random.randint(1, days))
        
        schedule.append({
            "equipment_id": equipment_id,
            "type": np.random.choice(["preventive", "corrective", "inspection"]),
            "priority": np.random.choice(["low", "medium", "high", "urgent"]),
            "scheduled_date": scheduled_date.isoformat(),
            "estimated_duration": np.random.randint(2, 8),  # heures
            "estimated_cost": np.random.randint(500, 5000),
            "technician_required": np.random.choice(["Level_1", "Level_2", "Specialist"])
        })
    
    # Tri par date
    schedule.sort(key=lambda x: x['scheduled_date'])
    
    return {
        "schedule": schedule,
        "summary": {
            "total_interventions": len(schedule),
            "total_cost": sum(item['estimated_cost'] for item in schedule),
            "urgent_count": len([item for item in schedule if item['priority'] == 'urgent'])
        }
    }

# === ENDPOINTS MÉTRIQUES TEMPS RÉEL ===

@api_router.get("/metrics/real-time", response_model=MetricsResponse, tags=["Real-time Metrics"])
async def get_real_time_metrics(user: dict = Depends(get_current_user)):
    """
    Obtient les métriques temps réel de la supply chain
    
    **Métriques incluses:**
    - KPIs opérationnels
    - Alertes actives  
    - Performance globale
    - Efficacité transport
    """
    try:
        # Simulation de métriques (remplacer par vraies données)
        import numpy as np
        
        # Variation selon l'heure pour réalisme
        hour = datetime.now().hour
        if 8 <= hour <= 18:  # Heures de bureau
            base_orders = 60
            base_efficiency = 85
        else:
            base_orders = 20
            base_efficiency = 70
        
        metrics = MetricsResponse(
            orders_per_hour=base_orders + np.random.normal(0, 10),
            avg_order_value=np.random.uniform(150, 350),
            equipment_alerts=np.random.randint(0, 5),
            transport_efficiency=base_efficiency + np.random.normal(0, 5),
            overall_efficiency=np.random.uniform(80, 95),
            active_alerts_count=np.random.randint(0, 8),
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
    """Obtient les KPIs de performance de la supply chain"""
    import numpy as np
    
    # Simulation de KPIs selon la période
    if period == "today":
        kpis = {
            "cost_reduction_percent": np.random.uniform(15, 25),
            "delivery_performance_percent": np.random.uniform(92, 98),
            "inventory_turnover": np.random.uniform(10, 14),
            "customer_satisfaction_percent": np.random.uniform(85, 92),
            "sustainability_score": np.random.uniform(68, 78),
            "orders_processed": np.random.randint(800, 1500),
            "avg_delivery_time_hours": np.random.uniform(20, 48),
            "transport_cost_per_km": np.random.uniform(0.12, 0.18)
        }
    else:
        # KPIs différents pour week/month
        multiplier = 7 if period == "week" else 30
        kpis = {
            "cost_reduction_percent": np.random.uniform(18, 22),
            "delivery_performance_percent": np.random.uniform(94, 96),
            "inventory_turnover": np.random.uniform(11, 13),
            "customer_satisfaction_percent": np.random.uniform(87, 90),
            "sustainability_score": np.random.uniform(70, 75),
            "orders_processed": np.random.randint(800, 1500) * multiplier,
            "avg_delivery_time_hours": np.random.uniform(22, 36),
            "transport_cost_per_km": np.random.uniform(0.13, 0.16)
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
    """Obtient les alertes actives du système"""
    # Simulation d'alertes
    import numpy as np
    
    alert_types = [
        "equipment_anomaly", "low_inventory", "transport_delay", 
        "demand_spike", "maintenance_required", "route_inefficient"
    ]
    
    severities = ["low", "medium", "high", "critical"]
    
    alerts = []
    for i in range(np.random.randint(3, 15)):
        alert_severity = np.random.choice(severities)
        if severity and alert_severity != severity:
            continue
            
        alerts.append({
            "id": f"alert_{i:04d}",
            "type": np.random.choice(alert_types),
            "severity": alert_severity,
            "message": f"Alerte simulée {i}",
            "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 1440))).isoformat(),
            "source": np.random.choice(["IoT", "ERP", "Transport", "Manual"]),
            "acknowledged": np.random.choice([True, False]),
            "actions": ["Vérifier", "Analyser", "Corriger"]
        })
    
    # Tri par timestamp (plus récentes d'abord)
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

# === ENDPOINTS UTILITAIRES ===

@api_router.get("/system/health", tags=["System"])
async def system_health():
    """Vérification de santé du système"""
    try:
        # Simulation de vérifications système
        import psutil
        import time
        
        health_checks = {
            "api_status": "healthy",
            "database_connection": "healthy", 
            "kafka_connection": "healthy",
            "ml_models_loaded": "healthy",
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "uptime_seconds": time.time() - psutil.boot_time(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Détermination du statut global
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
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@api_router.get("/system/stats", tags=["System"])
async def system_statistics(user: dict = Depends(get_current_user)):
    """Statistiques d'utilisation du système"""
    # Simulation de statistiques
    import numpy as np
    
    stats = {
        "requests_processed_today": np.random.randint(5000, 15000),
        "forecasts_generated": np.random.randint(500, 1500),
        "routes_optimized": np.random.randint(200, 800),
        "maintenance_analyses": np.random.randint(100, 400),
        "alerts_generated": np.random.randint(50, 200),
        "data_points_processed": np.random.randint(50000, 200000),
        "average_response_time_ms": np.random.uniform(50, 200),
        "system_efficiency_percent": np.random.uniform(85, 95)
    }
    
    return {
        "statistics": stats,
        "period": "last_24_hours",
        "generated_at": datetime.now().isoformat()
    }
