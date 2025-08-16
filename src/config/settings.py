"""
Configuration centrale de l'application Big Data Supply Chain
Gestion des paramètres et variables d'environnement
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Configuration principale de l'application"""
    
    # Application
    ENVIRONMENT: str = Field(default="development", description="Environnement d'exécution")
    DEBUG: bool = Field(default=True, description="Mode debug")
    TESTING: bool = Field(default=False, description="Mode test")
    
    # Base de données
    DATABASE_URL: str = Field(default="postgresql://user:password@localhost:5432/supply_chain_db")
    MONGODB_URL: str = Field(default="mongodb://localhost:27017/supply_chain_mongo")
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    
    # APIs externes
    WEATHER_API_KEY: Optional[str] = Field(default=None, description="Clé API météo")
    GOOGLE_MAPS_API_KEY: Optional[str] = Field(default=None, description="Clé API Google Maps")
    TRAFFIC_API_KEY: Optional[str] = Field(default=None, description="Clé API trafic")
    
    # Services Cloud
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None)
    AWS_REGION: str = Field(default="us-east-1")
    S3_BUCKET_NAME: Optional[str] = Field(default="supply-chain-data")
    
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = Field(default=None)
    AZURE_CONTAINER_NAME: Optional[str] = Field(default="supply-chain-container")
    
    # IoT et capteurs
    MQTT_BROKER_HOST: str = Field(default="localhost")
    MQTT_BROKER_PORT: int = Field(default=1883)
    MQTT_USERNAME: Optional[str] = Field(default=None)
    MQTT_PASSWORD: Optional[str] = Field(default=None)
    
    INFLUXDB_URL: str = Field(default="http://localhost:8086")
    INFLUXDB_TOKEN: Optional[str] = Field(default=None)
    INFLUXDB_ORG: Optional[str] = Field(default="supply_chain_org")
    INFLUXDB_BUCKET: str = Field(default="supply_chain_metrics")
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = Field(default="localhost:9092")
    KAFKA_TOPIC_DEMAND: str = Field(default="demand_forecasting")
    KAFKA_TOPIC_TRANSPORT: str = Field(default="transport_optimization")
    KAFKA_TOPIC_MAINTENANCE: str = Field(default="predictive_maintenance")
    KAFKA_TOPIC_RISK: str = Field(default="risk_management")
    
    # Machine Learning
    ML_MODEL_PATH: str = Field(default="models/")
    PROPHET_MODEL_PATH: str = Field(default="models/prophet/")
    XGBOOST_MODEL_PATH: str = Field(default="models/xgboost/")
    LSTM_MODEL_PATH: str = Field(default="models/lstm/")
    
    # Dashboard et API
    DASH_HOST: str = Field(default="0.0.0.0")
    DASH_PORT: int = Field(default=8050)
    DASH_DEBUG: bool = Field(default=True)
    
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_SECRET_KEY: str = Field(default="your_super_secret_key_here")
    
    # Monitoring
    PROMETHEUS_PORT: int = Field(default=8001)
    LOG_LEVEL: str = Field(default="INFO")
    
    # Notifications
    TWILIO_ACCOUNT_SID: Optional[str] = Field(default=None)
    TWILIO_AUTH_TOKEN: Optional[str] = Field(default=None)
    TWILIO_PHONE_NUMBER: Optional[str] = Field(default=None)
    
    SMTP_SERVER: str = Field(default="smtp.gmail.com")
    SMTP_PORT: int = Field(default=587)
    SMTP_USERNAME: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    
    # Sécurité
    JWT_SECRET_KEY: str = Field(default="your_jwt_secret_key")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_HOURS: int = Field(default=24)
    
    # Optimisation
    OPTIMIZATION_THREADS: int = Field(default=4, description="Nombre de threads pour l'optimisation")
    BATCH_SIZE: int = Field(default=1000, description="Taille des lots de traitement")
    MAX_ITERATIONS: int = Field(default=1000, description="Nombre max d'itérations pour l'optimisation")
    
    # Cache
    CACHE_TTL_SECONDS: int = Field(default=3600, description="TTL du cache en secondes")
    CACHE_MAX_SIZE: int = Field(default=1000, description="Taille maximale du cache")
    
    # Prévision de demande
    FORECAST_HORIZON_DAYS: int = Field(default=30, description="Horizon de prévision en jours")
    FORECAST_UPDATE_INTERVAL_HOURS: int = Field(default=6, description="Intervalle de mise à jour des prévisions")
    
    # Transport et optimisation
    MAX_VEHICLE_CAPACITY: float = Field(default=1000.0, description="Capacité maximale des véhicules")
    MAX_ROUTE_DISTANCE_KM: float = Field(default=500.0, description="Distance maximale des routes")
    FUEL_COST_PER_KM: float = Field(default=0.15, description="Coût carburant par km")
    CO2_EMISSION_KG_PER_KM: float = Field(default=0.25, description="Émission CO2 par km")
    
    # Maintenance prédictive
    MAINTENANCE_CHECK_INTERVAL_HOURS: int = Field(default=1, description="Intervalle de vérification maintenance")
    CRITICAL_THRESHOLD: float = Field(default=0.8, description="Seuil critique pour alertes")
    WARNING_THRESHOLD: float = Field(default=0.6, description="Seuil d'avertissement")
    
    # Gestion des risques
    RISK_ASSESSMENT_INTERVAL_HOURS: int = Field(default=4, description="Intervalle d'évaluation des risques")
    SUPPLIER_MONITORING_ENABLED: bool = Field(default=True, description="Surveillance des fournisseurs")
    WEATHER_MONITORING_ENABLED: bool = Field(default=True, description="Surveillance météorologique")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_database_url(self) -> str:
        """Retourne l'URL de la base de données configurée"""
        return self.DATABASE_URL
    
    def get_kafka_topics(self) -> List[str]:
        """Retourne la liste des topics Kafka"""
        return [
            self.KAFKA_TOPIC_DEMAND,
            self.KAFKA_TOPIC_TRANSPORT,
            self.KAFKA_TOPIC_MAINTENANCE,
            self.KAFKA_TOPIC_RISK
        ]
    
    def get_model_paths(self) -> dict:
        """Retourne les chemins des modèles ML"""
        base_path = Path(self.ML_MODEL_PATH)
        return {
            "prophet": base_path / "prophet",
            "xgboost": base_path / "xgboost", 
            "lstm": base_path / "lstm",
            "maintenance": base_path / "maintenance",
            "risk": base_path / "risk"
        }
    
    def is_production(self) -> bool:
        """Vérifie si l'environnement est en production"""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Vérifie si l'environnement est en développement"""
        return self.ENVIRONMENT.lower() == "development"

# Instance globale des paramètres
settings = Settings()

# Configuration des constantes métier
class BusinessConstants:
    """Constantes métier pour la supply chain"""
    
    # Seuils de performance
    PERFECT_DELIVERY_RATE_TARGET = 0.95  # 95% de livraisons parfaites
    STOCK_TURNOVER_TARGET = 12  # 12 rotations par an
    COST_REDUCTION_TARGET = 0.20  # 20% de réduction des coûts
    
    # Délais standards
    SUPPLIER_LEAD_TIME_DAYS = 7
    PRODUCTION_LEAD_TIME_DAYS = 3
    TRANSPORT_LEAD_TIME_DAYS = 2
    
    # Capacités
    WAREHOUSE_CAPACITY_UTILIZATION_MAX = 0.85  # 85% max
    VEHICLE_CAPACITY_UTILIZATION_TARGET = 0.80  # 80% cible
    
    # Coûts
    STORAGE_COST_PER_UNIT_PER_DAY = 0.10
    SHORTAGE_COST_MULTIPLIER = 5.0  # 5x le coût normal
    OBSOLESCENCE_COST_RATE = 0.02  # 2% par mois
    
    # Qualité
    DEFECT_RATE_THRESHOLD = 0.01  # 1% max
    CUSTOMER_SATISFACTION_TARGET = 0.90  # 90% min
    
    # Environnement
    CO2_REDUCTION_TARGET = 0.30  # 30% de réduction CO2
    RENEWABLE_ENERGY_TARGET = 0.50  # 50% d'énergie renouvelable

# Constantes techniques
class TechnicalConstants:
    """Constantes techniques pour le système"""
    
    # Traitement de données
    MAX_BATCH_SIZE = 10000
    STREAM_BUFFER_SIZE = 1000
    MAX_MEMORY_USAGE_MB = 2048
    
    # Machine Learning
    MODEL_RETRAIN_THRESHOLD = 0.05  # 5% de dégradation
    FEATURE_IMPORTANCE_THRESHOLD = 0.01
    CROSS_VALIDATION_FOLDS = 5
    
    # API et performance
    MAX_REQUEST_SIZE_MB = 100
    REQUEST_TIMEOUT_SECONDS = 30
    RATE_LIMIT_PER_MINUTE = 1000
    
    # Sécurité
    PASSWORD_MIN_LENGTH = 8
    SESSION_TIMEOUT_MINUTES = 30
    MAX_LOGIN_ATTEMPTS = 5
