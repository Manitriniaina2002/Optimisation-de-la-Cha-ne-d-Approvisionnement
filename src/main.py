"""
Application principale Big Data Supply Chain
Orchestrateur des différents modules d'optimisation
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from config.settings import settings
from api.routes import api_router
from data_ingestion.kafka_consumer import KafkaConsumerManager
from real_time.stream_processor import StreamProcessor
from ml_models.demand_forecasting import DemandForecaster
from optimization.transport_optimizer import TransportOptimizer
from ml_models.predictive_maintenance import MaintenancePredictor
from dashboard.app import create_dash_app
from utils.logger import setup_logger

# Configuration du logging
logger = setup_logger(__name__)

class SupplyChainApplication:
    """Application principale de gestion de la chaîne d'approvisionnement"""
    
    def __init__(self):
        self.kafka_consumer = None
        self.stream_processor = None
        self.demand_forecaster = None
        self.transport_optimizer = None
        self.maintenance_predictor = None
        self.executor = ThreadPoolExecutor(max_workers=settings.OPTIMIZATION_THREADS)
        
    async def initialize_services(self):
        """Initialise tous les services de l'application"""
        try:
            logger.info("Initialisation des services Big Data Supply Chain...")
            
            # Initialisation des modèles ML
            self.demand_forecaster = DemandForecaster()
            await self.demand_forecaster.load_models()
            
            self.transport_optimizer = TransportOptimizer()
            await self.transport_optimizer.initialize()
            
            self.maintenance_predictor = MaintenancePredictor()
            await self.maintenance_predictor.load_models()
            
            # Initialisation du traitement temps réel
            self.stream_processor = StreamProcessor(
                demand_forecaster=self.demand_forecaster,
                transport_optimizer=self.transport_optimizer,
                maintenance_predictor=self.maintenance_predictor
            )
            
            # Initialisation des consommateurs Kafka
            self.kafka_consumer = KafkaConsumerManager(
                stream_processor=self.stream_processor
            )
            
            logger.info("✅ Tous les services ont été initialisés avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des services: {e}")
            raise
    
    async def start_background_services(self):
        """Démarre les services en arrière-plan"""
        try:
            # Démarrage des consommateurs Kafka
            await self.kafka_consumer.start()
            
            # Démarrage du processeur de flux temps réel
            await self.stream_processor.start()
            
            logger.info("🚀 Services en arrière-plan démarrés")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage des services: {e}")
            raise
    
    async def stop_services(self):
        """Arrête tous les services proprement"""
        try:
            if self.kafka_consumer:
                await self.kafka_consumer.stop()
                
            if self.stream_processor:
                await self.stream_processor.stop()
                
            self.executor.shutdown(wait=True)
            
            logger.info("⏹️ Tous les services ont été arrêtés")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt des services: {e}")

# Instance globale de l'application
supply_chain_app = SupplyChainApplication()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application FastAPI"""
    # Startup
    await supply_chain_app.initialize_services()
    await supply_chain_app.start_background_services()
    yield
    # Shutdown
    await supply_chain_app.stop_services()

# Création de l'application FastAPI
app = FastAPI(
    title="Big Data Supply Chain Optimization",
    description="API complète pour l'optimisation de la chaîne d'approvisionnement avec Big Data",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ajout des routes API
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Endpoint racine avec informations sur l'API"""
    return {
        "message": "Big Data Supply Chain Optimization API",
        "version": "1.0.0",
        "features": [
            "Prévision intelligente de la demande",
            "Optimisation temps réel des transports", 
            "Maintenance prédictive",
            "Gestion des risques",
            "Traitement Big Data en temps réel"
        ],
        "endpoints": {
            "api": "/api/v1",
            "docs": "/docs",
            "dashboard": f"http://localhost:{settings.DASH_PORT}"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de santé de l'application"""
    try:
        # Vérification des services critiques
        services_status = {
            "demand_forecaster": supply_chain_app.demand_forecaster is not None,
            "transport_optimizer": supply_chain_app.transport_optimizer is not None,
            "maintenance_predictor": supply_chain_app.maintenance_predictor is not None,
            "stream_processor": supply_chain_app.stream_processor is not None,
            "kafka_consumer": supply_chain_app.kafka_consumer is not None
        }
        
        all_healthy = all(services_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": services_status,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de santé: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }

async def start_dash_application():
    """Démarre l'application Dash pour le dashboard"""
    try:
        dash_app = create_dash_app()
        logger.info(f"🎯 Dashboard Dash démarré sur http://localhost:{settings.DASH_PORT}")
        dash_app.run_server(
            host=settings.DASH_HOST,
            port=settings.DASH_PORT,
            debug=settings.DASH_DEBUG
        )
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage du dashboard Dash: {e}")

if __name__ == "__main__":
    logger.info("🚀 Démarrage de l'application Big Data Supply Chain")
    
    try:
        # Démarrage du dashboard Dash en parallèle
        import threading
        dash_thread = threading.Thread(target=asyncio.run, args=(start_dash_application(),))
        dash_thread.daemon = True
        dash_thread.start()
        
        # Démarrage de l'API FastAPI
        uvicorn.run(
            "main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Arrêt de l'application demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        raise
    finally:
        logger.info("🏁 Application arrêtée")
