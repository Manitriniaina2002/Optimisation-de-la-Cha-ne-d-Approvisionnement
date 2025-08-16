"""
Ingestion de données temps réel avec Apache Kafka
Collecte et traitement des flux de données IoT, ERP, WMS, TMS
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
import uuid

# Kafka
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import kafka

# Configuration
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class DataMessage:
    """Message de données standardisé"""
    id: str
    source: str
    timestamp: datetime
    data_type: str
    payload: Dict[str, Any]
    quality_score: float = 1.0
    
    def to_json(self) -> str:
        """Sérialise en JSON"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DataMessage':
        """Désérialise depuis JSON"""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class KafkaProducerManager:
    """
    Gestionnaire de producteurs Kafka
    Publie les données vers différents topics selon leur type
    """
    
    def __init__(self):
        self.producer = None
        self.topics = {
            'demand': settings.KAFKA_TOPIC_DEMAND,
            'transport': settings.KAFKA_TOPIC_TRANSPORT,
            'maintenance': settings.KAFKA_TOPIC_MAINTENANCE,
            'risk': settings.KAFKA_TOPIC_RISK
        }
        
    async def initialize(self):
        """Initialise le producteur Kafka"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: v.encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Attendre confirmation de tous les replicas
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            logger.info("✅ Producteur Kafka initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation producteur Kafka: {e}")
            raise
    
    async def send_message(self, topic_type: str, message: DataMessage, key: str = None):
        """Envoie un message vers un topic spécifique"""
        if not self.producer:
            await self.initialize()
        
        topic = self.topics.get(topic_type)
        if not topic:
            logger.error(f"Topic inconnu: {topic_type}")
            return
        
        try:
            # Envoi du message
            future = self.producer.send(
                topic=topic,
                value=message.to_json(),
                key=key or message.id
            )
            
            # Attente de confirmation (non-bloquant)
            future.add_callback(lambda metadata: logger.debug(
                f"Message envoyé: topic={metadata.topic}, partition={metadata.partition}, offset={metadata.offset}"
            ))
            future.add_errback(lambda error: logger.error(f"Erreur envoi message: {error}"))
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi message vers {topic}: {e}")
    
    def close(self):
        """Ferme le producteur"""
        if self.producer:
            self.producer.close()
            logger.info("Producteur Kafka fermé")

class KafkaConsumerManager:
    """
    Gestionnaire de consommateurs Kafka
    Écoute les différents topics et route les messages
    """
    
    def __init__(self, stream_processor=None):
        self.consumers = {}
        self.running = False
        self.stream_processor = stream_processor
        self.message_handlers = {}
        
    async def initialize(self):
        """Initialise les consommateurs pour chaque topic"""
        try:
            topics = settings.get_kafka_topics()
            
            for topic in topics:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                    value_deserializer=lambda m: m.decode('utf-8'),
                    key_deserializer=lambda k: k.decode('utf-8') if k else None,
                    group_id=f'supply_chain_consumer_{topic}',
                    enable_auto_commit=True,
                    auto_offset_reset='latest'
                )
                
                self.consumers[topic] = consumer
                logger.info(f"✅ Consommateur créé pour topic: {topic}")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation consommateurs: {e}")
            raise
    
    def register_handler(self, topic_pattern: str, handler: Callable[[DataMessage], Any]):
        """Enregistre un handler pour un type de topic"""
        self.message_handlers[topic_pattern] = handler
    
    async def start(self):
        """Démarre l'écoute des topics"""
        if not self.consumers:
            await self.initialize()
        
        self.running = True
        
        # Création des tâches d'écoute pour chaque topic
        tasks = []
        for topic, consumer in self.consumers.items():
            task = asyncio.create_task(self._consume_topic(topic, consumer))
            tasks.append(task)
        
        logger.info(f"🎧 Écoute démarrée sur {len(tasks)} topics")
        
        # Attendre que toutes les tâches se terminent
        await asyncio.gather(*tasks)
    
    async def _consume_topic(self, topic: str, consumer: KafkaConsumer):
        """Consomme les messages d'un topic spécifique"""
        logger.info(f"🎯 Écoute démarrée sur topic: {topic}")
        
        while self.running:
            try:
                # Poll pour nouveaux messages (timeout 1 seconde)
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._process_message(topic, message)
                
                # Sleep pour éviter une boucle trop intensive
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Erreur consommation topic {topic}: {e}")
                await asyncio.sleep(1)  # Pause avant retry
    
    async def _process_message(self, topic: str, kafka_message):
        """Traite un message individuel"""
        try:
            # Désérialisation du message
            data_message = DataMessage.from_json(kafka_message.value)
            
            logger.debug(f"📨 Message reçu: topic={topic}, source={data_message.source}")
            
            # Routage vers le handler approprié
            handler = None
            for pattern, h in self.message_handlers.items():
                if pattern in topic:
                    handler = h
                    break
            
            if handler:
                await handler(data_message)
            elif self.stream_processor:
                await self.stream_processor.process_message(data_message)
            else:
                logger.warning(f"Aucun handler pour topic: {topic}")
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
    
    async def stop(self):
        """Arrête l'écoute des topics"""
        self.running = False
        
        for topic, consumer in self.consumers.items():
            consumer.close()
            logger.info(f"Consommateur fermé pour topic: {topic}")
        
        logger.info("🛑 Tous les consommateurs Kafka fermés")

class IoTDataSimulator:
    """
    Simulateur de données IoT pour tests et démonstration
    Génère des données réalistes de capteurs
    """
    
    def __init__(self, producer_manager: KafkaProducerManager):
        self.producer = producer_manager
        self.running = False
        
    async def start_simulation(self):
        """Démarre la simulation de données IoT"""
        self.running = True
        
        # Lancement des simulateurs en parallèle
        tasks = [
            asyncio.create_task(self._simulate_sensor_data()),
            asyncio.create_task(self._simulate_erp_data()),
            asyncio.create_task(self._simulate_transport_data()),
            asyncio.create_task(self._simulate_demand_data())
        ]
        
        logger.info("🔄 Simulation de données IoT démarrée")
        await asyncio.gather(*tasks)
    
    async def _simulate_sensor_data(self):
        """Simule des données de capteurs IoT"""
        equipment_ids = ['PUMP_001', 'CONV_002', 'ROBOT_003', 'PUMP_004', 'CRANE_005']
        
        while self.running:
            try:
                for equipment_id in equipment_ids:
                    # Simulation de données de capteurs
                    sensor_data = {
                        'equipment_id': equipment_id,
                        'temperature': 60 + np.random.normal(0, 5),
                        'vibration': 2.0 + np.random.normal(0, 0.5),
                        'pressure': 5.0 + np.random.normal(0, 0.8),
                        'current': 10.0 + np.random.normal(0, 1.5),
                        'operating_hours': np.random.randint(8500, 12000),
                        'location': f'Zone_{equipment_id[0]}'
                    }
                    
                    # Simulation d'anomalies occasionnelles
                    if np.random.random() < 0.05:  # 5% de chance
                        sensor_data['temperature'] += 20  # Surchauffe
                        sensor_data['vibration'] *= 2     # Vibration excessive
                    
                    message = DataMessage(
                        id=str(uuid.uuid4()),
                        source='iot_sensors',
                        timestamp=datetime.now(),
                        data_type='sensor_reading',
                        payload=sensor_data,
                        quality_score=np.random.uniform(0.95, 1.0)
                    )
                    
                    await self.producer.send_message('maintenance', message, equipment_id)
                
                # Pause entre les envois
                await asyncio.sleep(60)  # Données toutes les minutes
                
            except Exception as e:
                logger.error(f"❌ Erreur simulation capteurs: {e}")
                await asyncio.sleep(5)
    
    async def _simulate_erp_data(self):
        """Simule des données ERP (commandes, stocks, finances)"""
        products = ['PROD_A', 'PROD_B', 'PROD_C', 'PROD_D']
        
        while self.running:
            try:
                for product in products:
                    # Simulation de transaction ERP
                    erp_data = {
                        'product_id': product,
                        'transaction_type': np.random.choice(['order', 'inventory_update', 'shipment']),
                        'quantity': np.random.randint(10, 1000),
                        'unit_price': np.random.uniform(10, 100),
                        'warehouse': f'WH_{np.random.randint(1, 5)}',
                        'customer_id': f'CUST_{np.random.randint(1000, 9999)}',
                        'priority': np.random.choice(['normal', 'high', 'urgent'])
                    }
                    
                    message = DataMessage(
                        id=str(uuid.uuid4()),
                        source='erp_system',
                        timestamp=datetime.now(),
                        data_type='transaction',
                        payload=erp_data
                    )
                    
                    await self.producer.send_message('demand', message, product)
                
                await asyncio.sleep(30)  # Données toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"❌ Erreur simulation ERP: {e}")
                await asyncio.sleep(5)
    
    async def _simulate_transport_data(self):
        """Simule des données de transport (GPS, livraisons, véhicules)"""
        vehicle_ids = ['VEH_001', 'VEH_002', 'VEH_003', 'VEH_004']
        
        while self.running:
            try:
                for vehicle_id in vehicle_ids:
                    # Position GPS simulée (région Lyon)
                    base_lat, base_lon = 45.764, 4.835
                    lat = base_lat + np.random.normal(0, 0.1)
                    lon = base_lon + np.random.normal(0, 0.1)
                    
                    transport_data = {
                        'vehicle_id': vehicle_id,
                        'latitude': lat,
                        'longitude': lon,
                        'speed_kmh': np.random.uniform(0, 90),
                        'fuel_level': np.random.uniform(20, 100),
                        'cargo_weight': np.random.uniform(0, 1000),
                        'destination': f'DEST_{np.random.randint(1, 20)}',
                        'eta_minutes': np.random.randint(15, 180),
                        'driver_id': f'DRV_{np.random.randint(100, 999)}',
                        'route_efficiency': np.random.uniform(0.7, 1.0)
                    }
                    
                    message = DataMessage(
                        id=str(uuid.uuid4()),
                        source='vehicle_tracking',
                        timestamp=datetime.now(),
                        data_type='gps_position',
                        payload=transport_data
                    )
                    
                    await self.producer.send_message('transport', message, vehicle_id)
                
                await asyncio.sleep(15)  # Données toutes les 15 secondes
                
            except Exception as e:
                logger.error(f"❌ Erreur simulation transport: {e}")
                await asyncio.sleep(5)
    
    async def _simulate_demand_data(self):
        """Simule des données de demande client"""
        while self.running:
            try:
                # Simulation de commandes clients
                demand_data = {
                    'order_id': f'ORD_{np.random.randint(10000, 99999)}',
                    'customer_id': f'CUST_{np.random.randint(1000, 9999)}',
                    'product_id': f'PROD_{np.random.choice(["A", "B", "C", "D"])}',
                    'quantity': np.random.randint(1, 50),
                    'order_value': np.random.uniform(50, 5000),
                    'delivery_date_requested': (datetime.now() + timedelta(days=np.random.randint(1, 14))).isoformat(),
                    'priority': np.random.choice(['standard', 'express', 'next_day']),
                    'channel': np.random.choice(['online', 'retail', 'b2b']),
                    'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'])
                }
                
                message = DataMessage(
                    id=str(uuid.uuid4()),
                    source='customer_orders',
                    timestamp=datetime.now(),
                    data_type='order',
                    payload=demand_data
                )
                
                await self.producer.send_message('demand', message, demand_data['order_id'])
                
                # Fréquence variable selon l'heure (plus de commandes en journée)
                hour = datetime.now().hour
                if 8 <= hour <= 18:  # Heures de bureau
                    await asyncio.sleep(np.random.uniform(5, 15))
                else:
                    await asyncio.sleep(np.random.uniform(30, 60))
                
            except Exception as e:
                logger.error(f"❌ Erreur simulation demande: {e}")
                await asyncio.sleep(10)
    
    def stop(self):
        """Arrête la simulation"""
        self.running = False
        logger.info("🛑 Simulation de données IoT arrêtée")

# Import numpy pour les simulations
import numpy as np

async def test_kafka_system():
    """Fonction de test du système Kafka"""
    logger.info("🧪 Test du système Kafka")
    
    # Initialisation du producteur
    producer = KafkaProducerManager()
    await producer.initialize()
    
    # Test d'envoi de message
    test_message = DataMessage(
        id=str(uuid.uuid4()),
        source='test_system',
        timestamp=datetime.now(),
        data_type='test',
        payload={'message': 'Hello Kafka!', 'value': 42}
    )
    
    await producer.send_message('demand', test_message)
    logger.info("✅ Message de test envoyé")
    
    # Nettoyage
    producer.close()

if __name__ == '__main__':
    asyncio.run(test_kafka_system())
