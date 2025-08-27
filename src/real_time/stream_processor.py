"""
Processeur de flux temps réel
Traite les données en streaming et déclenche les actions d'optimisation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

# Configuration
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.data_ingestion.kafka_consumer import DataMessage

logger = setup_logger(__name__)

@dataclass
class StreamAlert:
    """Alerte générée par le processeur de flux"""
    id: str
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    source_data: Dict[str, Any]
    timestamp: datetime
    recommended_actions: List[str]

class StreamProcessor:
    """
    Processeur de flux temps réel pour la supply chain
    
    Features:
    - Traitement en streaming des données IoT, ERP, transport
    - Détection d'anomalies en temps réel
    - Déclenchement d'optimisations automatiques
    - Génération d'alertes intelligentes
    - Agrégation de métriques temps réel
    """
    
    def __init__(self, demand_forecaster=None, transport_optimizer=None, maintenance_predictor=None):
        self.demand_forecaster = demand_forecaster
        self.transport_optimizer = transport_optimizer
        self.maintenance_predictor = maintenance_predictor
        
        # Buffers pour données temps réel
        self.sensor_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.order_buffer = deque(maxlen=5000)
        self.transport_buffer = defaultdict(lambda: deque(maxlen=500))
        
        # Métriques temps réel
        self.real_time_metrics = {
            'orders_per_hour': 0,
            'avg_order_value': 0,
            'equipment_alerts': 0,
            'transport_efficiency': 0,
            'demand_forecast_accuracy': 0
        }
        
        # Alertes actives
        self.active_alerts = {}
        
        # État du processeur
        self.running = False
        self.last_processing_time = datetime.now()
        
        logger.info("🌊 StreamProcessor initialisé")
    
    async def start(self):
        """Démarre le processeur de flux"""
        self.running = True
        
        # Lancement des tâches de traitement en parallèle
        tasks = [
            asyncio.create_task(self._process_metrics_aggregation()),
            asyncio.create_task(self._process_anomaly_detection()),
            asyncio.create_task(self._process_optimization_triggers()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        logger.info("🚀 StreamProcessor démarré")
        await asyncio.gather(*tasks)
    
    async def process_message(self, message: DataMessage):
        """
        Traite un message de données en temps réel
        Point d'entrée principal pour tous les messages Kafka
        """
        try:
            self.last_processing_time = datetime.now()
            
            # Routage selon le type de données
            if message.data_type == 'sensor_reading':
                await self._process_sensor_data(message)
            elif message.data_type == 'order':
                await self._process_order_data(message)
            elif message.data_type == 'gps_position':
                await self._process_transport_data(message)
            elif message.data_type == 'transaction':
                await self._process_erp_data(message)
            else:
                logger.warning(f"Type de message non reconnu: {message.data_type}")
            
            # Mise à jour des métriques globales
            await self._update_global_metrics(message)
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
    
    async def _process_sensor_data(self, message: DataMessage):
        """Traite les données de capteurs IoT"""
        try:
            payload = message.payload
            equipment_id = payload.get('equipment_id')
            
            if not equipment_id:
                return
            
            # Ajout au buffer
            self.sensor_buffer[equipment_id].append({
                'timestamp': message.timestamp,
                'data': payload,
                'quality': message.quality_score
            })
            
            # Détection d'anomalies sur équipement spécifique
            await self._check_equipment_anomalies(equipment_id, payload)
            
            # Prédiction de maintenance si assez de données
            if len(self.sensor_buffer[equipment_id]) >= 10:
                await self._trigger_maintenance_analysis(equipment_id)
            
            logger.debug(f"📡 Données capteur traitées: {equipment_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement capteur: {e}")
    
    async def _process_order_data(self, message: DataMessage):
        """Traite les données de commandes"""
        try:
            payload = message.payload
            
            # Ajout au buffer
            self.order_buffer.append({
                'timestamp': message.timestamp,
                'data': payload
            })
            
            # Mise à jour des prévisions de demande
            await self._update_demand_forecast(payload)
            
            # Vérification des seuils de stock
            await self._check_inventory_levels(payload)
            
            logger.debug(f"🛒 Commande traitée: {payload.get('order_id')}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement commande: {e}")
    
    async def _process_transport_data(self, message: DataMessage):
        """Traite les données de transport/GPS"""
        try:
            payload = message.payload
            vehicle_id = payload.get('vehicle_id')
            
            if not vehicle_id:
                return
            
            # Ajout au buffer
            self.transport_buffer[vehicle_id].append({
                'timestamp': message.timestamp,
                'data': payload
            })
            
            # Vérification de l'efficacité du transport
            await self._check_transport_efficiency(vehicle_id, payload)
            
            # Optimisation de route si déviation détectée
            await self._check_route_optimization(vehicle_id, payload)
            
            logger.debug(f"🚚 Position GPS mise à jour: {vehicle_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement transport: {e}")
    
    async def _process_erp_data(self, message: DataMessage):
        """Traite les données ERP"""
        try:
            payload = message.payload
            transaction_type = payload.get('transaction_type')
            
            if transaction_type == 'inventory_update':
                await self._process_inventory_update(payload)
            elif transaction_type == 'shipment':
                await self._process_shipment_data(payload)
            
            logger.debug(f"💼 Transaction ERP traitée: {transaction_type}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement ERP: {e}")
    
    async def _check_equipment_anomalies(self, equipment_id: str, sensor_data: Dict[str, Any]):
        """Vérifie les anomalies sur un équipement"""
        try:
            # Seuils d'alerte simples (en réalité, utiliserait des modèles ML)
            temperature = sensor_data.get('temperature', 0)
            vibration = sensor_data.get('vibration', 0)
            pressure = sensor_data.get('pressure', 0)
            
            alerts = []
            
            # Température excessive
            if temperature > 85:
                alerts.append({
                    'type': 'temperature_high',
                    'severity': 'high' if temperature > 90 else 'medium',
                    'message': f'Température élevée détectée: {temperature:.1f}°C',
                    'actions': ['Vérifier refroidissement', 'Planifier maintenance']
                })
            
            # Vibrations anormales
            if vibration > 4.0:
                alerts.append({
                    'type': 'vibration_high',
                    'severity': 'high' if vibration > 6.0 else 'medium',
                    'message': f'Vibrations excessives détectées: {vibration:.1f}mm/s',
                    'actions': ['Vérifier alignement', 'Inspecter roulements']
                })
            
            # Pression anormale
            if pressure < 2.0 or pressure > 8.0:
                alerts.append({
                    'type': 'pressure_abnormal',
                    'severity': 'medium',
                    'message': f'Pression anormale: {pressure:.1f}bar',
                    'actions': ['Vérifier circuit hydraulique']
                })
            
            # Génération des alertes
            for alert_data in alerts:
                alert = StreamAlert(
                    id=f"{equipment_id}_{alert_data['type']}_{int(datetime.now().timestamp())}",
                    type=alert_data['type'],
                    severity=alert_data['severity'],
                    message=f"[{equipment_id}] {alert_data['message']}",
                    source_data={'equipment_id': equipment_id, **sensor_data},
                    timestamp=datetime.now(),
                    recommended_actions=alert_data['actions']
                )
                
                await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification anomalies: {e}")
    
    async def _trigger_maintenance_analysis(self, equipment_id: str):
        """Déclenche une analyse de maintenance prédictive"""
        try:
            if not self.maintenance_predictor:
                return
            
            # Récupération des données récentes
            recent_data = list(self.sensor_buffer[equipment_id])[-24:]  # 24 dernières mesures
            
            if len(recent_data) < 10:
                return
            
            # Simulation d'équipement pour l'analyse
            from src.ml_models.predictive_maintenance import Equipment
            equipment = Equipment(
                id=equipment_id,
                name=f"Equipment {equipment_id}",
                type='pump',  # Type par défaut
                location='Zone_A',
                installation_date=datetime.now() - timedelta(days=365),
                last_maintenance=datetime.now() - timedelta(days=90),
                criticality_level=2,
                operating_hours=8760,
                manufacturer='Generic',
                model='Model_X'
            )
            
            # Conversion en SensorReading
            from src.ml_models.predictive_maintenance import SensorReading
            sensor_readings = []
            
            for reading in recent_data:
                data = reading['data']
                for sensor_type, value in data.items():
                    if sensor_type in ['temperature', 'vibration', 'pressure', 'current']:
                        sensor_readings.append(SensorReading(
                            equipment_id=equipment_id,
                            timestamp=reading['timestamp'],
                            sensor_type=sensor_type,
                            value=float(value),
                            unit='°C' if sensor_type == 'temperature' else 'unit',
                            quality=reading['quality']
                        ))
            
            # Analyse de maintenance
            analysis = await self.maintenance_predictor.analyze_equipment(equipment, sensor_readings)
            
            # Génération d'alertes si nécessaire
            if analysis.get('status') in ['warning', 'critical']:
                alert = StreamAlert(
                    id=f"maintenance_{equipment_id}_{int(datetime.now().timestamp())}",
                    type='maintenance_required',
                    severity=analysis.get('status', 'medium'),
                    message=f"Maintenance recommandée pour {equipment_id}",
                    source_data=analysis,
                    timestamp=datetime.now(),
                    recommended_actions=[rec.get('action', 'Vérifier équipement') 
                                       for rec in analysis.get('recommendations', [])]
                )
                
                await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse maintenance: {e}")
    
    async def _update_demand_forecast(self, order_data: Dict[str, Any]):
        """Met à jour les prévisions de demande avec nouvelle commande"""
        try:
            product_id = order_data.get('product_id')
            quantity = order_data.get('quantity', 0)
            
            if not product_id or not self.demand_forecaster:
                return
            
            # Calcul de la demande horaire actuelle
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            hourly_orders = [
                order for order in self.order_buffer
                if order['timestamp'].replace(minute=0, second=0, microsecond=0) == current_hour
                and order['data'].get('product_id') == product_id
            ]
            
            hourly_demand = sum(order['data'].get('quantity', 0) for order in hourly_orders)
            
            # Alerte si demande inhabituelle
            if len(hourly_orders) > 0:
                avg_quantity = hourly_demand / len(hourly_orders)
                if avg_quantity > 100:  # Seuil arbitraire
                    alert = StreamAlert(
                        id=f"demand_spike_{product_id}_{int(datetime.now().timestamp())}",
                        type='demand_spike',
                        severity='medium',
                        message=f"Pic de demande détecté pour {product_id}: {hourly_demand} unités/heure",
                        source_data={'product_id': product_id, 'hourly_demand': hourly_demand},
                        timestamp=datetime.now(),
                        recommended_actions=['Vérifier stock disponible', 'Alerter production']
                    )
                    
                    await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour prévision: {e}")
    
    async def _check_inventory_levels(self, order_data: Dict[str, Any]):
        """Vérifie les niveaux de stock suite à une commande"""
        try:
            product_id = order_data.get('product_id')
            quantity = order_data.get('quantity', 0)
            
            # Simulation de vérification de stock
            # En réalité, cela interrogerait la base de données
            simulated_stock = np.random.randint(50, 500)
            
            if simulated_stock < quantity * 5:  # Moins de 5 fois la commande
                alert = StreamAlert(
                    id=f"low_stock_{product_id}_{int(datetime.now().timestamp())}",
                    type='low_inventory',
                    severity='high' if simulated_stock < quantity * 2 else 'medium',
                    message=f"Stock faible pour {product_id}: {simulated_stock} unités restantes",
                    source_data={'product_id': product_id, 'current_stock': simulated_stock},
                    timestamp=datetime.now(),
                    recommended_actions=['Déclencher réapprovisionnement', 'Alerter fournisseur']
                )
                
                await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification stock: {e}")
    
    async def _check_transport_efficiency(self, vehicle_id: str, transport_data: Dict[str, Any]):
        """Vérifie l'efficacité du transport"""
        try:
            route_efficiency = transport_data.get('route_efficiency', 1.0)
            fuel_level = transport_data.get('fuel_level', 100)
            speed = transport_data.get('speed_kmh', 0)
            
            alerts = []
            
            # Efficacité de route faible
            if route_efficiency < 0.7:
                alerts.append({
                    'type': 'route_inefficient',
                    'severity': 'medium',
                    'message': f'Efficacité de route faible: {route_efficiency:.1%}',
                    'actions': ['Recalculer itinéraire', 'Vérifier trafic']
                })
            
            # Niveau de carburant bas
            if fuel_level < 20:
                alerts.append({
                    'type': 'low_fuel',
                    'severity': 'high' if fuel_level < 10 else 'medium',
                    'message': f'Niveau de carburant bas: {fuel_level:.1f}%',
                    'actions': ['Localiser station-service', 'Ajuster itinéraire']
                })
            
            # Vitesse excessive
            if speed > 90:
                alerts.append({
                    'type': 'speeding',
                    'severity': 'high',
                    'message': f'Vitesse excessive: {speed:.1f} km/h',
                    'actions': ['Alerter conducteur', 'Vérifier sécurité']
                })
            
            # Génération des alertes
            for alert_data in alerts:
                alert = StreamAlert(
                    id=f"{vehicle_id}_{alert_data['type']}_{int(datetime.now().timestamp())}",
                    type=alert_data['type'],
                    severity=alert_data['severity'],
                    message=f"[{vehicle_id}] {alert_data['message']}",
                    source_data={'vehicle_id': vehicle_id, **transport_data},
                    timestamp=datetime.now(),
                    recommended_actions=alert_data['actions']
                )
                
                await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification transport: {e}")
    
    async def _check_route_optimization(self, vehicle_id: str, transport_data: Dict[str, Any]):
        """Vérifie si une optimisation de route est nécessaire"""
        try:
            if not self.transport_optimizer:
                return
            
            route_efficiency = transport_data.get('route_efficiency', 1.0)
            
            # Déclencher réoptimisation si efficacité faible
            if route_efficiency < 0.6:
                logger.info(f"🔄 Déclenchement réoptimisation route pour {vehicle_id}")
                
                # En réalité, cela déclencherait une réoptimisation complète
                # Ici on simule juste l'alerte
                alert = StreamAlert(
                    id=f"reoptimization_{vehicle_id}_{int(datetime.now().timestamp())}",
                    type='route_reoptimization',
                    severity='medium',
                    message=f"Réoptimisation de route recommandée pour {vehicle_id}",
                    source_data={'vehicle_id': vehicle_id, **transport_data},
                    timestamp=datetime.now(),
                    recommended_actions=['Recalculer itinéraire optimal', 'Informer conducteur']
                )
                
                await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification optimisation: {e}")
    
    async def _process_inventory_update(self, erp_data: Dict[str, Any]):
        """Traite une mise à jour d'inventaire"""
        try:
            product_id = erp_data.get('product_id')
            quantity = erp_data.get('quantity', 0)
            warehouse = erp_data.get('warehouse')
            
            logger.debug(f"📦 Mise à jour stock: {product_id} = {quantity} unités à {warehouse}")
            
            # Vérification des seuils de stock
            if quantity < 50:  # Seuil arbitraire
                alert = StreamAlert(
                    id=f"inventory_low_{product_id}_{int(datetime.now().timestamp())}",
                    type='inventory_below_threshold',
                    severity='medium',
                    message=f"Stock sous seuil pour {product_id}: {quantity} unités",
                    source_data=erp_data,
                    timestamp=datetime.now(),
                    recommended_actions=['Vérifier prévisions', 'Planifier réapprovisionnement']
                )
                
                await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement inventaire: {e}")
    
    async def _process_shipment_data(self, erp_data: Dict[str, Any]):
        """Traite les données d'expédition"""
        try:
            logger.debug(f"📦 Expédition traitée: {erp_data.get('product_id')}")
            
            # Mise à jour des métriques de livraison
            # En réalité, cela mettrait à jour des KPIs temps réel
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement expédition: {e}")
    
    async def _emit_alert(self, alert: StreamAlert):
        """Émet une alerte dans le système"""
        try:
            # Éviter les doublons d'alertes
            alert_key = f"{alert.type}_{alert.source_data.get('equipment_id', alert.source_data.get('vehicle_id', 'unknown'))}"
            
            if alert_key in self.active_alerts:
                last_alert = self.active_alerts[alert_key]
                if (alert.timestamp - last_alert.timestamp).seconds < 300:  # 5 minutes
                    return  # Éviter le spam d'alertes
            
            self.active_alerts[alert_key] = alert
            
            logger.warning(f"🚨 ALERTE [{alert.severity.upper()}]: {alert.message}")
            
            # En réalité, cela enverrait l'alerte via email, SMS, dashboard, etc.
            
        except Exception as e:
            logger.error(f"❌ Erreur émission alerte: {e}")
    
    async def _update_global_metrics(self, message: DataMessage):
        """Met à jour les métriques globales temps réel"""
        try:
            current_time = datetime.now()
            
            if message.data_type == 'order':
                # Comptage des commandes par heure
                hour_start = current_time.replace(minute=0, second=0, microsecond=0)
                hourly_orders = [
                    order for order in self.order_buffer
                    if order['timestamp'] >= hour_start
                ]
                self.real_time_metrics['orders_per_hour'] = len(hourly_orders)
                
                # Valeur moyenne des commandes
                if hourly_orders:
                    avg_value = np.mean([order['data'].get('order_value', 0) for order in hourly_orders])
                    self.real_time_metrics['avg_order_value'] = avg_value
            
            elif message.data_type == 'sensor_reading':
                # Comptage des alertes équipements
                recent_alerts = [
                    alert for alert in self.active_alerts.values()
                    if (current_time - alert.timestamp).seconds < 3600  # Dernière heure
                ]
                self.real_time_metrics['equipment_alerts'] = len(recent_alerts)
            
            elif message.data_type == 'gps_position':
                # Efficacité moyenne du transport
                recent_positions = []
                for vehicle_data in self.transport_buffer.values():
                    recent_positions.extend([
                        pos['data'].get('route_efficiency', 0)
                        for pos in vehicle_data
                        if (current_time - pos['timestamp']).seconds < 3600
                    ])
                
                if recent_positions:
                    self.real_time_metrics['transport_efficiency'] = np.mean(recent_positions) * 100
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour métriques: {e}")
    
    async def _process_metrics_aggregation(self):
        """Agrège les métriques en continu"""
        while self.running:
            try:
                # Calcul de métriques avancées toutes les minutes
                await self._calculate_advanced_metrics()
                await asyncio.sleep(60)  # Toutes les minutes
                
            except Exception as e:
                logger.error(f"❌ Erreur agrégation métriques: {e}")
                await asyncio.sleep(5)
    
    async def _process_anomaly_detection(self):
        """Processus de détection d'anomalies global"""
        while self.running:
            try:
                # Analyse d'anomalies globales toutes les 5 minutes
                await self._detect_global_anomalies()
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Erreur détection anomalies: {e}")
                await asyncio.sleep(30)
    
    async def _process_optimization_triggers(self):
        """Déclenche les optimisations automatiques"""
        while self.running:
            try:
                # Vérification des triggers d'optimisation toutes les 10 minutes
                await self._check_optimization_triggers()
                await asyncio.sleep(600)  # Toutes les 10 minutes
                
            except Exception as e:
                logger.error(f"❌ Erreur triggers optimisation: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self):
        """Nettoie les anciennes données des buffers"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # Nettoyage des buffers
                for equipment_id, buffer in self.sensor_buffer.items():
                    while buffer and buffer[0]['timestamp'] < cutoff_time:
                        buffer.popleft()
                
                for vehicle_id, buffer in self.transport_buffer.items():
                    while buffer and buffer[0]['timestamp'] < cutoff_time:
                        buffer.popleft()
                
                # Nettoyage des alertes anciennes
                expired_alerts = [
                    key for key, alert in self.active_alerts.items()
                    if (datetime.now() - alert.timestamp).hours > 24
                ]
                for key in expired_alerts:
                    del self.active_alerts[key]
                
                await asyncio.sleep(3600)  # Toutes les heures
                
            except Exception as e:
                logger.error(f"❌ Erreur nettoyage données: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_advanced_metrics(self):
        """Calcule des métriques avancées"""
        try:
            # Efficacité globale de la supply chain
            transport_eff = self.real_time_metrics.get('transport_efficiency', 0)
            equipment_health = max(0, 100 - self.real_time_metrics.get('equipment_alerts', 0) * 10)
            order_flow = min(100, self.real_time_metrics.get('orders_per_hour', 0) * 2)
            
            overall_efficiency = (transport_eff + equipment_health + order_flow) / 3
            self.real_time_metrics['overall_efficiency'] = overall_efficiency
            
            logger.debug(f"📊 Métriques calculées: efficacité globale = {overall_efficiency:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul métriques avancées: {e}")
    
    async def _detect_global_anomalies(self):
        """Détecte les anomalies au niveau global"""
        try:
            # Détection de patterns anormaux dans le flux global
            current_orders = self.real_time_metrics.get('orders_per_hour', 0)
            
            # Heures normales de bureau
            hour = datetime.now().hour
            if 8 <= hour <= 18:
                expected_orders = 50  # Valeur attendue
                if current_orders < expected_orders * 0.3:  # 30% de la normale
                    alert = StreamAlert(
                        id=f"global_low_activity_{int(datetime.now().timestamp())}",
                        type='low_business_activity',
                        severity='medium',
                        message=f"Activité commerciale inhabituellement faible: {current_orders} commandes/h",
                        source_data={'orders_per_hour': current_orders, 'expected': expected_orders},
                        timestamp=datetime.now(),
                        recommended_actions=['Vérifier systèmes clients', 'Analyser tendances marché']
                    )
                    
                    await self._emit_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Erreur détection anomalies globales: {e}")
    
    async def _check_optimization_triggers(self):
        """Vérifie les conditions de déclenchement d'optimisations"""
        try:
            # Déclenchement d'optimisation transport si efficacité faible
            transport_eff = self.real_time_metrics.get('transport_efficiency', 100)
            if transport_eff < 70:
                logger.info(f"🔄 Déclenchement optimisation transport: efficacité = {transport_eff:.1f}%")
                # En réalité, déclencher une réoptimisation complète
            
            # Déclenchement de prévision de demande si pic détecté
            current_orders = self.real_time_metrics.get('orders_per_hour', 0)
            if current_orders > 100:  # Pic de demande
                logger.info(f"📈 Pic de demande détecté: {current_orders} commandes/h")
                # En réalité, déclencher une mise à jour des prévisions
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification triggers: {e}")
    
    async def stop(self):
        """Arrête le processeur de flux"""
        self.running = False
        logger.info("🛑 StreamProcessor arrêté")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques temps réel actuelles"""
        return {
            **self.real_time_metrics,
            'active_alerts_count': len(self.active_alerts),
            'buffer_sizes': {
                'sensors': sum(len(buffer) for buffer in self.sensor_buffer.values()),
                'orders': len(self.order_buffer),
                'transport': sum(len(buffer) for buffer in self.transport_buffer.values())
            },
            'last_processing': self.last_processing_time.isoformat(),
            'status': 'running' if self.running else 'stopped'
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retourne les alertes actives"""
        return [
            {
                'id': alert.id,
                'type': alert.type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'actions': alert.recommended_actions
            }
            for alert in self.active_alerts.values()
        ]
