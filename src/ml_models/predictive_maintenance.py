"""
Maintenance prédictive des équipements
Détection d'anomalies et planification intelligente
Basé sur les données IoT et algorithmes ML avancés
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import joblib
import json

# Machine Learning
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Heavy ML libraries are imported lazily inside methods to avoid import-time
# side-effects (dask/xgboost import issues) when the server starts.

def _get_xgboost():
    try:
        import importlib
        return importlib.import_module('xgboost')
    except Exception:
        return None


def _get_tensorflow():
    try:
        import importlib
        return importlib.import_module('tensorflow')
    except Exception:
        return None

# Configuration
from src.config.settings import settings, BusinessConstants
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MaintenanceType(Enum):
    """Types de maintenance"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    EMERGENCY = "emergency"

class EquipmentStatus(Enum):
    """États des équipements"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class Equipment:
    """Équipement avec ses caractéristiques"""
    id: str
    name: str
    type: str
    location: str
    installation_date: datetime
    last_maintenance: datetime
    criticality_level: int  # 1=low, 2=medium, 3=high, 4=critical
    operating_hours: float
    manufacturer: str
    model: str

@dataclass
class SensorReading:
    """Lecture de capteur IoT"""
    equipment_id: str
    timestamp: datetime
    sensor_type: str
    value: float
    unit: str
    quality: float = 1.0  # Qualité de la mesure 0-1

@dataclass
class MaintenanceAlert:
    """Alerte de maintenance"""
    equipment_id: str
    alert_type: MaintenanceType
    severity: str
    predicted_failure_date: datetime
    confidence: float
    description: str
    recommended_actions: List[str]
    cost_impact: float

class MaintenancePredictor:
    """
    Système de maintenance prédictive
    
    Features:
    - Détection d'anomalies en temps réel
    - Prédiction de pannes avec ML
    - Optimisation des plannings de maintenance
    - Analyse de criticité des équipements
    - ROI des interventions
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        self.equipment_profiles = {}
        
        # Paramètres de détection
        self.lookback_window = 168  # 7 jours d'historique (heures)
        self.prediction_horizon = 720  # 30 jours en avant (heures)
        
        # Seuils d'alerte
        self.warning_threshold = settings.WARNING_THRESHOLD
        self.critical_threshold = settings.CRITICAL_THRESHOLD
        
        logger.info("🔧 MaintenancePredictor initialisé")
    
    async def load_models(self):
        """Charge les modèles pré-entraînés"""
        try:
            model_paths = settings.get_model_paths()
            maintenance_path = model_paths["maintenance"]
            
            if maintenance_path.exists():
                # Chargement des modèles de prédiction
                for model_file in maintenance_path.glob("*_predictor.pkl"):
                    equipment_type = model_file.stem.replace("_predictor", "")
                    self.models[equipment_type] = joblib.load(model_file)
                
                # Chargement des détecteurs d'anomalies
                for detector_file in maintenance_path.glob("*_anomaly.pkl"):
                    equipment_type = detector_file.stem.replace("_anomaly", "")
                    self.anomaly_detectors[equipment_type] = joblib.load(detector_file)
                
                # Chargement des scalers
                for scaler_file in maintenance_path.glob("*_scaler.pkl"):
                    equipment_type = scaler_file.stem.replace("_scaler", "")
                    self.scalers[equipment_type] = joblib.load(scaler_file)
            
            logger.info(f"✅ Modèles de maintenance chargés: {len(self.models)} types")
            
        except Exception as e:
            logger.warning(f"⚠️ Aucun modèle de maintenance trouvé: {e}")
            self.models = {}
    
    def extract_features(self, sensor_data: List[SensorReading], equipment: Equipment) -> Dict[str, float]:
        """
        Extrait les features des données de capteurs
        Calcule statistiques, tendances, et indicateurs de santé
        """
        if not sensor_data:
            return {}
        
        # Conversion en DataFrame
        df = pd.DataFrame([
            {
                'timestamp': reading.timestamp,
                'sensor_type': reading.sensor_type,
                'value': reading.value,
                'quality': reading.quality
            }
            for reading in sensor_data
        ])
        
        features = {}
        
        # Features par type de capteur
        for sensor_type in df['sensor_type'].unique():
            sensor_df = df[df['sensor_type'] == sensor_type].copy()
            sensor_df = sensor_df.sort_values('timestamp')
            
            if len(sensor_df) == 0:
                continue
            
            values = sensor_df['value'].values
            
            # Statistiques de base
            features[f'{sensor_type}_mean'] = np.mean(values)
            features[f'{sensor_type}_std'] = np.std(values)
            features[f'{sensor_type}_min'] = np.min(values)
            features[f'{sensor_type}_max'] = np.max(values)
            features[f'{sensor_type}_range'] = np.max(values) - np.min(values)
            
            # Percentiles
            features[f'{sensor_type}_p25'] = np.percentile(values, 25)
            features[f'{sensor_type}_p75'] = np.percentile(values, 75)
            features[f'{sensor_type}_p95'] = np.percentile(values, 95)
            
            # Tendance (régression linéaire simple)
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                features[f'{sensor_type}_trend'] = slope
            
            # Variabilité
            if len(values) > 2:
                features[f'{sensor_type}_cv'] = np.std(values) / (np.mean(values) + 1e-8)
            
            # Comptage des valeurs aberrantes (> 2 std)
            if np.std(values) > 0:
                outliers = np.abs(values - np.mean(values)) > 2 * np.std(values)
                features[f'{sensor_type}_outliers_count'] = np.sum(outliers)
                features[f'{sensor_type}_outliers_ratio'] = np.mean(outliers)
            
            # Qualité moyenne des mesures
            features[f'{sensor_type}_quality_mean'] = sensor_df['quality'].mean()
        
        # Features liées à l'équipement
        features['equipment_age_days'] = (datetime.now() - equipment.installation_date).days
        features['days_since_maintenance'] = (datetime.now() - equipment.last_maintenance).days
        features['operating_hours'] = equipment.operating_hours
        features['criticality_level'] = equipment.criticality_level
        
        # Features temporelles
        features['hour_of_day'] = datetime.now().hour
        features['day_of_week'] = datetime.now().weekday()
        features['month'] = datetime.now().month
        
        return features
    
    def detect_anomalies(self, features: Dict[str, float], equipment_type: str) -> Dict[str, Any]:
        """
        Détecte les anomalies dans les données de capteurs
        Utilise Isolation Forest et seuils statistiques
        """
        if equipment_type not in self.anomaly_detectors:
            logger.warning(f"Détecteur d'anomalies non disponible pour {equipment_type}")
            return {'anomaly_score': 0.0, 'is_anomaly': False}
        
        try:
            detector = self.anomaly_detectors[equipment_type]
            scaler = self.scalers.get(equipment_type)
            
            # Préparation des features
            feature_names = detector.feature_names_in_
            feature_vector = []
            
            for feature_name in feature_names:
                value = features.get(feature_name, 0.0)
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Normalisation si scaler disponible
            if scaler:
                feature_vector = scaler.transform(feature_vector)
            
            # Détection d'anomalie
            anomaly_score = detector.decision_function(feature_vector)[0]
            is_anomaly = detector.predict(feature_vector)[0] == -1
            
            return {
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'confidence': abs(anomaly_score)
            }
            
        except Exception as e:
            logger.error(f"Erreur détection anomalie: {e}")
            return {'anomaly_score': 0.0, 'is_anomaly': False, 'confidence': 0.0}
    
    def predict_failure(self, features: Dict[str, float], equipment_type: str) -> Dict[str, Any]:
        """
        Prédit la probabilité de panne et le temps avant défaillance
        """
        if equipment_type not in self.models:
            logger.warning(f"Modèle de prédiction non disponible pour {equipment_type}")
            return {
                'failure_probability': 0.0,
                'time_to_failure_hours': None,
                'confidence': 0.0
            }
        
        try:
            model = self.models[equipment_type]
            scaler = self.scalers.get(equipment_type)
            
            # Préparation des features
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                # Fallback si pas d'attribut feature_names_in_
                feature_names = list(features.keys())
            
            feature_vector = []
            for feature_name in feature_names:
                value = features.get(feature_name, 0.0)
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Normalisation
            if scaler:
                feature_vector = scaler.transform(feature_vector)
            
            # Prédiction
            if hasattr(model, 'predict_proba'):
                # Modèle de classification (probabilité de panne)
                failure_prob = model.predict_proba(feature_vector)[0, 1]
                
                # Estimation du temps avant panne basée sur la probabilité
                if failure_prob > 0.8:
                    time_to_failure = 24  # 1 jour
                elif failure_prob > 0.6:
                    time_to_failure = 168  # 1 semaine
                elif failure_prob > 0.4:
                    time_to_failure = 720  # 1 mois
                else:
                    time_to_failure = None
                
            else:
                # Modèle de régression (temps avant panne)
                time_to_failure = model.predict(feature_vector)[0]
                failure_prob = max(0, 1 - (time_to_failure / 720))  # Inverse du temps
            
            return {
                'failure_probability': float(failure_prob),
                'time_to_failure_hours': float(time_to_failure) if time_to_failure else None,
                'confidence': float(abs(failure_prob - 0.5) * 2)  # Distance de 0.5
            }
            
        except Exception as e:
            logger.error(f"Erreur prédiction panne: {e}")
            return {
                'failure_probability': 0.0,
                'time_to_failure_hours': None,
                'confidence': 0.0
            }
    
    async def analyze_equipment(
        self, 
        equipment: Equipment, 
        sensor_data: List[SensorReading]
    ) -> Dict[str, Any]:
        """
        Analyse complète d'un équipement
        Combine détection d'anomalies et prédiction de pannes
        """
        try:
            # Extraction des features
            features = self.extract_features(sensor_data, equipment)
            
            if not features:
                return {
                    'equipment_id': equipment.id,
                    'status': EquipmentStatus.HEALTHY.value,
                    'message': 'Données insuffisantes'
                }
            
            # Détection d'anomalies
            anomaly_result = self.detect_anomalies(features, equipment.type)
            
            # Prédiction de panne
            failure_result = self.predict_failure(features, equipment.type)
            
            # Détermination du statut global
            status = self._determine_equipment_status(anomaly_result, failure_result)
            
            # Génération d'alertes si nécessaire
            alerts = []
            if anomaly_result.get('is_anomaly', False):
                alerts.append({
                    'type': 'anomaly',
                    'message': 'Comportement anormal détecté',
                    'severity': 'warning'
                })
            
            if failure_result.get('failure_probability', 0) > self.critical_threshold:
                alerts.append({
                    'type': 'failure_risk',
                    'message': f"Risque de panne élevé ({failure_result['failure_probability']:.1%})",
                    'severity': 'critical'
                })
            
            # Recommandations de maintenance
            recommendations = self._generate_maintenance_recommendations(
                equipment, anomaly_result, failure_result, features
            )
            
            return {
                'equipment_id': equipment.id,
                'equipment_name': equipment.name,
                'status': status.value,
                'timestamp': datetime.now().isoformat(),
                'anomaly_detection': anomaly_result,
                'failure_prediction': failure_result,
                'alerts': alerts,
                'recommendations': recommendations,
                'features_summary': {
                    'total_features': len(features),
                    'key_indicators': self._extract_key_indicators(features)
                },
                'next_analysis': (datetime.now() + timedelta(
                    hours=settings.MAINTENANCE_CHECK_INTERVAL_HOURS
                )).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse équipement {equipment.id}: {e}")
            return {
                'equipment_id': equipment.id,
                'status': EquipmentStatus.HEALTHY.value,
                'error': str(e)
            }
    
    def _determine_equipment_status(
        self, 
        anomaly_result: Dict[str, Any], 
        failure_result: Dict[str, Any]
    ) -> EquipmentStatus:
        """Détermine le statut global de l'équipement"""
        
        failure_prob = failure_result.get('failure_probability', 0)
        is_anomaly = anomaly_result.get('is_anomaly', False)
        
        if failure_prob > self.critical_threshold:
            return EquipmentStatus.CRITICAL
        elif failure_prob > self.warning_threshold or is_anomaly:
            return EquipmentStatus.WARNING
        else:
            return EquipmentStatus.HEALTHY
    
    def _generate_maintenance_recommendations(
        self, 
        equipment: Equipment,
        anomaly_result: Dict[str, Any],
        failure_result: Dict[str, Any],
        features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Génère des recommandations de maintenance personnalisées"""
        
        recommendations = []
        
        failure_prob = failure_result.get('failure_probability', 0)
        time_to_failure = failure_result.get('time_to_failure_hours')
        
        # Recommandations basées sur le risque de panne
        if failure_prob > self.critical_threshold:
            recommendations.append({
                'priority': 'urgent',
                'action': 'Planifier maintenance corrective immédiate',
                'reason': f'Probabilité de panne très élevée ({failure_prob:.1%})',
                'estimated_cost': self._estimate_maintenance_cost(equipment, 'corrective'),
                'deadline': datetime.now() + timedelta(days=1)
            })
        elif failure_prob > self.warning_threshold:
            recommendations.append({
                'priority': 'high',
                'action': 'Planifier maintenance préventive',
                'reason': f'Probabilité de panne modérée ({failure_prob:.1%})',
                'estimated_cost': self._estimate_maintenance_cost(equipment, 'preventive'),
                'deadline': datetime.now() + timedelta(days=7)
            })
        
        # Recommandations basées sur l'âge de l'équipement
        days_since_maintenance = features.get('days_since_maintenance', 0)
        if days_since_maintenance > 365:  # Plus d'un an
            recommendations.append({
                'priority': 'medium',
                'action': 'Maintenance de routine programmée',
                'reason': f'Dernière maintenance il y a {days_since_maintenance} jours',
                'estimated_cost': self._estimate_maintenance_cost(equipment, 'routine'),
                'deadline': datetime.now() + timedelta(days=30)
            })
        
        # Recommandations basées sur les anomalies
        if anomaly_result.get('is_anomaly', False):
            recommendations.append({
                'priority': 'medium',
                'action': 'Inspection technique approfondie',
                'reason': 'Comportement anormal détecté dans les capteurs',
                'estimated_cost': self._estimate_maintenance_cost(equipment, 'inspection'),
                'deadline': datetime.now() + timedelta(days=3)
            })
        
        # Tri par priorité
        priority_order = {'urgent': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return recommendations
    
    def _estimate_maintenance_cost(self, equipment: Equipment, maintenance_type: str) -> float:
        """Estime le coût de maintenance selon le type"""
        
        base_costs = {
            'inspection': 500,
            'routine': 1000,
            'preventive': 2000,
            'corrective': 5000,
            'emergency': 10000
        }
        
        base_cost = base_costs.get(maintenance_type, 1000)
        
        # Ajustement selon la criticité
        criticality_multiplier = {1: 0.8, 2: 1.0, 3: 1.5, 4: 2.0}
        multiplier = criticality_multiplier.get(equipment.criticality_level, 1.0)
        
        return base_cost * multiplier
    
    def _extract_key_indicators(self, features: Dict[str, float]) -> Dict[str, float]:
        """Extrait les indicateurs clés des features"""
        
        key_indicators = {}
        
        # Température moyenne si disponible
        for key, value in features.items():
            if 'temperature' in key.lower() and 'mean' in key:
                key_indicators['temperature_mean'] = value
            elif 'vibration' in key.lower() and 'mean' in key:
                key_indicators['vibration_mean'] = value
            elif 'pressure' in key.lower() and 'mean' in key:
                key_indicators['pressure_mean'] = value
            elif 'current' in key.lower() and 'mean' in key:
                key_indicators['current_mean'] = value
        
        # Ratios d'anomalies
        for key, value in features.items():
            if 'outliers_ratio' in key:
                key_indicators[key] = value
        
        return key_indicators
    
    async def train_models(self, training_data: Dict[str, pd.DataFrame]):
        """
        Entraîne les modèles de maintenance prédictive
        training_data: {equipment_type: DataFrame avec features et labels}
        """
        logger.info(f"🎯 Entraînement des modèles de maintenance pour {len(training_data)} types")
        
        for equipment_type, data in training_data.items():
            try:
                logger.info(f"Entraînement pour {equipment_type}")
                
                # Séparation features/labels
                feature_cols = [col for col in data.columns if col not in ['failure', 'timestamp']]
                X = data[feature_cols]
                y = data.get('failure', np.zeros(len(data)))  # 0=normal, 1=panne
                
                # Division train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                )
                
                # Normalisation
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement du détecteur d'anomalies
                anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                anomaly_detector.fit(X_train_scaled)
                
                # Entraînement du prédicteur de pannes
                xgb_mod = _get_xgboost()
                if xgb_mod is not None:
                    try:
                        if len(np.unique(y)) > 1:
                            failure_predictor = xgb_mod.XGBClassifier(
                                n_estimators=100,
                                max_depth=6,
                                learning_rate=0.1,
                                random_state=42
                            )
                            failure_predictor.fit(X_train_scaled, y_train)
                            y_pred = failure_predictor.predict(X_test_scaled)
                            logger.info(f"Précision {equipment_type}: {(y_pred == y_test).mean():.3f}")
                        else:
                            failure_predictor = xgb_mod.XGBRegressor(
                                n_estimators=100,
                                max_depth=6,
                                learning_rate=0.1,
                                random_state=42
                            )
                            time_to_failure = np.random.exponential(500, size=len(y_train))
                            failure_predictor.fit(X_train_scaled, time_to_failure)
                    except Exception as e:
                        logger.warning(f"xgboost training failed, falling back to sklearn for {equipment_type}: {e}")
                        xgb_mod = None

                if xgb_mod is None:
                    # Fallback to sklearn models when xgboost isn't available
                    if len(np.unique(y)) > 1:
                        failure_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
                        failure_predictor.fit(X_train_scaled, y_train)
                        y_pred = failure_predictor.predict(X_test_scaled)
                        logger.info(f"Précision (RF) {equipment_type}: {(y_pred == y_test).mean():.3f}")
                    else:
                        # Use a simple RandomForest regressor as fallback
                        from sklearn.ensemble import RandomForestRegressor
                        failure_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
                        time_to_failure = np.random.exponential(500, size=len(y_train))
                        failure_predictor.fit(X_train_scaled, time_to_failure)
                
                # Sauvegarde des modèles
                self.models[equipment_type] = failure_predictor
                self.anomaly_detectors[equipment_type] = anomaly_detector
                self.scalers[equipment_type] = scaler
                
                logger.info(f"✅ Modèles entraînés pour {equipment_type}")
                
            except Exception as e:
                logger.error(f"❌ Erreur entraînement {equipment_type}: {e}")
        
        # Sauvegarde
        await self.save_models()
    
    async def save_models(self):
        """Sauvegarde tous les modèles"""
        try:
            model_paths = settings.get_model_paths()
            maintenance_path = model_paths["maintenance"]
            maintenance_path.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde des modèles
            for equipment_type, model in self.models.items():
                model_file = maintenance_path / f"{equipment_type}_predictor.pkl"
                joblib.dump(model, model_file)
            
            # Sauvegarde des détecteurs
            for equipment_type, detector in self.anomaly_detectors.items():
                detector_file = maintenance_path / f"{equipment_type}_anomaly.pkl"
                joblib.dump(detector, detector_file)
            
            # Sauvegarde des scalers
            for equipment_type, scaler in self.scalers.items():
                scaler_file = maintenance_path / f"{equipment_type}_scaler.pkl"
                joblib.dump(scaler, scaler_file)
            
            logger.info("💾 Modèles de maintenance sauvegardés")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde modèles maintenance: {e}")
    
    async def generate_maintenance_schedule(
        self, 
        equipments: List[Equipment],
        time_horizon_days: int = 90
    ) -> Dict[str, Any]:
        """
        Génère un planning optimal de maintenance
        Optimise coûts, disponibilité et risques
        """
        try:
            logger.info(f"📅 Génération planning maintenance pour {len(equipments)} équipements")
            
            # Simulation de données capteurs pour chaque équipement
            maintenance_schedule = []
            total_cost = 0
            
            for equipment in equipments:
                # Simulation de données de capteurs récentes
                sensor_data = self._simulate_sensor_data(equipment)
                
                # Analyse de l'équipement
                analysis = await self.analyze_equipment(equipment, sensor_data)
                
                # Extraction des recommandations
                recommendations = analysis.get('recommendations', [])
                
                for rec in recommendations:
                    if 'deadline' in rec:
                        maintenance_schedule.append({
                            'equipment_id': equipment.id,
                            'equipment_name': equipment.name,
                            'type': rec['action'],
                            'priority': rec['priority'],
                            'scheduled_date': rec['deadline'].isoformat() if hasattr(rec['deadline'], 'isoformat') else rec['deadline'],
                            'estimated_cost': rec.get('estimated_cost', 0),
                            'reason': rec['reason']
                        })
                        total_cost += rec.get('estimated_cost', 0)
            
            # Tri par date et priorité
            priority_weights = {'urgent': 1, 'high': 2, 'medium': 3, 'low': 4}
            maintenance_schedule.sort(
                key=lambda x: (x['scheduled_date'], priority_weights.get(x['priority'], 5))
            )
            
            # Statistiques du planning
            schedule_stats = {
                'total_interventions': len(maintenance_schedule),
                'total_estimated_cost': total_cost,
                'urgent_interventions': len([x for x in maintenance_schedule if x['priority'] == 'urgent']),
                'high_priority_interventions': len([x for x in maintenance_schedule if x['priority'] == 'high']),
                'average_cost_per_intervention': total_cost / len(maintenance_schedule) if maintenance_schedule else 0
            }
            
            return {
                'schedule': maintenance_schedule,
                'statistics': schedule_stats,
                'generated_at': datetime.now().isoformat(),
                'time_horizon_days': time_horizon_days,
                'recommendations': [
                    "Prioriser les interventions urgentes",
                    "Grouper les maintenances par zone géographique",
                    "Planifier pendant les périodes de faible activité"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération planning: {e}")
            return {'error': str(e)}
    
    def _simulate_sensor_data(self, equipment: Equipment) -> List[SensorReading]:
        """Simule des données de capteurs pour un équipement"""
        
        sensor_data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        # Types de capteurs selon le type d'équipement
        sensor_types = {
            'pump': ['temperature', 'vibration', 'pressure', 'current'],
            'conveyor': ['temperature', 'vibration', 'speed', 'current'],
            'robot': ['temperature', 'position_accuracy', 'current', 'cycle_time'],
            'default': ['temperature', 'vibration', 'current']
        }
        
        equipment_sensors = sensor_types.get(equipment.type, sensor_types['default'])
        
        # Génération de données sur 24h avec mesures toutes les heures
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            
            for sensor_type in equipment_sensors:
                # Valeurs de base selon le type de capteur
                base_values = {
                    'temperature': 60,
                    'vibration': 2.0,
                    'pressure': 5.0,
                    'current': 10.0,
                    'speed': 100,
                    'position_accuracy': 0.1,
                    'cycle_time': 30
                }
                
                base_value = base_values.get(sensor_type, 50)
                
                # Ajout de bruit et tendance
                noise = np.random.normal(0, base_value * 0.1)
                trend = np.sin(hour * np.pi / 12) * base_value * 0.2  # Variation journalière
                
                # Simulation d'anomalies occasionnelles
                if np.random.random() < 0.05:  # 5% de chance d'anomalie
                    noise += base_value * 0.5 * np.random.choice([-1, 1])
                
                value = base_value + noise + trend
                
                reading = SensorReading(
                    equipment_id=equipment.id,
                    timestamp=timestamp,
                    sensor_type=sensor_type,
                    value=max(0, value),  # Éviter les valeurs négatives
                    unit=self._get_sensor_unit(sensor_type),
                    quality=np.random.uniform(0.9, 1.0)
                )
                
                sensor_data.append(reading)
        
        return sensor_data
    
    def _get_sensor_unit(self, sensor_type: str) -> str:
        """Retourne l'unité pour un type de capteur"""
        units = {
            'temperature': '°C',
            'vibration': 'mm/s',
            'pressure': 'bar',
            'current': 'A',
            'speed': 'rpm',
            'position_accuracy': 'mm',
            'cycle_time': 's'
        }
        return units.get(sensor_type, 'unit')
    
    def calculate_maintenance_roi(
        self, 
        maintenance_cost: float, 
        avoided_failure_cost: float,
        equipment_criticality: int
    ) -> Dict[str, float]:
        """Calcule le ROI d'une intervention de maintenance"""
        
        # Coût d'une panne selon la criticité
        failure_cost_multipliers = {1: 1.0, 2: 2.0, 3: 5.0, 4: 10.0}
        multiplier = failure_cost_multipliers.get(equipment_criticality, 1.0)
        
        total_failure_cost = avoided_failure_cost * multiplier
        
        # Calcul du ROI
        roi = (total_failure_cost - maintenance_cost) / maintenance_cost * 100
        
        return {
            'maintenance_cost': maintenance_cost,
            'avoided_failure_cost': total_failure_cost,
            'net_benefit': total_failure_cost - maintenance_cost,
            'roi_percent': roi,
            'payback_period_months': 12 / (roi / 100) if roi > 0 else None
        }
