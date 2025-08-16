"""
Modèle de prévision intelligente de la demande
Utilise Prophet, XGBoost et LSTM pour des prédictions haute précision
Inspiré des méthodes utilisées par Walmart et Amazon
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import joblib
import asyncio
from pathlib import Path
import logging

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration
from config.settings import settings, BusinessConstants
from utils.logger import setup_logger
from data_ingestion.external_data import WeatherDataCollector, SocialMediaAnalyzer

logger = setup_logger(__name__)

class DemandForecaster:
    """
    Prévision intelligente de la demande avec algorithmes ML avancés
    
    Features:
    - Prophet pour capture de saisonnalité et tendances
    - XGBoost pour relations non-linéaires complexes
    - LSTM pour séquences temporelles longues
    - Variables externes (météo, événements, réseaux sociaux)
    - Ensemble methods pour robustesse
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Collecteurs de données externes
        self.weather_collector = WeatherDataCollector()
        self.social_analyzer = SocialMediaAnalyzer()
        
        # Configuration des modèles
        self.lookback_window = 60  # 60 jours d'historique
        self.forecast_horizon = settings.FORECAST_HORIZON_DAYS
        
        logger.info("🧠 DemandForecaster initialisé")
    
    async def load_models(self) -> None:
        """Charge les modèles pré-entraînés"""
        try:
            model_paths = settings.get_model_paths()
            
            # Chargement des modèles Prophet
            prophet_path = model_paths["prophet"]
            if prophet_path.exists():
                self.models['prophet'] = {}
                for model_file in prophet_path.glob("*.pkl"):
                    product_id = model_file.stem
                    self.models['prophet'][product_id] = joblib.load(model_file)
            
            # Chargement des modèles XGBoost
            xgboost_path = model_paths["xgboost"]
            if xgboost_path.exists():
                self.models['xgboost'] = {}
                for model_file in xgboost_path.glob("*.pkl"):
                    product_id = model_file.stem
                    self.models['xgboost'][product_id] = joblib.load(model_file)
            
            # Chargement des modèles LSTM
            lstm_path = model_paths["lstm"]
            if lstm_path.exists():
                self.models['lstm'] = {}
                for model_dir in lstm_path.iterdir():
                    if model_dir.is_dir():
                        product_id = model_dir.name
                        model_file = model_dir / "model.h5"
                        if model_file.exists():
                            self.models['lstm'][product_id] = load_model(str(model_file))
            
            # Chargement des scalers et encoders
            scalers_path = model_paths["prophet"].parent / "scalers"
            if scalers_path.exists():
                for scaler_file in scalers_path.glob("*.pkl"):
                    product_id = scaler_file.stem
                    self.scalers[product_id] = joblib.load(scaler_file)
            
            logger.info(f"✅ Modèles chargés: {len(self.models)} types")
            
        except Exception as e:
            logger.warning(f"⚠️ Aucun modèle pré-entraîné trouvé: {e}")
            self.models = {'prophet': {}, 'xgboost': {}, 'lstm': {}}
    
    def prepare_features(self, data: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """
        Prépare les features pour la prédiction
        Inclut variables temporelles, météo, événements, réseaux sociaux
        """
        df = data.copy()
        
        # Features temporelles
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['weekday'] = df.index.weekday
        df['quarter'] = df.index.quarter
        df['week_of_year'] = df.index.isocalendar().week
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Features cycliques (encodage sinusoïdal)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # Features de lag (valeurs passées)
        for lag in [1, 7, 14, 30]:
            df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
        
        # Moyennes mobiles
        for window in [7, 14, 30]:
            df[f'demand_ma_{window}'] = df['demand'].rolling(window=window).mean()
            df[f'demand_std_{window}'] = df['demand'].rolling(window=window).std()
        
        # Features de croissance
        df['demand_growth_7d'] = df['demand'].pct_change(7)
        df['demand_growth_30d'] = df['demand'].pct_change(30)
        
        # Détection d'événements spéciaux
        df['is_holiday'] = self._detect_holidays(df.index)
        df['is_promotion'] = self._detect_promotions(df.index, product_id)
        df['is_stockout'] = self._detect_stockouts(df)
        
        # Features météorologiques (si disponibles)
        try:
            weather_data = self.weather_collector.get_historical_data(
                start_date=df.index.min(),
                end_date=df.index.max()
            )
            df = df.merge(weather_data, left_index=True, right_index=True, how='left')
        except Exception as e:
            logger.warning(f"Données météo non disponibles: {e}")
        
        # Features de sentiment des réseaux sociaux
        try:
            social_data = self.social_analyzer.get_product_sentiment(
                product_id=product_id,
                start_date=df.index.min(),
                end_date=df.index.max()
            )
            df = df.merge(social_data, left_index=True, right_index=True, how='left')
        except Exception as e:
            logger.warning(f"Données réseaux sociaux non disponibles: {e}")
        
        # Nettoyage des valeurs manquantes
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def train_prophet_model(self, data: pd.DataFrame, product_id: str) -> Prophet:
        """Entraîne un modèle Prophet pour un produit"""
        # Préparation des données pour Prophet
        prophet_data = data.reset_index()[['date', 'demand']].rename(
            columns={'date': 'ds', 'demand': 'y'}
        )
        
        # Configuration du modèle Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            uncertainty_samples=100
        )
        
        # Ajout des régresseurs externes
        external_features = ['is_holiday', 'is_promotion', 'temperature', 'sentiment_score']
        for feature in external_features:
            if feature in data.columns:
                model.add_regressor(feature)
                prophet_data[feature] = data[feature].values
        
        # Entraînement
        model.fit(prophet_data)
        
        return model
    
    def train_xgboost_model(self, data: pd.DataFrame, product_id: str) -> xgb.XGBRegressor:
        """Entraîne un modèle XGBoost pour un produit"""
        # Préparation des features
        feature_cols = [col for col in data.columns if col != 'demand']
        X = data[feature_cols]
        y = data['demand']
        
        # Configuration du modèle XGBoost
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Entraînement avec validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            score = mean_absolute_error(y_val, y_pred)
            scores.append(score)
        
        # Entraînement final sur toutes les données
        model.fit(X, y)
        
        # Sauvegarde de l'importance des features
        self.feature_importance[product_id] = dict(
            zip(feature_cols, model.feature_importances_)
        )
        
        logger.info(f"XGBoost {product_id}: MAE CV = {np.mean(scores):.2f}")
        
        return model
    
    def train_lstm_model(self, data: pd.DataFrame, product_id: str) -> tf.keras.Model:
        """Entraîne un modèle LSTM pour un produit"""
        # Préparation des données pour LSTM
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[['demand']])
        
        # Création des séquences
        X, y = [], []
        for i in range(self.lookback_window, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_window:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Division train/validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Architecture du modèle LSTM
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback_window, 1)),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        # Entraînement
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Sauvegarde du scaler
        self.scalers[product_id] = scaler
        
        logger.info(f"LSTM {product_id}: MAE final = {min(history.history['val_mae']):.2f}")
        
        return model
    
    async def train_models(self, data: Dict[str, pd.DataFrame]) -> None:
        """Entraîne tous les modèles pour tous les produits"""
        logger.info(f"🎯 Entraînement des modèles pour {len(data)} produits")
        
        for product_id, product_data in data.items():
            try:
                logger.info(f"Entraînement pour produit {product_id}")
                
                # Préparation des features
                enriched_data = self.prepare_features(product_data, product_id)
                
                # Entraînement Prophet
                prophet_model = self.train_prophet_model(enriched_data, product_id)
                if 'prophet' not in self.models:
                    self.models['prophet'] = {}
                self.models['prophet'][product_id] = prophet_model
                
                # Entraînement XGBoost
                xgb_model = self.train_xgboost_model(enriched_data, product_id)
                if 'xgboost' not in self.models:
                    self.models['xgboost'] = {}
                self.models['xgboost'][product_id] = xgb_model
                
                # Entraînement LSTM
                lstm_model = self.train_lstm_model(enriched_data, product_id)
                if 'lstm' not in self.models:
                    self.models['lstm'] = {}
                self.models['lstm'][product_id] = lstm_model
                
                logger.info(f"✅ Modèles entraînés pour {product_id}")
                
            except Exception as e:
                logger.error(f"❌ Erreur entraînement {product_id}: {e}")
        
        # Sauvegarde des modèles
        await self.save_models()
    
    def predict_prophet(self, product_id: str, future_dates: pd.DatetimeIndex) -> np.ndarray:
        """Prédiction avec Prophet"""
        if product_id not in self.models.get('prophet', {}):
            raise ValueError(f"Modèle Prophet non disponible pour {product_id}")
        
        model = self.models['prophet'][product_id]
        
        # Création du dataframe futur
        future = pd.DataFrame({'ds': future_dates})
        
        # Ajout des régresseurs externes si nécessaires
        # (simplification - en réalité il faudrait prédire ces valeurs aussi)
        for regressor in model.extra_regressors:
            future[regressor] = 0  # Valeur par défaut
        
        # Prédiction
        forecast = model.predict(future)
        
        return forecast['yhat'].values
    
    def predict_xgboost(self, product_id: str, features: pd.DataFrame) -> np.ndarray:
        """Prédiction avec XGBoost"""
        if product_id not in self.models.get('xgboost', {}):
            raise ValueError(f"Modèle XGBoost non disponible pour {product_id}")
        
        model = self.models['xgboost'][product_id]
        
        # Prédiction
        predictions = model.predict(features)
        
        return predictions
    
    def predict_lstm(self, product_id: str, sequence: np.ndarray) -> float:
        """Prédiction avec LSTM"""
        if product_id not in self.models.get('lstm', {}):
            raise ValueError(f"Modèle LSTM non disponible pour {product_id}")
        
        model = self.models['lstm'][product_id]
        scaler = self.scalers[product_id]
        
        # Normalisation
        scaled_sequence = scaler.transform(sequence.reshape(-1, 1))
        
        # Reshape pour LSTM
        X = scaled_sequence[-self.lookback_window:].reshape(1, self.lookback_window, 1)
        
        # Prédiction
        scaled_pred = model.predict(X, verbose=0)
        
        # Dénormalisation
        prediction = scaler.inverse_transform(scaled_pred)[0, 0]
        
        return prediction
    
    async def forecast_demand(
        self, 
        product_id: str, 
        historical_data: pd.DataFrame,
        forecast_days: int = None
    ) -> Dict[str, Any]:
        """
        Prévision de demande avec ensemble de modèles
        Retourne prédictions, intervalles de confiance et métriques
        """
        if forecast_days is None:
            forecast_days = self.forecast_horizon
        
        try:
            # Préparation des données
            enriched_data = self.prepare_features(historical_data, product_id)
            
            # Dates futures
            last_date = historical_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            predictions = {}
            
            # Prédiction Prophet
            try:
                prophet_pred = self.predict_prophet(product_id, future_dates)
                predictions['prophet'] = prophet_pred
            except Exception as e:
                logger.warning(f"Prophet prediction failed for {product_id}: {e}")
            
            # Prédiction XGBoost (nécessite des features futures)
            try:
                # Simplification: utilisation des dernières valeurs comme proxy
                last_features = enriched_data.iloc[-1:].drop(columns=['demand'])
                xgb_features = pd.concat([last_features] * forecast_days, ignore_index=True)
                xgb_pred = self.predict_xgboost(product_id, xgb_features)
                predictions['xgboost'] = xgb_pred
            except Exception as e:
                logger.warning(f"XGBoost prediction failed for {product_id}: {e}")
            
            # Prédiction LSTM
            try:
                lstm_predictions = []
                last_sequence = historical_data['demand'].values[-self.lookback_window:]
                
                for _ in range(forecast_days):
                    next_pred = self.predict_lstm(product_id, last_sequence)
                    lstm_predictions.append(next_pred)
                    # Mise à jour de la séquence
                    last_sequence = np.append(last_sequence[1:], next_pred)
                
                predictions['lstm'] = np.array(lstm_predictions)
            except Exception as e:
                logger.warning(f"LSTM prediction failed for {product_id}: {e}")
            
            # Ensemble prediction (moyenne pondérée)
            weights = {'prophet': 0.4, 'xgboost': 0.4, 'lstm': 0.2}
            
            ensemble_pred = np.zeros(forecast_days)
            total_weight = 0
            
            for model_name, pred in predictions.items():
                if model_name in weights:
                    ensemble_pred += weights[model_name] * pred
                    total_weight += weights[model_name]
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            # Calcul des intervalles de confiance (simplifié)
            historical_errors = self._calculate_model_errors(product_id, historical_data)
            confidence_interval = 1.96 * np.std(historical_errors)  # 95% CI
            
            result = {
                'product_id': product_id,
                'forecast_dates': future_dates.tolist(),
                'predictions': {
                    'ensemble': ensemble_pred.tolist(),
                    **{k: v.tolist() for k, v in predictions.items()}
                },
                'confidence_interval': confidence_interval,
                'upper_bound': (ensemble_pred + confidence_interval).tolist(),
                'lower_bound': (ensemble_pred - confidence_interval).tolist(),
                'model_weights': weights,
                'forecast_accuracy': self.model_performance.get(product_id, {}),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur prévision demande {product_id}: {e}")
            raise
    
    def _detect_holidays(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Détecte les jours fériés"""
        # Simplification - liste des jours fériés fixes
        holidays = [
            '01-01',  # Nouvel An
            '05-01',  # Fête du Travail
            '07-14',  # Fête Nationale
            '12-25'   # Noël
        ]
        
        return dates.strftime('%m-%d').isin(holidays).astype(int)
    
    def _detect_promotions(self, dates: pd.DatetimeIndex, product_id: str) -> pd.Series:
        """Détecte les périodes de promotion"""
        # Simplification - détection basée sur des patterns
        # En réalité, cela viendrait d'une base de données marketing
        return pd.Series(0, index=dates)
    
    def _detect_stockouts(self, data: pd.DataFrame) -> pd.Series:
        """Détecte les ruptures de stock"""
        # Détection basée sur des chutes anormales de demande
        demand = data['demand']
        rolling_mean = demand.rolling(window=7).mean()
        threshold = 0.3 * rolling_mean
        
        return (demand < threshold).astype(int)
    
    def _calculate_model_errors(self, product_id: str, data: pd.DataFrame) -> np.ndarray:
        """Calcule les erreurs historiques des modèles"""
        # Simplification - retourne des erreurs simulées
        # En réalité, cela viendrait d'une validation croisée
        return np.random.normal(0, 0.1, size=min(30, len(data)))
    
    async def save_models(self) -> None:
        """Sauvegarde tous les modèles entraînés"""
        try:
            model_paths = settings.get_model_paths()
            
            # Création des dossiers
            for path in model_paths.values():
                path.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde Prophet
            for product_id, model in self.models.get('prophet', {}).items():
                model_file = model_paths['prophet'] / f"{product_id}.pkl"
                joblib.dump(model, model_file)
            
            # Sauvegarde XGBoost
            for product_id, model in self.models.get('xgboost', {}).items():
                model_file = model_paths['xgboost'] / f"{product_id}.pkl"
                joblib.dump(model, model_file)
            
            # Sauvegarde LSTM
            for product_id, model in self.models.get('lstm', {}).items():
                model_dir = model_paths['lstm'] / product_id
                model_dir.mkdir(exist_ok=True)
                model.save(str(model_dir / "model.h5"))
            
            # Sauvegarde des scalers
            scalers_path = model_paths['prophet'].parent / "scalers"
            scalers_path.mkdir(exist_ok=True)
            for product_id, scaler in self.scalers.items():
                scaler_file = scalers_path / f"{product_id}.pkl"
                joblib.dump(scaler, scaler_file)
            
            logger.info("💾 Modèles sauvegardés avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde modèles: {e}")
    
    async def evaluate_models(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Évalue la performance des modèles sur des données de test"""
        evaluation_results = {}
        
        for product_id, data in test_data.items():
            try:
                # Prédiction sur les données de test
                forecast_result = await self.forecast_demand(
                    product_id=product_id,
                    historical_data=data[:-self.forecast_horizon],
                    forecast_days=self.forecast_horizon
                )
                
                # Calcul des métriques
                actual = data['demand'].values[-self.forecast_horizon:]
                predicted = np.array(forecast_result['predictions']['ensemble'])
                
                metrics = {
                    'mae': mean_absolute_error(actual, predicted),
                    'mse': mean_squared_error(actual, predicted),
                    'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                    'mape': np.mean(np.abs((actual - predicted) / actual)) * 100,
                    'r2': r2_score(actual, predicted)
                }
                
                evaluation_results[product_id] = metrics
                self.model_performance[product_id] = metrics
                
                logger.info(f"📊 {product_id}: MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")
                
            except Exception as e:
                logger.error(f"❌ Erreur évaluation {product_id}: {e}")
        
        return evaluation_results
