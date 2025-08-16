"""
Optimisation temps réel des transports
Implémentation d'algorithmes inspirés d'ORION (UPS) et des méthodes Amazon
Optimisation multi-objectifs : Coût + Délai + Empreinte carbone
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import math

# Optimisation
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import networkx as nx
from scipy.optimize import minimize
from sklearn.cluster import KMeans

# Configuration
from config.settings import settings, BusinessConstants
from utils.logger import setup_logger
from data_ingestion.external_data import TrafficDataCollector, WeatherDataCollector

logger = setup_logger(__name__)

@dataclass
class DeliveryPoint:
    """Point de livraison avec ses caractéristiques"""
    id: str
    latitude: float
    longitude: float
    demand: float
    time_window_start: datetime
    time_window_end: datetime
    service_time: int  # minutes
    priority: int = 1  # 1=normal, 2=urgent, 3=critique

@dataclass
class Vehicle:
    """Véhicule avec ses capacités et contraintes"""
    id: str
    capacity: float
    start_location: Tuple[float, float]
    end_location: Tuple[float, float]
    available_start: datetime
    available_end: datetime
    cost_per_km: float
    co2_per_km: float
    speed_kmh: float = 50.0
    max_driving_time: int = 480  # minutes

@dataclass
class Route:
    """Route optimisée avec métriques"""
    vehicle_id: str
    stops: List[DeliveryPoint]
    total_distance: float
    total_time: int
    total_cost: float
    co2_emissions: float
    load_factor: float
    delivery_windows_respected: int

class TransportOptimizer:
    """
    Optimiseur de transport temps réel
    
    Features:
    - Routage dynamique avec contraintes multiples
    - Optimisation multi-objectifs (coût, délai, CO2)
    - Prédiction des délais avec trafic et météo
    - Réoptimisation en temps réel
    - Clustering intelligent des livraisons
    """
    
    def __init__(self):
        self.traffic_collector = TrafficDataCollector()
        self.weather_collector = WeatherDataCollector()
        self.executor = ThreadPoolExecutor(max_workers=settings.OPTIMIZATION_THREADS)
        
        # Cache des distances et temps de trajet
        self.distance_cache = {}
        self.time_cache = {}
        
        # Paramètres d'optimisation
        self.optimization_weights = {
            'cost': 0.4,
            'time': 0.3,
            'co2': 0.2,
            'service_quality': 0.1
        }
        
        logger.info("🚚 TransportOptimizer initialisé")
    
    async def initialize(self):
        """Initialise les services externes"""
        try:
            await self.traffic_collector.initialize()
            await self.weather_collector.initialize()
            logger.info("✅ Services externes initialisés")
        except Exception as e:
            logger.warning(f"⚠️ Erreur initialisation services: {e}")
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calcule la distance géodésique entre deux points"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Cache key
        cache_key = f"{lat1:.6f},{lon1:.6f}-{lat2:.6f},{lon2:.6f}"
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Formule de Haversine
        R = 6371  # Rayon de la Terre en km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # Cache du résultat
        self.distance_cache[cache_key] = distance
        
        return distance
    
    async def calculate_travel_time(
        self, 
        origin: Tuple[float, float], 
        destination: Tuple[float, float],
        departure_time: datetime
    ) -> Tuple[int, float]:
        """
        Calcule le temps de trajet avec conditions de trafic
        Retourne (temps en minutes, facteur de trafic)
        """
        try:
            # Récupération des conditions de trafic
            traffic_factor = await self.traffic_collector.get_traffic_factor(
                origin, destination, departure_time
            )
            
            # Récupération des conditions météo
            weather_factor = await self.weather_collector.get_weather_impact(
                destination, departure_time
            )
            
            # Distance de base
            distance = self.calculate_distance(origin, destination)
            
            # Vitesse ajustée selon conditions
            base_speed = 50.0  # km/h en ville
            adjusted_speed = base_speed * traffic_factor * weather_factor
            
            # Temps de trajet en minutes
            travel_time = (distance / adjusted_speed) * 60
            
            return int(travel_time), traffic_factor
            
        except Exception as e:
            logger.warning(f"Erreur calcul temps trajet: {e}")
            # Fallback sur calcul simple
            distance = self.calculate_distance(origin, destination)
            travel_time = (distance / 50.0) * 60  # 50 km/h par défaut
            return int(travel_time), 1.0
    
    def create_distance_matrix(self, locations: List[Tuple[float, float]]) -> np.ndarray:
        """Crée une matrice des distances entre tous les points"""
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.calculate_distance(locations[i], locations[j])
        
        return matrix
    
    async def create_time_matrix(
        self, 
        locations: List[Tuple[float, float]], 
        departure_time: datetime
    ) -> np.ndarray:
        """Crée une matrice des temps de trajet entre tous les points"""
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    travel_time, _ = await self.calculate_travel_time(
                        locations[i], locations[j], departure_time
                    )
                    matrix[i][j] = travel_time
        
        return matrix
    
    def cluster_deliveries(self, delivery_points: List[DeliveryPoint], n_clusters: int = None) -> Dict[int, List[DeliveryPoint]]:
        """
        Groupe les livraisons par zones géographiques
        Optimise le nombre de véhicules nécessaires
        """
        if not delivery_points:
            return {}
        
        # Extraction des coordonnées
        coordinates = np.array([
            [point.latitude, point.longitude] for point in delivery_points
        ])
        
        # Détermination automatique du nombre de clusters
        if n_clusters is None:
            # Règle empirique basée sur la capacité des véhicules
            total_demand = sum(point.demand for point in delivery_points)
            avg_vehicle_capacity = settings.MAX_VEHICLE_CAPACITY
            n_clusters = max(1, int(np.ceil(total_demand / avg_vehicle_capacity)))
        
        # Clustering K-means
        if len(delivery_points) <= n_clusters:
            # Si moins de points que de clusters, un cluster par point
            clusters = {}
            for i, point in enumerate(delivery_points):
                clusters[i] = [point]
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            
            # Regroupement par cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(delivery_points[i])
        
        logger.info(f"📍 {len(delivery_points)} livraisons regroupées en {len(clusters)} clusters")
        
        return clusters
    
    def solve_vrp_cluster(
        self, 
        delivery_points: List[DeliveryPoint], 
        vehicles: List[Vehicle],
        distance_matrix: np.ndarray,
        time_matrix: np.ndarray
    ) -> List[Route]:
        """
        Résout le problème de routage pour un cluster
        Utilise OR-Tools pour l'optimisation
        """
        if not delivery_points or not vehicles:
            return []
        
        # Préparation des données OR-Tools
        data = self._prepare_ortools_data(delivery_points, vehicles, distance_matrix, time_matrix)
        
        # Création du gestionnaire de routage
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            len(data['vehicles']),
            data['depot']
        )
        
        # Création du modèle de routage
        routing = pywrapcp.RoutingModel(manager)
        
        # Fonction de distance
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Contraintes de capacité
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Contraintes de temps
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node] + data['service_times'][from_node]
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            30,  # allow waiting time
            data['max_time'],  # maximum time per vehicle
            False,  # don't force start cumul to zero
            'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Fenêtres de temps
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == data['depot']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        
        # Fenêtres de temps pour le dépôt
        depot_idx = data['depot']
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data['time_windows'][depot_idx][0],
                data['time_windows'][depot_idx][1]
            )
        
        # Paramètres de recherche
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(30)
        
        # Résolution
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._extract_routes(data, manager, routing, solution, delivery_points, vehicles)
        else:
            logger.warning("❌ Aucune solution trouvée pour le cluster")
            return []
    
    def _prepare_ortools_data(
        self, 
        delivery_points: List[DeliveryPoint], 
        vehicles: List[Vehicle],
        distance_matrix: np.ndarray,
        time_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Prépare les données pour OR-Tools"""
        # Le dépôt est à l'index 0
        depot = 0
        
        # Demandes (le dépôt a une demande de 0)
        demands = [0] + [int(point.demand) for point in delivery_points]
        
        # Capacités des véhicules
        vehicle_capacities = [int(vehicle.capacity) for vehicle in vehicles]
        
        # Temps de service (en minutes)
        service_times = [0] + [point.service_time for point in delivery_points]
        
        # Fenêtres de temps (en minutes depuis minuit)
        def time_to_minutes(dt: datetime) -> int:
            return dt.hour * 60 + dt.minute
        
        # Fenêtre du dépôt (toute la journée)
        time_windows = [(0, 24 * 60)]
        
        for point in delivery_points:
            start_min = time_to_minutes(point.time_window_start)
            end_min = time_to_minutes(point.time_window_end)
            time_windows.append((start_min, end_min))
        
        return {
            'distance_matrix': distance_matrix.astype(int),
            'time_matrix': time_matrix.astype(int),
            'demands': demands,
            'vehicle_capacities': vehicle_capacities,
            'service_times': service_times,
            'time_windows': time_windows,
            'num_vehicles': len(vehicles),
            'depot': depot,
            'max_time': 8 * 60,  # 8 heures max par tournée
            'vehicles': vehicles
        }
    
    def _extract_routes(
        self,
        data: Dict[str, Any],
        manager: pywrapcp.RoutingIndexManager,
        routing: pywrapcp.RoutingModel,
        solution: pywrapcp.Assignment,
        delivery_points: List[DeliveryPoint],
        vehicles: List[Vehicle]
    ) -> List[Route]:
        """Extrait les routes de la solution OR-Tools"""
        routes = []
        
        for vehicle_idx in range(data['num_vehicles']):
            index = routing.Start(vehicle_idx)
            route_stops = []
            route_distance = 0
            route_load = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                if node_index != data['depot']:
                    # Point de livraison (index - 1 car le dépôt est à 0)
                    delivery_point = delivery_points[node_index - 1]
                    route_stops.append(delivery_point)
                    route_load += delivery_point.demand
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                
                if previous_index != index:
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
            
            if route_stops:
                vehicle = vehicles[vehicle_idx]
                
                # Calcul des métriques
                total_cost = route_distance * vehicle.cost_per_km
                co2_emissions = route_distance * vehicle.co2_per_km
                load_factor = route_load / vehicle.capacity
                
                # Temps total (distance + service)
                total_time = route_distance / vehicle.speed_kmh * 60  # minutes
                total_time += sum(stop.service_time for stop in route_stops)
                
                # Vérification des fenêtres de temps
                windows_respected = len(route_stops)  # Simplifié
                
                route = Route(
                    vehicle_id=vehicle.id,
                    stops=route_stops,
                    total_distance=route_distance,
                    total_time=int(total_time),
                    total_cost=total_cost,
                    co2_emissions=co2_emissions,
                    load_factor=load_factor,
                    delivery_windows_respected=windows_respected
                )
                
                routes.append(route)
        
        return routes
    
    async def optimize_routes(
        self, 
        delivery_points: List[DeliveryPoint], 
        vehicles: List[Vehicle]
    ) -> List[Route]:
        """
        Optimise les routes pour tous les points de livraison
        Pipeline complet d'optimisation
        """
        logger.info(f"🎯 Optimisation de {len(delivery_points)} livraisons avec {len(vehicles)} véhicules")
        
        try:
            # Étape 1: Clustering des livraisons
            clusters = self.cluster_deliveries(delivery_points, len(vehicles))
            
            # Étape 2: Optimisation par cluster
            all_routes = []
            
            for cluster_id, cluster_points in clusters.items():
                if not cluster_points:
                    continue
                
                logger.info(f"Optimisation cluster {cluster_id}: {len(cluster_points)} points")
                
                # Préparation des locations (dépôt + livraisons)
                depot_location = vehicles[0].start_location
                locations = [depot_location] + [
                    (point.latitude, point.longitude) for point in cluster_points
                ]
                
                # Matrices de distance et temps
                distance_matrix = self.create_distance_matrix(locations)
                time_matrix = await self.create_time_matrix(locations, datetime.now())
                
                # Sélection du véhicule pour ce cluster
                if cluster_id < len(vehicles):
                    cluster_vehicles = [vehicles[cluster_id]]
                else:
                    cluster_vehicles = [vehicles[0]]  # Fallback
                
                # Résolution VRP
                cluster_routes = self.solve_vrp_cluster(
                    cluster_points,
                    cluster_vehicles,
                    distance_matrix,
                    time_matrix
                )
                
                all_routes.extend(cluster_routes)
            
            # Étape 3: Évaluation globale
            self._evaluate_solution(all_routes)
            
            logger.info(f"✅ Optimisation terminée: {len(all_routes)} routes générées")
            
            return all_routes
            
        except Exception as e:
            logger.error(f"❌ Erreur optimisation routes: {e}")
            raise
    
    def _evaluate_solution(self, routes: List[Route]) -> Dict[str, float]:
        """Évalue la qualité de la solution d'optimisation"""
        if not routes:
            return {}
        
        # Métriques globales
        total_distance = sum(route.total_distance for route in routes)
        total_cost = sum(route.total_cost for route in routes)
        total_co2 = sum(route.co2_emissions for route in routes)
        avg_load_factor = np.mean([route.load_factor for route in routes])
        
        # Points livrés
        total_deliveries = sum(len(route.stops) for route in routes)
        
        # Respect des fenêtres de temps
        total_windows = sum(len(route.stops) for route in routes)
        respected_windows = sum(route.delivery_windows_respected for route in routes)
        window_respect_rate = respected_windows / total_windows if total_windows > 0 else 0
        
        metrics = {
            'total_distance_km': total_distance,
            'total_cost_euros': total_cost,
            'total_co2_kg': total_co2,
            'average_load_factor': avg_load_factor,
            'deliveries_count': total_deliveries,
            'vehicles_used': len(routes),
            'window_respect_rate': window_respect_rate,
            'cost_per_delivery': total_cost / total_deliveries if total_deliveries > 0 else 0,
            'co2_per_delivery': total_co2 / total_deliveries if total_deliveries > 0 else 0
        }
        
        logger.info(f"📊 Solution: {total_deliveries} livraisons, {total_distance:.1f}km, {total_cost:.2f}€, {total_co2:.1f}kg CO2")
        
        return metrics
    
    async def reoptimize_routes(
        self, 
        current_routes: List[Route], 
        new_deliveries: List[DeliveryPoint],
        traffic_updates: Dict[str, float] = None
    ) -> List[Route]:
        """
        Réoptimise les routes en temps réel
        Prend en compte nouvelles livraisons et conditions de trafic
        """
        logger.info(f"🔄 Réoptimisation avec {len(new_deliveries)} nouvelles livraisons")
        
        try:
            # Extraction des livraisons non encore effectuées
            pending_deliveries = []
            vehicles = []
            
            for route in current_routes:
                # Simulation: on suppose que certaines livraisons sont encore à faire
                pending_from_route = route.stops[len(route.stops)//2:]  # Dernière moitié
                pending_deliveries.extend(pending_from_route)
                
                # Reconstitution du véhicule
                vehicle = Vehicle(
                    id=route.vehicle_id,
                    capacity=settings.MAX_VEHICLE_CAPACITY,
                    start_location=(45.764, 4.835),  # Lyon par défaut
                    end_location=(45.764, 4.835),
                    available_start=datetime.now(),
                    available_end=datetime.now() + timedelta(hours=8),
                    cost_per_km=settings.FUEL_COST_PER_KM,
                    co2_per_km=settings.CO2_EMISSION_KG_PER_KM
                )
                vehicles.append(vehicle)
            
            # Ajout des nouvelles livraisons
            all_deliveries = pending_deliveries + new_deliveries
            
            # Réoptimisation complète
            new_routes = await self.optimize_routes(all_deliveries, vehicles)
            
            return new_routes
            
        except Exception as e:
            logger.error(f"❌ Erreur réoptimisation: {e}")
            return current_routes
    
    def calculate_savings(self, baseline_routes: List[Route], optimized_routes: List[Route]) -> Dict[str, float]:
        """Calcule les gains de l'optimisation"""
        baseline_metrics = self._evaluate_solution(baseline_routes)
        optimized_metrics = self._evaluate_solution(optimized_routes)
        
        savings = {}
        for metric in ['total_distance_km', 'total_cost_euros', 'total_co2_kg']:
            if metric in baseline_metrics and metric in optimized_metrics:
                baseline_value = baseline_metrics[metric]
                optimized_value = optimized_metrics[metric]
                
                if baseline_value > 0:
                    savings[f'{metric}_reduction_percent'] = (
                        (baseline_value - optimized_value) / baseline_value * 100
                    )
                    savings[f'{metric}_absolute_savings'] = baseline_value - optimized_value
        
        return savings
    
    async def monitor_route_execution(self, route: Route) -> Dict[str, Any]:
        """
        Surveille l'exécution d'une route en temps réel
        Détecte les écarts et propose des ajustements
        """
        try:
            monitoring_data = {
                'route_id': route.vehicle_id,
                'status': 'in_progress',
                'completed_stops': 0,
                'current_delay': 0,
                'traffic_alerts': [],
                'weather_alerts': [],
                'recommendations': []
            }
            
            # Simulation de données de monitoring
            # En réalité, cela viendrait de capteurs IoT, GPS, etc.
            
            current_time = datetime.now()
            
            # Vérification des conditions de trafic pour les prochains segments
            for i, stop in enumerate(route.stops[:-1]):
                next_stop = route.stops[i + 1]
                
                origin = (stop.latitude, stop.longitude)
                destination = (next_stop.latitude, next_stop.longitude)
                
                travel_time, traffic_factor = await self.calculate_travel_time(
                    origin, destination, current_time
                )
                
                if traffic_factor < 0.7:  # Trafic dense
                    monitoring_data['traffic_alerts'].append({
                        'segment': f"{stop.id} -> {next_stop.id}",
                        'traffic_factor': traffic_factor,
                        'recommendation': "Considérer route alternative"
                    })
            
            return monitoring_data
            
        except Exception as e:
            logger.error(f"❌ Erreur monitoring route: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_route_report(self, routes: List[Route]) -> Dict[str, Any]:
        """Génère un rapport détaillé des routes optimisées"""
        
        if not routes:
            return {'error': 'Aucune route à analyser'}
        
        # Métriques générales
        solution_metrics = self._evaluate_solution(routes)
        
        # Analyse par véhicule
        vehicle_analysis = []
        for route in routes:
            analysis = {
                'vehicle_id': route.vehicle_id,
                'stops_count': len(route.stops),
                'total_distance_km': round(route.total_distance, 2),
                'total_time_hours': round(route.total_time / 60, 2),
                'total_cost_euros': round(route.total_cost, 2),
                'co2_emissions_kg': round(route.co2_emissions, 2),
                'load_factor_percent': round(route.load_factor * 100, 1),
                'efficiency_score': self._calculate_route_efficiency(route)
            }
            vehicle_analysis.append(analysis)
        
        # Recommandations d'amélioration
        recommendations = self._generate_recommendations(routes)
        
        report = {
            'summary': solution_metrics,
            'vehicle_routes': vehicle_analysis,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat(),
            'total_savings_potential': self._estimate_savings_potential(routes)
        }
        
        return report
    
    def _calculate_route_efficiency(self, route: Route) -> float:
        """Calcule un score d'efficacité pour une route"""
        # Score basé sur plusieurs facteurs
        load_score = min(route.load_factor * 1.25, 1.0)  # Bonus si bien chargé
        time_efficiency = 1.0 - (route.total_time / (8 * 60))  # Ratio temps utilisé
        
        efficiency = (load_score + time_efficiency) / 2 * 100
        return round(efficiency, 1)
    
    def _generate_recommendations(self, routes: List[Route]) -> List[str]:
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        # Analyse des facteurs de charge
        low_load_routes = [r for r in routes if r.load_factor < 0.6]
        if low_load_routes:
            recommendations.append(
                f"Consolider {len(low_load_routes)} routes peu chargées (<60%)"
            )
        
        # Analyse des temps de trajet
        long_routes = [r for r in routes if r.total_time > 7 * 60]  # > 7h
        if long_routes:
            recommendations.append(
                f"Optimiser {len(long_routes)} routes longues (>7h)"
            )
        
        # Analyse des émissions
        high_emission_routes = [r for r in routes if r.co2_emissions > 50]
        if high_emission_routes:
            recommendations.append(
                f"Réduire les émissions de {len(high_emission_routes)} routes (>50kg CO2)"
            )
        
        return recommendations
    
    def _estimate_savings_potential(self, routes: List[Route]) -> Dict[str, float]:
        """Estime le potentiel d'économies supplémentaires"""
        
        # Économies potentielles par consolidation
        underutilized_routes = [r for r in routes if r.load_factor < 0.7]
        consolidation_savings = len(underutilized_routes) * 0.15  # 15% par route
        
        # Économies par optimisation des temps
        inefficient_routes = [r for r in routes if r.total_time > 6 * 60]
        time_savings = len(inefficient_routes) * 0.10  # 10% par route
        
        return {
            'consolidation_potential_percent': round(consolidation_savings * 100, 1),
            'time_optimization_potential_percent': round(time_savings * 100, 1),
            'estimated_cost_reduction_euros': round(
                sum(r.total_cost for r in routes) * (consolidation_savings + time_savings), 2
            )
        }
