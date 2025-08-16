"""
Dashboard interactif Big Data Supply Chain
Interface de visualisation temps réel pour tous les KPIs
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Any

# Configuration
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Données simulées pour le dashboard
def generate_sample_data():
    """Génère des données d'exemple pour le dashboard"""
    
    # Données de demande
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    demand_data = []
    
    for i, date in enumerate(dates):
        base_demand = 1000
        seasonal = 200 * np.sin(2 * np.pi * i / 365)  # Saisonnalité annuelle
        weekly = 100 * np.sin(2 * np.pi * i / 7)      # Saisonnalité hebdomadaire
        noise = np.random.normal(0, 50)
        
        demand = max(0, base_demand + seasonal + weekly + noise)
        
        demand_data.append({
            'date': date,
            'actual_demand': demand,
            'predicted_demand': demand * (1 + np.random.normal(0, 0.05)),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'])
        })
    
    demand_df = pd.DataFrame(demand_data)
    
    # Données de transport
    transport_data = {
        'vehicle_utilization': 85.2,
        'delivery_success_rate': 96.8,
        'average_delivery_time': 2.3,
        'fuel_efficiency': 12.5,
        'co2_emissions': 245.8,
        'cost_per_km': 0.15
    }
    
    # Données de maintenance
    maintenance_data = [
        {'equipment_id': 'PUMP_001', 'status': 'Healthy', 'failure_risk': 0.15, 'last_maintenance': '2024-01-15'},
        {'equipment_id': 'CONV_002', 'status': 'Warning', 'failure_risk': 0.65, 'last_maintenance': '2023-11-20'},
        {'equipment_id': 'ROBOT_003', 'status': 'Critical', 'failure_risk': 0.85, 'last_maintenance': '2023-09-10'},
        {'equipment_id': 'PUMP_004', 'status': 'Healthy', 'failure_risk': 0.25, 'last_maintenance': '2024-02-01'},
    ]
    
    # Données de risques
    risk_data = [
        {'supplier': 'Supplier A', 'risk_score': 0.25, 'category': 'Financial'},
        {'supplier': 'Supplier B', 'risk_score': 0.75, 'category': 'Geopolitical'},
        {'supplier': 'Supplier C', 'risk_score': 0.15, 'category': 'Quality'},
        {'supplier': 'Supplier D', 'risk_score': 0.90, 'category': 'Delivery'},
    ]
    
    # KPIs globaux
    kpis = {
        'cost_reduction': 18.5,
        'delivery_performance': 94.2,
        'inventory_turnover': 11.8,
        'customer_satisfaction': 87.3,
        'sustainability_score': 72.1
    }
    
    return {
        'demand': demand_df,
        'transport': transport_data,
        'maintenance': maintenance_data,
        'risks': risk_data,
        'kpis': kpis
    }

def create_dash_app():
    """Crée l'application Dash"""
    
    # Initialisation de l'app
    app = dash.Dash(__name__, external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
    ])
    
    app.title = "Big Data Supply Chain Dashboard"
    
    # Génération des données
    data = generate_sample_data()
    
    # Styles CSS
    colors = {
        'background': '#f8f9fa',
        'text': '#2c3e50',
        'primary': '#3498db',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'card': '#ffffff'
    }
    
    # Layout principal
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("🚀 Big Data Supply Chain Optimization", 
                   style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
            html.P("Tableau de bord temps réel pour l'optimisation de la chaîne d'approvisionnement",
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
        ], style={'backgroundColor': colors['card'], 'padding': '20px', 'marginBottom': '20px', 'borderRadius': '10px'}),
        
        # KPIs Cards Row
        html.Div([
            html.Div([
                create_kpi_card("💰 Réduction Coûts", f"{data['kpis']['cost_reduction']:.1f}%", colors['success']),
            ], className="three columns"),
            
            html.Div([
                create_kpi_card("🚚 Performance Livraison", f"{data['kpis']['delivery_performance']:.1f}%", colors['primary']),
            ], className="three columns"),
            
            html.Div([
                create_kpi_card("📦 Rotation Stock", f"{data['kpis']['inventory_turnover']:.1f}x", colors['warning']),
            ], className="three columns"),
            
            html.Div([
                create_kpi_card("😊 Satisfaction Client", f"{data['kpis']['customer_satisfaction']:.1f}%", colors['success']),
            ], className="three columns"),
        ], className="row", style={'marginBottom': '30px'}),
        
        # Graphiques principaux
        html.Div([
            # Prévision de demande
            html.Div([
                html.H3("📈 Prévision Intelligente de la Demande", style={'color': colors['text']}),
                dcc.Graph(id='demand-forecast-chart'),
                dcc.Interval(id='demand-interval', interval=30*1000, n_intervals=0)  # Update every 30s
            ], className="six columns", style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
            
            # Transport Optimization
            html.Div([
                html.H3("🚛 Optimisation Transport", style={'color': colors['text']}),
                dcc.Graph(id='transport-metrics-chart'),
                dcc.Interval(id='transport-interval', interval=30*1000, n_intervals=0)
            ], className="six columns", style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginLeft': '20px'}),
        ], className="row", style={'marginBottom': '30px'}),
        
        # Maintenance et risques
        html.Div([
            # Maintenance prédictive
            html.Div([
                html.H3("🔧 Maintenance Prédictive", style={'color': colors['text']}),
                dash_table.DataTable(
                    id='maintenance-table',
                    columns=[
                        {'name': 'Équipement', 'id': 'equipment_id'},
                        {'name': 'Statut', 'id': 'status'},
                        {'name': 'Risque Panne', 'id': 'failure_risk', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                        {'name': 'Dernière Maintenance', 'id': 'last_maintenance'}
                    ],
                    data=data['maintenance'],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{status} = Critical'},
                            'backgroundColor': '#ffebee',
                            'color': colors['danger'],
                        },
                        {
                            'if': {'filter_query': '{status} = Warning'},
                            'backgroundColor': '#fff3e0',
                            'color': colors['warning'],
                        }
                    ]
                ),
                dcc.Interval(id='maintenance-interval', interval=60*1000, n_intervals=0)
            ], className="six columns", style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px'}),
            
            # Gestion des risques
            html.Div([
                html.H3("⚠️ Gestion des Risques", style={'color': colors['text']}),
                dcc.Graph(id='risk-analysis-chart'),
                dcc.Interval(id='risk-interval', interval=60*1000, n_intervals=0)
            ], className="six columns", style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginLeft': '20px'}),
        ], className="row", style={'marginBottom': '30px'}),
        
        # Analytics avancés
        html.Div([
            html.H3("📊 Analytics Avancés", style={'color': colors['text'], 'marginBottom': '20px'}),
            
            dcc.Tabs(id="analytics-tabs", value='demand-analysis', children=[
                dcc.Tab(label='Analyse Demande', value='demand-analysis'),
                dcc.Tab(label='Performance Transport', value='transport-analysis'),
                dcc.Tab(label='Optimisation Stock', value='inventory-analysis'),
                dcc.Tab(label='Impact Environnemental', value='sustainability-analysis'),
            ], style={'marginBottom': '20px'}),
            
            html.Div(id='analytics-content')
        ], style={'backgroundColor': colors['card'], 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
        
        # Footer
        html.Div([
            html.P(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                  style={'textAlign': 'center', 'color': '#95a5a6', 'margin': '0'}),
            html.P("🚀 Powered by Big Data & Machine Learning", 
                  style={'textAlign': 'center', 'color': '#95a5a6', 'margin': '5px 0 0 0'})
        ], style={'backgroundColor': colors['card'], 'padding': '15px', 'borderRadius': '10px'})
        
    ], style={'backgroundColor': colors['background'], 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
    
    # Callbacks pour les graphiques
    @app.callback(
        Output('demand-forecast-chart', 'figure'),
        [Input('demand-interval', 'n_intervals')]
    )
    def update_demand_chart(n):
        demand_df = data['demand'].tail(90)  # 3 derniers mois
        
        fig = go.Figure()
        
        # Demande réelle
        fig.add_trace(go.Scatter(
            x=demand_df['date'],
            y=demand_df['actual_demand'],
            mode='lines+markers',
            name='Demande Réelle',
            line=dict(color=colors['primary'], width=2),
            marker=dict(size=4)
        ))
        
        # Prédiction
        fig.add_trace(go.Scatter(
            x=demand_df['date'],
            y=demand_df['predicted_demand'],
            mode='lines',
            name='Prédiction ML',
            line=dict(color=colors['warning'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Évolution et Prédiction de la Demande',
            xaxis_title='Date',
            yaxis_title='Demande (unités)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @app.callback(
        Output('transport-metrics-chart', 'figure'),
        [Input('transport-interval', 'n_intervals')]
    )
    def update_transport_chart(n):
        metrics = data['transport']
        
        # Graphique en gauge pour les métriques de transport
        fig = go.Figure()
        
        # Utilisation véhicules
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = metrics['vehicle_utilization'],
            domain = {'row': 0, 'column': 0},
            title = {'text': "Utilisation Véhicules (%)"},
            delta = {'reference': 80},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': colors['primary']},
                     'steps' : [{'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 80], 'color': colors['warning']},
                                {'range': [80, 100], 'color': colors['success']}],
                     'threshold' : {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}
        ))
        
        fig.update_layout(
            grid = {'rows': 1, 'columns': 1, 'pattern': "independent"},
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('risk-analysis-chart', 'figure'),
        [Input('risk-interval', 'n_intervals')]
    )
    def update_risk_chart(n):
        risk_df = pd.DataFrame(data['risks'])
        
        # Graphique en barres pour les risques fournisseurs
        fig = px.bar(
            risk_df, 
            x='supplier', 
            y='risk_score',
            color='category',
            title='Analyse des Risques Fournisseurs',
            labels={'risk_score': 'Score de Risque', 'supplier': 'Fournisseur'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            template='plotly_white',
            height=400,
            yaxis={'range': [0, 1]}
        )
        
        # Ligne de seuil critique
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                      annotation_text="Seuil Critique")
        
        return fig
    
    @app.callback(
        Output('analytics-content', 'children'),
        [Input('analytics-tabs', 'value')]
    )
    def render_analytics_content(active_tab):
        if active_tab == 'demand-analysis':
            return create_demand_analysis_content(data)
        elif active_tab == 'transport-analysis':
            return create_transport_analysis_content(data)
        elif active_tab == 'inventory-analysis':
            return create_inventory_analysis_content(data)
        elif active_tab == 'sustainability-analysis':
            return create_sustainability_analysis_content(data)
        else:
            return html.Div("Contenu en cours de développement...")
    
    return app

def create_kpi_card(title, value, color):
    """Crée une carte KPI"""
    return html.Div([
        html.H4(title, style={'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '16px'}),
        html.H2(value, style={'color': color, 'marginBottom': '0', 'fontSize': '32px', 'fontWeight': 'bold'}),
    ], style={
        'backgroundColor': '#ffffff',
        'padding': '20px',
        'borderRadius': '10px',
        'textAlign': 'center',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'border': f'3px solid {color}'
    })

def create_demand_analysis_content(data):
    """Contenu de l'analyse de demande"""
    demand_df = data['demand']
    
    # Analyse par catégorie
    category_analysis = demand_df.groupby('product_category').agg({
        'actual_demand': ['mean', 'std', 'sum'],
        'predicted_demand': 'mean'
    }).round(2)
    
    return html.Div([
        html.H4("📊 Analyse Détaillée de la Demande"),
        
        html.Div([
            html.Div([
                html.H5("Répartition par Catégorie"),
                dcc.Graph(
                    figure=px.pie(
                        demand_df.groupby('product_category')['actual_demand'].sum().reset_index(),
                        values='actual_demand',
                        names='product_category',
                        title='Volume de Demande par Catégorie'
                    )
                )
            ], className="six columns"),
            
            html.Div([
                html.H5("Tendances Saisonnières"),
                dcc.Graph(
                    figure=px.box(
                        demand_df,
                        x=demand_df['date'].dt.month,
                        y='actual_demand',
                        title='Variations Saisonnières'
                    )
                )
            ], className="six columns"),
        ], className="row"),
        
        html.Div([
            html.H5("Métriques de Performance"),
            html.Ul([
                html.Li(f"Précision de prédiction: 94.2%"),
                html.Li(f"MAPE (Mean Absolute Percentage Error): 5.8%"),
                html.Li(f"Amélioration vs méthodes traditionnelles: +23%"),
                html.Li(f"Réduction des ruptures de stock: -35%")
            ])
        ])
    ])

def create_transport_analysis_content(data):
    """Contenu de l'analyse transport"""
    transport_data = data['transport']
    
    return html.Div([
        html.H4("🚛 Analyse Performance Transport"),
        
        html.Div([
            html.Div([
                html.H5("Métriques Clés"),
                html.Table([
                    html.Tr([html.Th("Métrique"), html.Th("Valeur"), html.Th("Objectif")]),
                    html.Tr([html.Td("Taux de livraison"), html.Td(f"{transport_data['delivery_success_rate']:.1f}%"), html.Td("95%")]),
                    html.Tr([html.Td("Temps moyen livraison"), html.Td(f"{transport_data['average_delivery_time']:.1f}j"), html.Td("2j")]),
                    html.Tr([html.Td("Efficacité carburant"), html.Td(f"{transport_data['fuel_efficiency']:.1f}L/100km"), html.Td("10L/100km")]),
                    html.Tr([html.Td("Coût par km"), html.Td(f"{transport_data['cost_per_km']:.2f}€"), html.Td("0.12€")])
                ], style={'width': '100%', 'border': '1px solid #ddd'})
            ], className="six columns"),
            
            html.Div([
                html.H5("Économies Réalisées"),
                html.Ul([
                    html.Li("💰 Réduction coûts transport: 15.2%"),
                    html.Li("🌱 Réduction émissions CO2: 18.7%"),
                    html.Li("⏱️ Amélioration délais: 22.1%"),
                    html.Li("📦 Optimisation charge: 85.2%")
                ])
            ], className="six columns"),
        ], className="row"),
        
        html.Div([
            html.H5("🎯 Recommandations"),
            html.Ol([
                html.Li("Consolider 3 routes faiblement chargées pour économiser 12% de coûts"),
                html.Li("Implémenter livraisons nocturnes pour éviter 15% du trafic"),
                html.Li("Optimiser maintenance véhicules pour améliorer efficacité de 8%"),
                html.Li("Négocier tarifs préférentiels avec transporteurs performants")
            ])
        ])
    ])

def create_inventory_analysis_content(data):
    """Contenu de l'analyse des stocks"""
    return html.Div([
        html.H4("📦 Optimisation des Stocks"),
        
        html.Div([
            html.Div([
                html.H5("Indicateurs Stock"),
                html.Table([
                    html.Tr([html.Th("Indicateur"), html.Th("Actuel"), html.Th("Cible")]),
                    html.Tr([html.Td("Rotation stock"), html.Td("11.8x"), html.Td("12x")]),
                    html.Tr([html.Td("Taux de rupture"), html.Td("2.1%"), html.Td("<2%")]),
                    html.Tr([html.Td("Stock de sécurité"), html.Td("15j"), html.Td("10j")]),
                    html.Tr([html.Td("Obsolescence"), html.Td("1.8%"), html.Td("<1%")])
                ], style={'width': '100%', 'border': '1px solid #ddd'})
            ], className="six columns"),
            
            html.Div([
                html.H5("Optimisations Identifiées"),
                html.Ul([
                    html.Li("🎯 Réduction stock sécurité: -25%"),
                    html.Li("📈 Amélioration rotation: +15%"),
                    html.Li("💸 Réduction obsolescence: -40%"),
                    html.Li("🔄 Optimisation réapprovisionnement: +20%")
                ])
            ], className="six columns"),
        ], className="row")
    ])

def create_sustainability_analysis_content(data):
    """Contenu de l'analyse environnementale"""
    return html.Div([
        html.H4("🌱 Impact Environnemental"),
        
        html.Div([
            html.Div([
                html.H5("Métriques Durabilité"),
                html.Table([
                    html.Tr([html.Th("Métrique"), html.Th("Valeur"), html.Th("Évolution")]),
                    html.Tr([html.Td("Émissions CO2"), html.Td("245.8 kg"), html.Td("-18.7%")]),
                    html.Tr([html.Td("Consommation carburant"), html.Td("12.5 L/100km"), html.Td("-12.3%")]),
                    html.Tr([html.Td("Efficacité énergétique"), html.Td("87.2%"), html.Td("+5.1%")]),
                    html.Tr([html.Td("Déchets"), html.Td("2.1 kg"), html.Td("-22.4%")])
                ], style={'width': '100%', 'border': '1px solid #ddd'})
            ], className="six columns"),
            
            html.Div([
                html.H5("Objectifs 2025"),
                html.Ul([
                    html.Li("🎯 Réduction CO2: -30%"),
                    html.Li("♻️ Économie circulaire: 50%"),
                    html.Li("🌿 Énergie renouvelable: 70%"),
                    html.Li("📊 Score ESG: >80")
                ])
            ], className="six columns"),
        ], className="row")
    ])

if __name__ == '__main__':
    app = create_dash_app()
    app.run_server(
        host=settings.DASH_HOST,
        port=settings.DASH_PORT,
        debug=settings.DASH_DEBUG
    )
