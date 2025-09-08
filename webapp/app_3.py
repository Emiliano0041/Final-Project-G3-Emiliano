import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time
from scipy import spatial
import pickle
import hashlib
from datetime import datetime
import os
import json

# Configuración básica
st.set_page_config(
    page_title="Barcelona Accident Predictor",
    page_icon="🚗",
    layout="wide"
)

# EVITAR BUCLES - Inicializar estado de sesión UNA SOLA VEZ
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.route_data = {
        'ruta_directa': None,
        'riesgo_directo': None,
        'dist_directa': None,
        'dur_directa': None,
        'ruta_segura': None,
        'riesgo_seguro': None,
        'dist_segura': None,
        'dur_segura': None,
        'origen_coords': None,
        'destino_coords': None,
        'last_calculation': None,
        'rutas_diferentes': False
    }
    st.session_state.loaded_models = {}
    # Inicializar favoritos
    st.session_state.rutas_favoritas = {}

# TODAS LAS FUNCIONES DEFINIDAS AQUÍ AL PRINCIPIO
def load_base_dataframe():
    """Carga el DataFrame base - VERSIÓN SIMPLE"""
    try:
        df_path = r"C:\Users\emili\sp-ml-17-final-project-g3\notebooks\df_processed.csv"
        
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            st.success(f"DataFrame cargado: {len(df)} filas")
            
            # LIMITAR INMEDIATAMENTE para evitar problemas
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)
                st.warning("Limitado a 1000 puntos para mejor rendimiento")
            
            return df
        else:
            st.error("Archivo no encontrado")
            # DataFrame mínimo por defecto
            lat_range = np.linspace(41.35, 41.45, 10)
            lon_range = np.linspace(2.1, 2.25, 10)
            grid_points = [(lat, lon) for lat in lat_range for lon in lon_range]
            return pd.DataFrame(grid_points, columns=['lat', 'lon'])
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame({'lat': [41.3851], 'lon': [2.1734]})

def load_model_simple():
    """Carga el modelo - VERSIÓN SIMPLE"""
    try:
        model_path = r"C:\Users\emili\sp-ml-17-final-project-g3\notebooks\modelo_emiliano_new.pkl"
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            st.success("Modelo cargado exitosamente")
            return model
        else:
            st.error("Modelo no encontrado")
            return None
            
    except Exception as e:
        st.error(f"Error cargando modelo: {str(e)}")
        return None

def get_default_weather():
    """Valores meteorológicos por defecto para Barcelona"""
    return {
        'temperature_2m (°C)': 18.0,
        'precipitation (mm)': 0.0,
        'wind_speed_10m (km/h)': 12.0,
        'humidity': 60.0,
        'lluvia': 0,
        'viento_fuerte': 0,
        'temp_extrema': 0
    }

def get_weather_data_safe(fecha, hora):
    """Obtiene datos meteorológicos de OpenMeteo - VERSIÓN SEGURA"""
    try:
        # Solo intentar si no está en cache reciente
        cache_key = f"weather_{fecha}_{hora}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        # Coordenadas de Barcelona
        lat, lon = 41.3851, 2.1734
        date_str = fecha.strftime('%Y-%m-%d')
        
        # URL con timeout muy corto para evitar bloqueos
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m',
            'start_date': date_str,
            'end_date': date_str,
            'timezone': 'Europe/Madrid'
        }
        
        # Timeout muy corto para evitar bloqueos
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'hourly' in data and hora < len(data['hourly']['temperature_2m']):
                weather_info = {
                    'temperature_2m (°C)': data['hourly']['temperature_2m'][hora] or 18.0,
                    'precipitation (mm)': data['hourly']['precipitation'][hora] or 0.0,
                    'wind_speed_10m (km/h)': data['hourly']['wind_speed_10m'][hora] or 12.0,
                    'humidity': data['hourly']['relative_humidity_2m'][hora] or 60.0,
                }
                
                # Variables derivadas
                weather_info['lluvia'] = 1 if weather_info['precipitation (mm)'] > 0.5 else 0
                weather_info['viento_fuerte'] = 1 if weather_info['wind_speed_10m (km/h)'] > 25 else 0
                weather_info['temp_extrema'] = 1 if weather_info['temperature_2m (°C)'] < 5 or weather_info['temperature_2m (°C)'] > 30 else 0
                
                # Guardar en cache
                st.session_state[cache_key] = weather_info
                return weather_info
        
        # Si falla, usar valores por defecto
        return get_default_weather()
        
    except Exception as e:
        # Si hay cualquier error, usar valores por defecto silenciosamente
        return get_default_weather()

def get_risk_color(risk_level):
    """Retorna color según nivel de riesgo"""
    if risk_level >= 0.15:
        return "#dc3545", "Alto"
    elif risk_level >= 0.08:
        return "#ffc107", "Medio"
    else:
        return "#28a745", "Bajo"

# FUNCIONES PARA RUTAS SEGURAS
def geocode_address_safe(address):
    """Geocodifica una dirección de forma segura"""
    try:
        # Cache para geocodificación
        cache_key = f"geocode_{address}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        geolocator = Nominatim(user_agent="barcelona_predictor", timeout=10)
        
        location = geolocator.geocode(f"{address}, Barcelona, Spain")
        if location:
            coords = (location.latitude, location.longitude)
            st.session_state[cache_key] = coords
            return coords
        else:
            return None, None
            
    except Exception as e:
        st.warning(f"Error en geocodificación: {str(e)}")
        return None, None

def get_route_osrm_safe(start_lat, start_lon, end_lat, end_lon):
    """Obtiene ruta usando OSRM de forma segura"""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('code') == 'Ok' and 'routes' in data:
                route = data['routes'][0]
                coordinates = route['geometry']['coordinates']
                route_coords = [[coord[1], coord[0]] for coord in coordinates]
                distance = route['distance'] / 1000  # km
                duration = route['duration'] / 60     # minutos
                return route_coords, distance, duration
        
        return None, None, None
        
    except Exception as e:
        st.error(f"Error obteniendo ruta: {str(e)}")
        return None, None, None

# NUEVA FUNCIÓN PARA RUTAS ALTERNATIVAS SEGURAS
def get_alternative_safe_route(start_lat, start_lon, end_lat, end_lon, risk_points_df, threshold=0.1):
    """
    Obtiene una ruta alternativa evitando puntos de alto riesgo
    """
    try:
        # Primero intentar obtener alternativas de OSRM
        url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson&alternatives=true&steps=true"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('code') == 'Ok' and 'routes' in data and len(data['routes']) > 1:
                # Si hay rutas alternativas, evaluar cuál tiene menor riesgo
                best_route = None
                min_risk = float('inf')
                best_distance = None
                best_duration = None
                
                for route in data['routes']:
                    coordinates = route['geometry']['coordinates']
                    route_coords = [[coord[1], coord[0]] for coord in coordinates]
                    
                    # Calcular riesgo de esta ruta
                    route_risk = calculate_route_risk_simple(route_coords, risk_points_df, 0.5)
                    
                    if route_risk < min_risk:
                        min_risk = route_risk
                        best_route = route_coords
                        best_distance = route['distance'] / 1000
                        best_duration = route['duration'] / 60
                
                if best_route:
                    return best_route, best_distance, best_duration
        
        # Si no hay alternativas, intentar con waypoint seguro
        return create_route_with_safe_waypoint(start_lat, start_lon, end_lat, end_lon, risk_points_df, threshold)
        
    except Exception as e:
        st.warning(f"Error calculando ruta segura: {str(e)}")
        return None, None, None

def create_route_with_safe_waypoint(start_lat, start_lon, end_lat, end_lon, risk_points_df, threshold):
    """
    Crea una ruta con un waypoint intermedio que evite zonas de riesgo
    """
    try:
        # Filtrar puntos de alto riesgo
        high_risk = risk_points_df[risk_points_df['y_proba'] > threshold]
        
        if high_risk.empty:
            return get_route_osrm_safe(start_lat, start_lon, end_lat, end_lon)
        
        # Encontrar punto medio
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        
        # Buscar waypoint seguro alejado de puntos de riesgo
        offset = 0.01  # Aproximadamente 1km
        best_waypoint = None
        min_total_risk = float('inf')
        
        # Probar diferentes waypoints
        for lat_offset in [-offset, 0, offset]:
            for lon_offset in [-offset, 0, offset]:
                if lat_offset == 0 and lon_offset == 0:
                    continue
                    
                waypoint_lat = mid_lat + lat_offset
                waypoint_lon = mid_lon + lon_offset
                
                # Calcular riesgo acumulado cerca de este waypoint
                total_risk = 0
                for _, risk_point in high_risk.iterrows():
                    dist = geodesic(
                        (waypoint_lat, waypoint_lon),
                        (risk_point['lat'], risk_point['lon'])
                    ).kilometers
                    if dist < 0.5:
                        total_risk += risk_point['y_proba'] * (1 - dist/0.5)
                
                if total_risk < min_total_risk:
                    min_total_risk = total_risk
                    best_waypoint = (waypoint_lat, waypoint_lon)
        
        if best_waypoint:
            # Crear ruta con waypoint
            route1, dist1, dur1 = get_route_osrm_safe(
                start_lat, start_lon,
                best_waypoint[0], best_waypoint[1]
            )
            route2, dist2, dur2 = get_route_osrm_safe(
                best_waypoint[0], best_waypoint[1],
                end_lat, end_lon
            )
            
            if route1 and route2:
                combined_route = route1[:-1] + route2
                total_distance = dist1 + dist2
                total_duration = dur1 + dur2
                return combined_route, total_distance, total_duration
        
        return None, None, None
        
    except Exception as e:
        st.warning(f"Error creando waypoint: {str(e)}")
        return None, None, None

def calculate_route_risk_simple(route_coords, df_risk, buffer_km=0.5):
    """Calcula el riesgo de una ruta de forma simple"""
    if not route_coords or df_risk.empty:
        return 0.0
    
    try:
        total_risk = 0
        risk_points = 0
        
        # Procesar cada 5º punto de la ruta para optimizar
        for route_point in route_coords[::5]:
            for _, risk_point in df_risk.iterrows():
                distance = geodesic(route_point, (risk_point['lat'], risk_point['lon'])).kilometers
                if distance <= buffer_km:
                    weight = 1 - (distance / buffer_km)
                    total_risk += risk_point['y_proba'] * weight
                    risk_points += 1
        
        return total_risk / max(risk_points, 1) if risk_points > 0 else 0.0
        
    except Exception as e:
        st.error(f"Error calculando riesgo de ruta: {str(e)}")
        return 0.0

# NUEVA FUNCIÓN PARA GOOGLE MAPS
def create_google_maps_url(origen, destino):
    """Crea URL de Google Maps para la ruta"""
    if not origen or not destino:
        return None
    
    origin_str = f"{origen[0]},{origen[1]}"
    dest_str = f"{destino[0]},{destino[1]}"
    
    url = f"https://www.google.com/maps/dir/{origin_str}/{dest_str}/"
    return url

def generate_simple_predictions(model, base_df, fecha, hora):
    """Genera predicciones - VERSIÓN CON METEOROLOGÍA SEGURA"""
    if model is None:
        st.error("Modelo no disponible")
        return pd.DataFrame()
    
    try:
        # LÍMITE ESTRICTO
        max_points = 100
        if len(base_df) > max_points:
            df = base_df.sample(n=max_points, random_state=42).copy()
            st.info(f"Usando {max_points} puntos de {len(base_df)} total")
        else:
            df = base_df.copy()
        
        # OBTENER DATOS METEOROLÓGICOS DE FORMA SEGURA
        with st.spinner("🌤️ Obteniendo datos meteorológicos..."):
            weather_data = get_weather_data_safe(fecha, hora)
        
        # Mostrar información meteorológica MEJORADA
        st.markdown("### 🌤️ Condiciones Meteorológicas Actuales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Temperatura", f"{weather_data['temperature_2m (°C)']}°C")
            
        with col2:
            st.metric("🌧️ Precipitación", f"{weather_data['precipitation (mm)']} mm")
            
        with col3:
            st.metric("💨 Viento", f"{weather_data['wind_speed_10m (km/h)']} km/h")
            
        with col4:
            lluvia_texto = "Sí" if weather_data['lluvia'] else "No"
            lluvia_emoji = "☔" if weather_data['lluvia'] else "☀️"
            st.metric(f"{lluvia_emoji} Lluvia", lluvia_texto)
        
        # Alertas meteorológicas
        alertas = []
        if weather_data['lluvia']:
            alertas.append("🌧️ Lluvia detectada - Mayor riesgo de accidentes")
        if weather_data['viento_fuerte']:
            alertas.append("💨 Viento fuerte - Condiciones adversas")
        if weather_data['temp_extrema']:
            alertas.append("🌡️ Temperatura extrema - Precaución adicional")
        
        if alertas:
            st.warning("⚠️ **Alertas meteorológicas:**")
            for alerta in alertas:
                st.write(f"- {alerta}")
        else:
            st.success("✅ Condiciones meteorológicas favorables")
        
        # Características básicas
        df['hora'] = int(hora)
        df['dia'] = int(fecha.weekday() if hasattr(fecha, 'weekday') else 1)
        df['mes'] = int(fecha.month if hasattr(fecha, 'month') else 9)
        
        # Características derivadas
        df['es_fin_semana'] = int(df['dia'].iloc[0] >= 5)
        df['es_hora_rush'] = int(hora in [7, 8, 9, 17, 18, 19])
        df['es_noche'] = int(22 <= hora or hora <= 5)
        
        # APLICAR DATOS METEOROLÓGICOS REALES
        for key, value in weather_data.items():
            df[key] = value
        
        # Características meteorológicas derivadas
        df['rush_x_lluvia'] = df['es_hora_rush'] * df['lluvia']
        df['noche_x_weekend'] = df['es_noche'] * df['es_fin_semana']
        df['Tipo_dia_Fin de semana'] = df['es_fin_semana']
        df['Tipo_dia_Laboral'] = 1 - df['es_fin_semana']
        
        # Características cíclicas
        df['hora_sin'] = np.sin(2 * np.pi * hora / 24.0)
        df['hora_cos'] = np.cos(2 * np.pi * hora / 24.0)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12.0)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12.0)
        
        # Obtener características del modelo
        if hasattr(model, 'feature_name_'):
            required_features = model.feature_name_
        else:
            st.error("No se pueden obtener características del modelo")
            return pd.DataFrame()
        
        # Completar características faltantes con valores inteligentes
        missing_features = []
        for col in required_features:
            if col not in df.columns:
                missing_features.append(col)
                # Asignar valores según el tipo de característica
                if 'temperature' in col.lower():
                    df[col] = weather_data['temperature_2m (°C)']
                elif 'precipitation' in col.lower() or 'lluvia' in col.lower():
                    df[col] = weather_data['precipitation (mm)']
                elif 'wind' in col.lower() or 'viento' in col.lower():
                    df[col] = weather_data['wind_speed_10m (km/h)']
                elif 'humidity' in col.lower():
                    df[col] = weather_data['humidity']
                elif 'visibility' in col.lower():
                    df[col] = 10000.0
                else:
                    df[col] = 0.0
        
        if missing_features:
            st.info(f"Completadas {len(missing_features)} características faltantes con datos meteorológicos")
        
        # Generar predicciones
        df_features = df[required_features]
        y_proba = model.predict_proba(df_features)[:, 1]
        
        # MOSTRAR ESTADÍSTICAS
        st.write("**📊 Estadísticas de predicción:**")
        st.write(f"- Media de riesgo: {y_proba.mean():.3f} ({y_proba.mean()*100:.1f}%)")
        st.write(f"- Puntos con riesgo >10%: {np.sum(y_proba > 0.1)} de {len(y_proba)}")
        
        # CALIBRACIÓN AJUSTADA SEGÚN CONDICIONES METEOROLÓGICAS
        calibration_factor = 0.3  # Factor base
        
        # Ajustar calibración según el clima
        if weather_data['lluvia']:
            calibration_factor = 0.4  # Menos calibración con lluvia (más riesgo)
            st.info("🌧️ Calibración ajustada por lluvia")
        if weather_data['viento_fuerte']:
            calibration_factor = 0.35  # Menos calibración con viento fuerte
            st.info("💨 Calibración ajustada por viento fuerte")
        if weather_data['temp_extrema']:
            calibration_factor = 0.35  # Menos calibración con temperatura extrema
            st.info("🌡️ Calibración ajustada por temperatura extrema")
        
        # Aplicar calibración
        y_proba_calibrated = y_proba * calibration_factor
        
        # Crear resultado
        df_result = df[['lat', 'lon']].copy()
        df_result['y_proba'] = y_proba_calibrated
        df_result['Fecha'] = pd.to_datetime(fecha)
        df_result['hora'] = int(hora)
        
        # FILTRAR solo puntos con riesgo mínimo
        threshold = 0.005  # 0.5% mínimo para mayor recall
        df_result = df_result[df_result['y_proba'] > threshold]
        
        # LÍMITE FINAL ABSOLUTO
        if len(df_result) > 50:
            df_result = df_result.nlargest(50, 'y_proba')
        
        st.success(f"Predicciones con meteorología: {len(df_result)} puntos (calibración: {calibration_factor})")
        return df_result
        
    except Exception as e:
        st.error(f"Error en predicciones: {str(e)}")
        return pd.DataFrame()

# INICIO DE LA APLICACIÓN
st.title("🚗 Predictor de Accidentes Barcelona - Versión Completa")
st.markdown("**Versión estable con datos meteorológicos reales y rutas seguras**")

# Cargar datos UNA SOLA VEZ
if 'base_df' not in st.session_state:
    with st.spinner("Cargando datos..."):
        st.session_state.base_df = load_base_dataframe()

if 'model' not in st.session_state:
    with st.spinner("Cargando modelo..."):
        st.session_state.model = load_model_simple()

base_df = st.session_state.base_df
model = st.session_state.model

if base_df.empty or model is None:
    st.error("No se pudieron cargar los datos o el modelo")
    st.stop()

# CONTROLES SIMPLES EN LA SIDEBAR
with st.sidebar:
    st.header("🎛️ Controles de Predicción")
    
    fecha = st.date_input(
        "📅 Fecha:", 
        value=datetime.now().date()
    )
    
    hora = st.slider("🕐 Hora:", 0, 23, 18)
    st.write(f"**Hora:** {hora:02d}:00")
    
    risk_filter = st.selectbox(
        "🔍 Filtro de riesgo:",
        ["Solo alto riesgo (>15%)", "Solo medio (>8%)", "Todos (>3%)"],
        index=0
    )
    
    map_style = st.radio(
        "🗺️ Estilo de mapa:",
        ["Mapa de calor", "Puntos", "Ambos"]
    )

    choose_model = st.radio(
        "Modelo predictivo:",
        ["Emiliano (LightGBM Optimizado)"]
    )
    
    # SECCIÓN DE RUTAS SEGURAS
    st.markdown("---")
    st.header("🛣️ Rutas Seguras")
    st.markdown("*Encuentra rutas evitando zonas de riesgo*")
    
    # NUEVA SECCIÓN: RUTAS FAVORITAS
    st.subheader("⭐ Rutas Favoritas")
    
    # Selector de rutas favoritas
    if st.session_state.rutas_favoritas:
        ruta_seleccionada = st.selectbox(
            "Cargar ruta favorita:",
            [""] + list(st.session_state.rutas_favoritas.keys()),
            key="select_favorite"
        )
        
        if ruta_seleccionada and ruta_seleccionada != "":
            if st.button("📂 Cargar Ruta", key="load_favorite"):
                ruta_fav = st.session_state.rutas_favoritas[ruta_seleccionada]
                st.session_state.route_data.update({
                    'origen_coords': ruta_fav['origen'],
                    'destino_coords': ruta_fav['destino']
                })
                st.success(f"✅ Ruta '{ruta_seleccionada}' cargada")
                st.rerun()
    else:
        st.info("No hay rutas favoritas guardadas")
    
    st.markdown("---")
    
    # Método de selección
    route_method = st.radio(
        "Método de selección:",
        ["📝 Direcciones", "📍 Click en mapa"],
        key="route_method"
    )
    
    if route_method == "📝 Direcciones":
        origen_addr = st.text_input(
            "🏠 Dirección de origen:",
            placeholder="Ej: Plaza Cataluña, Barcelona",
            key="origen_addr"
        )
        destino_addr = st.text_input(
            "🎯 Dirección de destino:",
            placeholder="Ej: Sagrada Familia, Barcelona", 
            key="destino_addr"
        )
        
        # Botón para calcular ruta por direcciones
        if st.button("🚀 Calcular Ruta por Direcciones", key="calc_by_address"):
            if origen_addr and destino_addr:
                with st.spinner("Geocodificando direcciones..."):
                    origen_coords = geocode_address_safe(origen_addr)
                    destino_coords = geocode_address_safe(destino_addr)
                    
                    if origen_coords[0] and destino_coords[0]:
                        st.session_state.route_data.update({
                            'origen_coords': origen_coords,
                            'destino_coords': destino_coords,
                            'method': 'address'
                        })
                        st.success("✅ Direcciones geocodificadas")
                    else:
                        st.error("❌ No se pudieron geocodificar las direcciones")
            else:
                st.warning("⚠️ Ingresa origen y destino")
    
    else:
        st.info("👆 Haz click en el mapa para seleccionar origen y destino")
        if st.button("🗑️ Limpiar puntos del mapa", key="clear_map_points"):
            st.session_state.route_data.update({
                'origen_coords': None,
                'destino_coords': None
            })
            st.success("Puntos limpiados")
    
    # Configuración de ruta
    st.subheader("⚙️ Configuración")
    
    risk_threshold = st.slider(
        "Umbral de riesgo a evitar (%):",
        5, 20, 10, 1,
        help="La ruta segura evitará puntos con riesgo mayor a este valor"
    )
    
    buffer_riesgo = st.slider(
        "Buffer de riesgo (km):",
        0.1, 2.0, 0.5, 0.1,
        help="Distancia a considerar alrededor de puntos de riesgo"
    )
    
    # Mostrar estado actual
    if st.session_state.route_data.get('origen_coords') and st.session_state.route_data.get('destino_coords'):
        st.success("✅ Origen y destino configurados")
        
        # GUARDAR EN FAVORITOS
        st.markdown("---")
        nombre_ruta = st.text_input(
            "💾 Nombre para guardar ruta:",
            placeholder="Ej: Trabajo, Casa, Gimnasio",
            key="save_route_name"
        )
        
        if st.button("⭐ Guardar en Favoritos", key="save_favorite"):
            if nombre_ruta:
                st.session_state.rutas_favoritas[nombre_ruta] = {
                    'origen': st.session_state.route_data['origen_coords'],
                    'destino': st.session_state.route_data['destino_coords'],
                    'fecha_guardado': datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.success(f"✅ Ruta '{nombre_ruta}' guardada en favoritos")
            else:
                st.warning("⚠️ Ingresa un nombre para la ruta")
        
        st.markdown("---")
        
        if st.button("🔄 Calcular Rutas", key="calc_safe_route"):
            st.session_state.route_data['calculate'] = True
            st.session_state.route_data['risk_threshold'] = risk_threshold / 100.0
    else:
        st.info("Configura origen y destino para calcular ruta")

# GENERAR PREDICCIONES - Solo cuando se cambie algo importante
prediction_key = f"{fecha}_{hora}_{risk_filter}"
if 'last_prediction_key' not in st.session_state or st.session_state.last_prediction_key != prediction_key:
    with st.spinner("Generando predicciones..."):
        df_preds = generate_simple_predictions(model, base_df, fecha, hora)
        st.session_state.df_preds = df_preds
        st.session_state.last_prediction_key = prediction_key
else:
    df_preds = st.session_state.df_preds

# Aplicar filtros
df_filtrado = df_preds.copy()

if risk_filter == "Solo alto riesgo (>15%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] >= 0.15]
elif risk_filter == "Solo medio (>8%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] >= 0.08]
else:  # Todos
    df_filtrado = df_filtrado[df_filtrado['y_proba'] >= 0.03]

# CÁLCULO DE RUTAS SEGURAS - CORREGIDO PARA MOSTRAR DOS RUTAS DIFERENTES
if st.session_state.route_data.get('calculate') and st.session_state.route_data.get('origen_coords') and st.session_state.route_data.get('destino_coords'):
    with st.spinner("🚗 Calculando ruta más rápida..."):
        origen = st.session_state.route_data['origen_coords']
        destino = st.session_state.route_data['destino_coords']
        
        # Calcular ruta directa
        ruta_directa, dist_directa, dur_directa = get_route_osrm_safe(
            origen[0], origen[1], destino[0], destino[1]
        )
        
        if ruta_directa:
            # Calcular riesgo de la ruta directa
            riesgo_directo = calculate_route_risk_simple(ruta_directa, df_preds, buffer_riesgo)
            
            # Guardar ruta directa
            st.session_state.route_data.update({
                'ruta_directa': ruta_directa,
                'dist_directa': dist_directa,
                'dur_directa': dur_directa,
                'riesgo_directo': riesgo_directo
            })
            
            st.success(f"✅ Ruta rápida: {dist_directa:.1f} km, {dur_directa:.0f} min")
            
            # CALCULAR RUTA ALTERNATIVA SEGURA
            with st.spinner("🛡️ Calculando ruta segura evitando zonas de riesgo..."):
                threshold = st.session_state.route_data.get('risk_threshold', 0.1)
                
                # Obtener ruta alternativa que evite puntos de riesgo
                ruta_segura, dist_segura, dur_segura = get_alternative_safe_route(
                    origen[0], origen[1], destino[0], destino[1],
                    df_preds, threshold
                )
                
                if ruta_segura:
                    # Calcular riesgo de la ruta segura
                    riesgo_seguro = calculate_route_risk_simple(ruta_segura, df_preds, buffer_riesgo)
                    
                    # Verificar si las rutas son diferentes
                    rutas_diferentes = False
                    if len(ruta_directa) != len(ruta_segura):
                        rutas_diferentes = True
                    else:
                        # Comparar algunos puntos
                        for i in range(0, min(len(ruta_directa), len(ruta_segura)), max(1, len(ruta_directa)//10)):
                            if (abs(ruta_directa[i][0] - ruta_segura[i][0]) > 0.0001 or 
                                abs(ruta_directa[i][1] - ruta_segura[i][1]) > 0.0001):
                                rutas_diferentes = True
                                break
                    
                    # Guardar ruta segura
                    st.session_state.route_data.update({
                        'ruta_segura': ruta_segura,
                        'dist_segura': dist_segura,
                        'dur_segura': dur_segura,
                        'riesgo_seguro': riesgo_seguro,
                        'rutas_diferentes': rutas_diferentes
                    })
                    
                    if rutas_diferentes:
                        mejora = ((riesgo_directo - riesgo_seguro) / max(riesgo_directo, 0.001)) * 100
                        st.success(f"✅ Ruta segura: {dist_segura:.1f} km, {dur_segura:.0f} min (Reducción riesgo: {mejora:.1f}%)")
                    else:
                        st.info("ℹ️ La ruta más rápida ya es la más segura disponible")
                else:
                    # Si no se puede calcular ruta segura, usar la directa
                    st.session_state.route_data.update({
                        'ruta_segura': ruta_directa,
                        'dist_segura': dist_directa,
                        'dur_segura': dur_directa,
                        'riesgo_seguro': riesgo_directo,
                        'rutas_diferentes': False
                    })
                    st.warning("⚠️ No se encontró ruta alternativa más segura")
        else:
            st.error("❌ No se pudo calcular la ruta")
    
    # Resetear flag de cálculo
    st.session_state.route_data['calculate'] = False

# MÉTRICAS
if not df_filtrado.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📍 Puntos", f"{len(df_filtrado)}")
    with col2:
        st.metric("📊 Riesgo Promedio", f"{df_filtrado['y_proba'].mean():.1%}")
    with col3:
        st.metric("🚨 Alto Riesgo", f"{len(df_filtrado[df_filtrado['y_proba'] >= 0.15])}")
    with col4:
        st.metric("⚠️ Riesgo Máximo", f"{df_filtrado['y_proba'].max():.1%}")

# LAYOUT PRINCIPAL
st.markdown("---")
col_mapa, col_info = st.columns([2, 1])

# Mapa interactivo
with col_mapa:
    st.subheader("🗺️ Mapa de Riesgo e Interactivo")
    
    if not df_filtrado.empty:
        m = folium.Map(location=[41.3851, 2.1734], zoom_start=12)
        
        # Agregar puntos de riesgo
        if map_style in ["Mapa de calor", "Ambos"]:
            heat_data = df_filtrado[["lat", "lon", "y_proba"]].values.tolist()
            plugins.HeatMap(
                heat_data,
                radius=15,
                blur=10,
                min_opacity=0.4
            ).add_to(m)
        
        if map_style in ["Puntos", "Ambos"]:
            for _, row in df_filtrado.iterrows():
                color, level = get_risk_color(row['y_proba'])
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=6,
                    popup=f"Riesgo: {row['y_proba']:.1%}",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Agregar marcadores de origen y destino si existen
        if st.session_state.route_data.get('origen_coords'):
            origen = st.session_state.route_data['origen_coords']
            folium.Marker(
                location=origen,
                popup="🏠 Origen",
                icon=folium.Icon(color='blue', icon='home')
            ).add_to(m)
        
        if st.session_state.route_data.get('destino_coords'):
            destino = st.session_state.route_data['destino_coords']
            folium.Marker(
                location=destino,
                popup="🎯 Destino",
                icon=folium.Icon(color='red', icon='flag')
            ).add_to(m)
        
        # Agregar rutas si existen con colores distintivos
        if st.session_state.route_data.get('ruta_directa'):
            folium.PolyLine(
                locations=st.session_state.route_data['ruta_directa'],
                weight=6,
                color='#FF4444',  # Rojo para ruta rápida
                opacity=0.9,
                popup=f"🚀 Ruta Rápida<br>📏 {st.session_state.route_data.get('dist_directa', 0):.1f} km<br>⏱️ {st.session_state.route_data.get('dur_directa', 0):.0f} min<br>⚠️ Riesgo: {st.session_state.route_data.get('riesgo_directo', 0):.3f}"
            ).add_to(m)
        
        # Solo mostrar ruta segura si es diferente de la rápida
        if (st.session_state.route_data.get('ruta_segura') and 
            st.session_state.route_data.get('rutas_diferentes', False)):
            folium.PolyLine(
                locations=st.session_state.route_data['ruta_segura'],
                weight=6,
                color='#00AA44',  # Verde para ruta segura
                opacity=0.9,
                popup=f"🛡️ Ruta Segura<br>📏 {st.session_state.route_data.get('dist_segura', 0):.1f} km<br>⏱️ {st.session_state.route_data.get('dur_segura', 0):.0f} min<br>⚠️ Riesgo: {st.session_state.route_data.get('riesgo_seguro', 0):.3f}"
            ).add_to(m)
        
        # Agregar leyenda mejorada solo si hay rutas
        if st.session_state.route_data.get('ruta_directa'):
            if st.session_state.route_data.get('rutas_diferentes', False):
                # Leyenda para dos rutas
                leyenda_html = '''
                <div style="position: fixed; 
                           bottom: 20px; left: 20px; width: 220px; height: auto; 
                           background-color: white; border:2px solid grey; z-index:9999; 
                           font-size:12px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 10px 0; color: #333;">🗺️ Leyenda del Mapa</h4>
                <p style="margin: 5px 0;"><span style="color:#FF4444; font-weight:bold; font-size:14px;">━━━</span> Ruta Más Rápida</p>
                <p style="margin: 5px 0;"><span style="color:#00AA44; font-weight:bold; font-size:14px;">━━━</span> Ruta Más Segura</p>
                <p style="margin: 5px 0;"><span style="color:#dc3545;">●</span> Alto Riesgo (>15%)</p>
                <p style="margin: 5px 0;"><span style="color:#ffc107;">●</span> Medio Riesgo (8-15%)</p>
                <p style="margin: 5px 0;"><span style="color:#28a745;">●</span> Bajo Riesgo (<8%)</p>
                </div>
                '''
            else:
                # Leyenda para una sola ruta
                leyenda_html = '''
                <div style="position: fixed; 
                           bottom: 20px; left: 20px; width: 220px; height: auto; 
                           background-color: white; border:2px solid grey; z-index:9999; 
                           font-size:12px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h4 style="margin: 0 0 10px 0; color: #333;">🗺️ Leyenda del Mapa</h4>
                <p style="margin: 5px 0;"><span style="color:#FF4444; font-weight:bold; font-size:14px;">━━━</span> Ruta Calculada</p>
                <p style="margin: 5px 0;"><span style="color:#dc3545;">●</span> Alto Riesgo (>15%)</p>
                <p style="margin: 5px 0;"><span style="color:#ffc107;">●</span> Medio Riesgo (8-15%)</p>
                <p style="margin: 5px 0;"><span style="color:#28a745;">●</span> Bajo Riesgo (<8%)</p>
                <p style="margin: 3px 0; font-style: italic;">* Solo se encontró una ruta viable</p>
                </div>
                '''
            
            m.get_root().html.add_child(folium.Element(leyenda_html))
        
        # Capturar clicks del mapa
        map_data = st_folium(m, width=700, height=500, key=f"map_{prediction_key}", returned_objects=["last_clicked"])
        
        # Procesar clicks del mapa para selección de puntos
        if map_data['last_clicked'] is not None and route_method == "📍 Click en mapa":
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            # Determinar si es origen o destino
            if not st.session_state.route_data.get('origen_coords'):
                st.session_state.route_data['origen_coords'] = (clicked_lat, clicked_lon)
                st.success(f"✅ Origen seleccionado: {clicked_lat:.4f}, {clicked_lon:.4f}")
                st.rerun()
            elif not st.session_state.route_data.get('destino_coords'):
                st.session_state.route_data['destino_coords'] = (clicked_lat, clicked_lon)
                st.success(f"✅ Destino seleccionado: {clicked_lat:.4f}, {clicked_lon:.4f}")
                st.rerun()
            else:
                st.info("Origen y destino ya seleccionados. Usa 'Limpiar puntos' para resetear.")
        
    else:
        st.warning("No hay datos para mostrar con el filtro seleccionado")
        # Mapa básico para clicks
        m_empty = folium.Map(location=[41.3851, 2.1734], zoom_start=12)
        st_folium(m_empty, width=700, height=500, key="empty_map")

# Información
with col_info:
    st.subheader("📊 Información de Riesgo")
    
    if not df_filtrado.empty:
        st.info(f"""
        **Análisis para {fecha} - {hora:02d}:00**
        
        📍 **Puntos:** {len(df_filtrado)}
        📊 **Riesgo promedio:** {df_filtrado['y_proba'].mean():.1%}
        ⚠️ **Riesgo máximo:** {df_filtrado['y_proba'].max():.1%}
        """)
        
        # Top 5 zonas
        if len(df_filtrado) >= 5:
            st.subheader("🎯 Top 5 Zonas Peligrosas")
            top_risk = df_filtrado.nlargest(5, 'y_proba')
            for i, (_, row) in enumerate(top_risk.iterrows(), 1):
                st.write(f"**{i}.** {row['y_proba']:.1%} - ({row['lat']:.4f}, {row['lon']:.4f})")
    else:
        st.warning("No hay datos para mostrar")
    
    # INFORMACIÓN DE RUTAS MEJORADA CON BOTONES DE GOOGLE MAPS
    st.markdown("---")
    st.subheader("🛣️ Información de Rutas")
    
    if st.session_state.route_data.get('ruta_directa'):
        # Crear dos columnas para las rutas
        col_rapida, col_segura = st.columns(2)
        
        with col_rapida:
            st.markdown("### 🔴 Ruta Más Rápida")
            st.success(f"""
            📏 **Distancia:** {st.session_state.route_data['dist_directa']:.1f} km
            ⏱️ **Tiempo:** {st.session_state.route_data['dur_directa']:.0f} min
            ⚠️ **Índice de riesgo:** {st.session_state.route_data['riesgo_directo']:.3f}
            """)
            
            # Botón de Google Maps para ruta rápida
            google_url = create_google_maps_url(
                st.session_state.route_data.get('origen_coords'),
                st.session_state.route_data.get('destino_coords')
            )
            if google_url:
                st.markdown(f"[🗺️ Abrir en Google Maps]({google_url})")
        
        with col_segura:
            if st.session_state.route_data.get('rutas_diferentes'):
                st.markdown("### 🟢 Ruta Más Segura")
                st.info(f"""
                📏 **Distancia:** {st.session_state.route_data['dist_segura']:.1f} km
                ⏱️ **Tiempo:** {st.session_state.route_data['dur_segura']:.0f} min
                ⚠️ **Índice de riesgo:** {st.session_state.route_data['riesgo_seguro']:.3f}
                """)
                
                # Botón de Google Maps para ruta segura
                google_url_segura = create_google_maps_url(
                    st.session_state.route_data.get('origen_coords'),
                    st.session_state.route_data.get('destino_coords')
                )
                if google_url_segura:
                    st.markdown(f"[🗺️ Abrir en Google Maps]({google_url_segura})")
            else:
                st.markdown("### ℹ️ Ruta Única")
                st.info("La ruta más rápida ya es la más segura disponible")
        
        # Comparación de rutas
        if st.session_state.route_data.get('rutas_diferentes'):
            st.markdown("---")
            st.markdown("### 📊 Comparación")
            
            diff_dist = st.session_state.route_data['dist_segura'] - st.session_state.route_data['dist_directa']
            diff_dur = st.session_state.route_data['dur_segura'] - st.session_state.route_data['dur_directa']
            diff_risk = st.session_state.route_data['riesgo_directo'] - st.session_state.route_data['riesgo_seguro']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📏 Distancia extra", f"+{diff_dist:.1f} km" if diff_dist > 0 else "Sin cambio")
            
            with col2:
                st.metric("⏱️ Tiempo extra", f"+{diff_dur:.0f} min" if diff_dur > 0 else "Sin cambio")
            
            with col3:
                if diff_risk > 0:
                    pct_mejora = (diff_risk / max(st.session_state.route_data['riesgo_directo'], 0.001)) * 100
                    st.metric("🛡️ Reducción riesgo", f"{pct_mejora:.1f}%")
                else:
                    st.metric("🛡️ Riesgo", "Sin mejora")
    
    elif st.session_state.route_data.get('origen_coords') or st.session_state.route_data.get('destino_coords'):
        st.info("Configura origen y destino, luego presiona 'Calcular Rutas'")
    else:
        st.markdown("""
        **🚀 Cómo usar las rutas:**
        
        1. **Método direcciones:** Escribe direcciones y presiona calcular
        2. **Método click:** Haz click en el mapa para seleccionar origen y destino
        3. **Configurar:** Ajusta el buffer de riesgo
        4. **Calcular:** Presiona el botón para obtener rutas
        5. **Guardar:** Opcionalmente guarda la ruta en favoritos
        
        El sistema calculará una ruta directa y buscará alternativas más seguras.
        """)

# Agregar sección de información general sobre el clima actual
if st.session_state.get('df_preds') is not None and not st.session_state.df_preds.empty:
    st.markdown("---")
    st.subheader("🌤️ Resumen Meteorológico del Análisis")
    
    # Obtener datos meteorológicos usados en el análisis
    weather_key = f"weather_{fecha}_{hora}"
    if weather_key in st.session_state:
        weather = st.session_state[weather_key]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌡️ Temperatura actual", f"{weather['temperature_2m (°C)']}°C")
            st.metric("💨 Velocidad del viento", f"{weather['wind_speed_10m (km/h)']} km/h")
        
        with col2:
            st.metric("🌧️ Precipitación", f"{weather['precipitation (mm)']} mm")
            st.metric("💧 Humedad relativa", f"{weather['humidity']}%")
        
        # Interpretación del impacto en la seguridad vial
        impacto_clima = []
        if weather['lluvia']:
            impacto_clima.append("🌧️ Lluvia presente - Aumenta riesgo de accidentes por reducción de adherencia")
        if weather['viento_fuerte']:
            impacto_clima.append("💨 Viento fuerte - Puede afectar estabilidad de vehículos altos")
        if weather['temp_extrema']:
            if weather['temperature_2m (°C)'] < 5:
                impacto_clima.append("🧊 Temperatura baja - Posible formación de hielo")
            else:
                impacto_clima.append("🔥 Temperatura alta - Posible fatiga del conductor")
        
        if impacto_clima:
            st.warning("**⚠️ Factores climáticos que afectan la seguridad:**")
            for factor in impacto_clima:
                st.write(f"- {factor}")
        else:
            st.success("✅ Condiciones climáticas favorables para la conducción")

# FOOTER
st.markdown("---")
st.markdown("## 🚗 Barcelona Accident Risk Predictor - Versión Completa")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Nuevas Características:**
    - Datos meteorológicos en tiempo real
    - Cálculo de rutas seguras
    - Selección por direcciones o click
    - Análisis de riesgo de rutas
    - Sistema de rutas favoritas
    """)

with col2:
    st.markdown("""
    **🌤️ Datos Meteorológicos:**
    - Temperatura actual
    - Precipitación y lluvia
    - Velocidad del viento
    - Alertas de condiciones adversas
    """)

with col3:
    st.markdown("""
    **🛣️ Rutas Inteligentes:**
    - Evita zonas de alto riesgo
    - Comparación directa vs segura
    - Buffer de riesgo configurable
    - Mapa interactivo
    - Botones Google Maps
    """)

st.markdown("---")
st.markdown("*Desarrollado para mejorar la seguridad vial en Barcelona con datos meteorológicos reales y rutas alternativas seguras*")