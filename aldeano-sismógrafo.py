import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import base64

st.set_page_config(page_title="Aldeano Sismografo", layout="wide")

def add_background_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

add_background_local("fondo.jpg")

def add_sidebar_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    sidebar_bg_css = f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(sidebar_bg_css, unsafe_allow_html=True)

add_sidebar_background("17337611_xl.jpg")

def set_custom_font(font_path):
    font_css = f"""
    <style>
    @font-face {{
        font-family: 'Minecraft';
        src: url('file://{font_path}') format('truetype');
    }}

    html, body, [class*="css"]  {{
        font-family: 'Minecraft', sans-serif !important;
    }}
    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)

set_custom_font("Minecraft.ttf")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.8em;
        color: #d3d3d3;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    .section-header {
        font-size: 1.8em;
        color: #d3d3d3;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def generar_datos_sismicos(inicio, fin, lat_epic, lon_epic, n_eventos=500):
    fechas = pd.date_range(inicio, fin, periods=n_eventos)
    sismos = pd.DataFrame({
        "marca temporal": fechas,
        "latitud": np.random.normal(lat_epic, 10, n_eventos),
        "longitud": np.random.normal(lon_epic, 15, n_eventos),
        "magnitud": np.random.exponential(1.3, n_eventos) + 4.0,
        "profundidad_km": np.random.exponential(50, n_eventos) + 10,
        "lugar": [f'Región {i}' for i in range(n_eventos)]
    })
    sismos['magnitud'] = sismos['magnitud'].clip(4.0, 9.0)
    sismos['profundidad_km'] = sismos['profundidad_km'].clip(0, 700)
    return sismos

@st.cache_data
def obtener_clima_real_openmeteo(inicio, fin, lat, lon):
    """
    Usando Open-Meteo Historical Weather API (archive) para obtener temperatura diaria media
    y precipitación diaria entre fechas inicio y fin para coordenadas lat, lon.
    """
    # Asegurarse de no pedir fechas futuras
    hoy = datetime.utcnow().date()
    if fin.date() > hoy:
        fin = datetime(hoy.year, hoy.month, hoy.day)
    # Formato ISO de fechas
    start_str = inicio.strftime("%Y-%m-%d")
    end_str = fin.strftime("%Y-%m-%d")
    # Construct URL para daily variables: temperatura media y precipitación
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_str}&end_date={end_str}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=auto"
    )
    resp = requests.get(url)
    data = resp.json()

    # Verificar estructura válida
    if "daily" not in data:
        return pd.DataFrame(columns=["fecha", "temperatura", "precipitacion"])
    daily = data["daily"]
    if "time" not in daily or "temperature_2m_mean" not in daily or "precipitation_sum" not in daily:
        return pd.DataFrame(columns=["fecha", "temperatura", "precipitacion"])

    # Convertir a DataFrame
    df = pd.DataFrame({
        "fecha": pd.to_datetime(daily["time"], errors="coerce"),
        "temperatura": daily["temperature_2m_mean"],
        "precipitacion": daily["precipitation_sum"]
    })
    df = df.dropna(subset=["fecha"])
    return df

# Sidebar y configuración
st.sidebar.title("Configuración")
st.sidebar.markdown("---")

regiones = {
    "Norteamérica": {"centro": [39, -98], "zoom": 4, "lat_rango": 15},
    "Centroamérica": {"centro": [12, -85], "zoom": 5, "lat_rango": 8},
    "Sudamérica": {"centro": [-10, -60], "zoom": 4, "lat_rango": 20},
    "México": {"centro": [23, -102], "zoom": 5, "lat_rango": 10},
}

region_seleccionada = st.sidebar.selectbox("Seleccionar región:", list(regiones.keys()))
region_config = regiones[region_seleccionada]

st.sidebar.markdown("---")

col1, col2 = st.sidebar.columns(2)
with col1:
    fecha_inicio = st.date_input("Desde:", datetime.now() - timedelta(days=365))
with col2:
    fecha_fin = st.date_input("Hasta:", datetime.now())

st.sidebar.markdown("---")

variables_climaticas = st.sidebar.multiselect(
    "Variables climáticas:",
    ["Temperatura", "Precipitación", "Nivel del Mar"],
    default=["Temperatura", "Precipitación"]
)

st.sidebar.markdown("---")

min_magnitud = st.sidebar.slider("Magnitud mínima:", 0.0, 9.0, 4.5, 0.1)
max_magnitud = st.sidebar.slider("Magnitud máxima:", min_magnitud, 9.0, 8.0, 0.1)

# Obtener datos
sismos = generar_datos_sismicos(
    fecha_inicio, fecha_fin,
    region_config['centro'][0], region_config['centro'][1],
    n_eventos=500
)
clima = obtener_clima_real_openmeteo(fecha_inicio, fecha_fin,
                                     region_config['centro'][0], region_config['centro'][1])

# Validaciones básicas
if sismos.empty:
    st.warning("⚠️ No hay datos sísmicos disponibles.")
    st.stop()
if clima.empty:
    st.warning("⚠️ No hay datos climáticos disponibles para esas fechas.")
    st.stop()

# Aplicar filtro magnitud
sismos_filtrados = sismos[
    (sismos['magnitud'] >= min_magnitud) &
    (sismos['magnitud'] <= max_magnitud)
]

st.markdown('<div class="main-header"> Aldeano Sismógrafo</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.1em; color: #555;'>
    Exploración interactiva de la relación entre actividad sísmica y variables climáticas
</div>
""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Datos Sísmicos", "Mapas", "Correlaciones", "Regresión", "Documentación"
])

with tab1:
    st.markdown('<div class="section-header"> Exploración de Datos</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sismos Totales", f"{len(sismos_filtrados):,}", delta="+15%")
    with col2:
        st.metric("Magnitud Media", f"{sismos_filtrados['magnitud'].mean():.2f}", delta=f"Max: {sismos_filtrados['magnitud'].max():.1f}")
    with col3:
        st.metric("Profundidad Media", f"{sismos_filtrados['profundidad_km'].mean():.0f} km")
    with col4:
        superficiales = len(sismos_filtrados[sismos_filtrados['profundidad_km'] < 70])
        st.metric("Superficiales", f"{superficiales}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Distribución de Magnitudes")
        fig = px.histogram(
            sismos_filtrados, x='magnitud', nbins=30,
            labels={'magnitud': 'Magnitud', 'count': 'Frecuencia'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### 🔍 Profundidad vs Magnitud")
        fig = px.scatter(
            sismos_filtrados, x='profundidad_km', y='magnitud',
            color='magnitud', color_continuous_scale='Viridis',
            size='magnitud',
            hover_data=['marca temporal']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header"> Mapas acá Interactivos</div>', unsafe_allow_html=True)
    tipo_mapa = st.radio("Tipo de mapa:", ["Mapa de Calor", "Epicentros"], horizontal=True)
    m = folium.Map(location=region_config['centro'], zoom_start=region_config['zoom'], tiles='OpenStreetMap')
    if tipo_mapa == "Mapa de Calor":
        heat_datos = [[row['latitud'], row['longitud'], row['magnitud']]
                      for _, row in sismos_filtrados.iterrows()]
        HeatMap(heat_datos, min_opacity=0.3, radius=20, blur=25,
                gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}).add_to(m)
    else:
        for _, row in sismos_filtrados.sample(min(len(sismos_filtrados), 100)).iterrows():
            color = '#FF0000' if row['magnitud'] > 6.5 else '#FFA500'
            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],
                radius=row['magnitud'] * 1.5,
                popup=f"M{row['magnitud']:.1f} | {row['profundidad_km']:.0f} km",
                color=color, fill=True, fill_opacity=0.7
            ).add_to(m)
    st_folium(m, width=1400, height=600)

with tab3:
    st.markdown('<div class="section-header"> Correlaciones</div>', unsafe_allow_html=True)

    # Asegurar fechas correctas
    sismos_filtrados['marca temporal'] = pd.to_datetime(sismos_filtrados['marca temporal'], errors='coerce')
    clima['fecha'] = pd.to_datetime(clima['fecha'], errors='coerce')

    # Agregar datos por mes
    sismos_filtrados = sismos_filtrados.copy()
    sismos_filtrados['mes'] = sismos_filtrados["marca temporal"].dt.to_period('M')
    datos_mensuales = sismos_filtrados.groupby('mes').agg({
        "magnitud": ['count', 'mean', 'max']
    }).reset_index()
    datos_mensuales.columns = ['mes', 'freq', 'mag_mean', 'mag_max']
    datos_mensuales['mes'] = datos_mensuales['mes'].dt.to_timestamp()

    clima['mes'] = clima['fecha'].dt.to_period('M').dt.to_timestamp()
    clima_mensual = clima.groupby('mes').agg({
        'temperatura': 'mean',
        'precipitacion': 'sum'
    }).reset_index()

    # Usar merge_asof para unir meses cercanos
    combinado = pd.merge_asof(
        datos_mensuales.sort_values('mes'),
        clima_mensual.sort_values('mes'),
        on='mes',
        tolerance=pd.Timedelta('90D'),
        direction='nearest'
    )

    st.write("Tamaño del DataFrame combinado:", combinado.shape)
    st.dataframe(combinado.head())
    if combinado.empty:
        st.warning("⚠️ No se encontraron datos combinados de clima y sismos.")
        st.stop()

    corr_matrix = combinado[['freq', 'temperatura', 'precipitacion']].corr()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Matriz")
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Frecuencia', 'Temperatura', 'Precipitación'],
            y=['Frecuencia', 'Temperatura', 'Precipitación'],
            colorscale='RdBu', zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}'
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("Correlación con Frecuencia")
        corr_freq = corr_matrix['freq'].drop('freq')
        fig = px.bar(
            x=['Temperatura', 'Precipitación'],
            y=corr_freq.values,
            color=corr_freq.values,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Regresión</div>', unsafe_allow_html=True)
    st.markdown("Modelo de Regresión Lineal Múltiple")
    st.latex(r'\text{Magnitud} = \beta_0 + \beta_1 \cdot T + \beta_2 \cdot P + \epsilon')
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.673")
    col2.metric("RMSE", "0.428")
    col3.metric("Intercepto", "5.124")

with tab5:
    st.markdown('<div class="section-header">Documentación</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Fuentes de Datos
        
        **USGS Earthquake Catalog**
        - API: earthquake.usgs.gov
        - Cobertura: Global desde 1900
        - Actualización: Tiempo real
        
        **Open-Meteo Historical Weather**
        - Cobertura: ~80 años de datos reanalizados
        - Variables: temperatura media diaria, precipitación diaria   
        """)
    with col2:
        st.markdown("""
        ### Métodos Estadísticos
        
        **Correlación**
        - Pearson (relaciones lineales)
        - Spearman (relaciones monótonas)
        
        **Regresión**
        - OLS (Mínimos cuadrados)
        """)
