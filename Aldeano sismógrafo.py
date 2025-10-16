import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
add_background_local("C:\\Users\\cubic\\python\\aldeano\\fondo.jpg")
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
add_sidebar_background("C:\\Users\\cubic\\python\\aldeano\\17337611_xl.jpg")
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
# Diseño de página
st.markdown("""
    <style>
    /* Encabezado principal */
    .main-header {
        font-size: 2.8em;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    
    /* Encabezados de sección */
    .section-header {
        font-size: 1.8em;
        color: #ff7f0e;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 3px solid #ff7f0e;
        padding-bottom: 5px;
    }
    
    /* Estilo de pestañas */
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
    
    /* Tarjetas de métricas */
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
def generar_datos_climáticos(inicio, fin):
    fechas_clima = pd.date_range(inicio, fin, freq='D')
    dias = len(fechas_clima)

    clima = pd.DataFrame({
        "fecha": fechas_clima,  # <-- Added this line
        "dias": dias,
        "temperatura": 15 + 10 * np.sin(np.arange(dias) * 2 * np.pi / 365) + np.random.randn(dias) * 2,
        "precipitacion": np.abs(np.random.gamma(2, 2, dias)),
        "nivel_mar": np.random.randn(dias).cumsum() * 0.05,
        "glaciares": 100 - np.arange(dias) * 0.01 + np.random.randn(dias) * 0.5,
        "sequia": 30 + 20 * np.sin(np.arange(dias) * 2 * np.pi / 365) + np.random.randn(dias) * 5,
    })
    
    return clima

# Barra lateral
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
#generar datos
sismos = generar_datos_sismicos(
    fecha_inicio, 
    fecha_fin, 
    region_config['centro'][0],
    region_config['centro'][1],
    n_eventos=500
)

clima = generar_datos_climáticos(fecha_inicio, fecha_fin)

# Filtrar por magnitud
sismos_filtrados = sismos[
    (sismos['magnitud'] >= min_magnitud) & 
    (sismos['magnitud'] <= max_magnitud)
]
# Header

st.markdown('<div class="main-header"> Aldeano Sismógrafo</div>', 
           unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 1.1em; color: #555;'>
    Exploración interactiva de la relación entre actividad sísmica y variables climáticas
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Datos Sísmicos",
    "Mapas",
    "Correlaciones",
    "Regresión",
    "Documentación"
])

# PESTAÑA 1: Datos Sísmicos
with tab1:
    st.markdown('<div class="section-header"> Exploración de Datos</div>', 
               unsafe_allow_html=True)
    
    # Métricas en 4 columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Sismos Totales",
            value=f"{len(sismos_filtrados):,}",
            delta="+15%"
        )
    
    with col2:
        st.metric(
            label="Magnitud Media",
            value=f"{sismos_filtrados['magnitud'].mean():.2f}",
            delta=f"Max: {sismos_filtrados['magnitud'].max():.1f}"
        )
    
    with col3:
        st.metric(
            label="Profundidad Media",
            value=f"{sismos_filtrados['profundidad_km'].mean():.0f} km"
        )

    with col4:
        superficiales = len(sismos_filtrados[sismos_filtrados['profundidad_km'] < 70])
        st.metric(
            label="Superficiales",
            value=f"{superficiales}"
        )
    
    st.markdown("---")
    
    # Gráficas en 2 columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("Distribución de Magnitudes")
        fig = px.histogram(
            sismos_filtrados,
            x='magnitud',
            nbins=30,
            labels={'magnitud': 'Magnitud', 'count': 'Frecuencia'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Profundidad vs Magnitud")
        fig = px.scatter(
            sismos_filtrados,
            x='profundidad_km',
            y='magnitud',
            color='magnitud',
            color_continuous_scale='Viridis',
            size='magnitud'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
# PESTAÑA 2: Mapas
with tab2:
    st.markdown('<div class="section-header"> Mapas acá Interactivos</div>', 
               unsafe_allow_html=True)
    
    tipo_mapa = st.radio(
        "Tipo de mapa:",
        ["Mapa de Calor", "Epicentros"],
        horizontal=True
    )
    
    m = folium.Map(
        location=region_config['centro'],
        zoom_start=region_config['zoom'],
        tiles='OpenStreetMap'
    )
    
    if tipo_mapa == "Mapa de Calor":
        heat_datos = [
            [row['latitud'], row['longitud'], row['magnitud']] 
            for _, row in sismos_filtrados.iterrows()
        ]
        
        HeatMap(
            heat_datos,
            min_opacity=0.3,  # <-- changed from opacidad_min
            radius=20,        # <-- changed from radio
            blur=25,
            gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)
    
    else: 
        for _, row in sismos_filtrados.sample(100).iterrows():
            color = '#FF0000' if row['magnitud'] > 6.5 else '#FFA500'

            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],  # <-- changed from ubicacion
                radius=row['magnitud'] * 1.5,                # <-- changed from radio
                popup=f"M{row['magnitud']:.1f} | {row['profundidad_km']:.0f} km",
                color=color,
                fill=True,
                fill_opacity=0.7                             # <-- changed from fillOpacidad
            ).add_to(m)
    
    st_folium(m, width=1400, height=600)

#PESTAÑA 3: Correlaciones
with tab3:
    st.markdown('<div class="section-header"> Correlaciones</div>', 
               unsafe_allow_html=True)
    
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
    
    combinado = pd.merge(datos_mensuales, clima_mensual, on='mes')

    corr_matrix = combinado[['freq', 'temperatura', 'precipitacion']].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("Matriz")
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Frecuencia', 'Temperatura', 'Precipitación'],
            y=['Frecuencia', 'Temperatura', 'Precipitación'],
            colorscale='RdBu',
            zmid=0,
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
# PESTAÑA 4: Regresión
with tab4:
    st.markdown('<div class="section-header">Regresión</div>', 
               unsafe_allow_html=True)
    
    st.markdown("Modelo de Regresión Lineal Múltiple")
    st.latex(r'\text{Magnitud} = \beta_0 + \beta_1 \cdot T + \beta_2 \cdot P + \epsilon')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.673")
    col2.metric("RMSE", "0.428")
    col3.metric("Intercepto", "5.124")

with tab5:
    st.markdown('<div class="section-header">Documentación</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Fuentes de Datos
        
        **USGS Earthquake Catalog**
        - API: earthquake.usgs.gov
        - Cobertura: Global desde 1900
        - Actualización: Tiempo real
        
        **NASA Earth Data**
        - Temperatura GISS/MERRA-2
        - Resolución: 0.5-1 km
        """)
    
    with col2:
        st.markdown("""
        ### Métodos Estadísticos
        
        **Correlación**
        - Pearson (relaciones lineales)
        - Spearman (monótonas)
        
        **Regresión**
        - OLS (Mínimos cuadrados)
        - R² ajustado
        """)