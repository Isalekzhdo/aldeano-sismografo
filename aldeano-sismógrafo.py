import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
import folium
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(page_title="Aldeano Sismógrafo", layout="wide")

# ==========================
# Funciones auxiliares
# ==========================

@st.cache_data
def obtener_clima_real_openmeteo(inicio, fin, lat, lon):
    """
    Obtiene datos climáticos reales (temperatura y precipitación)
    desde la API Open-Meteo.
    """
    # Convertir fechas si son tipo date
    if isinstance(inicio, date):
        inicio = datetime.combine(inicio, datetime.min.time())
    if isinstance(fin, date):
        fin = datetime.combine(fin, datetime.min.time())

    hoy = datetime.now()
    if fin > hoy:
        fin = hoy

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": inicio.strftime("%Y-%m-%d"),
        "end_date": fin.strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto"
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        st.warning("Error al obtener datos de Open-Meteo.")
        return pd.DataFrame()

    data = resp.json()
    if "daily" not in data:
        st.warning("Sin datos climáticos disponibles para ese rango.")
        return pd.DataFrame()

    clima = pd.DataFrame({
        "fecha": data["daily"]["time"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "temp_min": data["daily"]["temperature_2m_min"],
        "precipitacion": data["daily"]["precipitation_sum"]
    })

    clima["fecha"] = pd.to_datetime(clima["fecha"])
    return clima


@st.cache_data
def obtener_sismos_reales(inicio, fin, min_magnitud=4.0, max_magnitud=9.0, region="Mexico"):
    """
    Obtiene datos sísmicos reales desde la API del USGS.
    """
    if isinstance(inicio, date):
        inicio = datetime.combine(inicio, datetime.min.time())
    if isinstance(fin, date):
        fin = datetime.combine(fin, datetime.min.time())

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": inicio.strftime("%Y-%m-%d"),
        "endtime": fin.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitud,
        "maxmagnitude": max_magnitud,
        "limit": 1000
    }

    resp = requests.get(url, params=params)
    data = resp.json()

    if "features" not in data:
        return pd.DataFrame(columns=["marca temporal", "latitud", "longitud", "magnitud", "profundidad_km", "lugar"])

    registros = []
    for sismo in data["features"]:
        props = sismo["properties"]
        coords = sismo["geometry"]["coordinates"]
        registros.append({
            "marca temporal": pd.to_datetime(props["time"], unit="ms"),
            "latitud": coords[1],
            "longitud": coords[0],
            "profundidad_km": coords[2],
            "magnitud": props["mag"],
            "lugar": props["place"]
        })

    df = pd.DataFrame(registros)
    if region.lower() == "mexico":
        df = df[df["lugar"].str.contains("Mexico", case=False, na=False)]

    return df


def mostrar_mapa_sismos(sismos):
    """
    Crea un mapa interactivo con los sismos.
    """
    if sismos.empty:
        st.info("No hay sismos disponibles para mostrar.")
        return None

    mapa = folium.Map(location=[sismos["latitud"].mean(), sismos["longitud"].mean()], zoom_start=4)

    for _, row in sismos.iterrows():
        popup_text = f"<b>{row['lugar']}</b><br>Magnitud: {row['magnitud']:.1f}<br>Profundidad: {row['profundidad_km']:.1f} km"
        color = "red" if row["magnitud"] >= 6 else "orange" if row["magnitud"] >= 5 else "yellow"
        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=row["magnitud"] * 1.2,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=popup_text
        ).add_to(mapa)

    return mapa

# ==========================
# Interfaz principal
# ==========================

st.title("🌋 Aldeano Sismógrafo – Datos Reales")
st.markdown("Visualizador de **sismos reales y clima histórico** basado en APIs públicas (USGS + Open-Meteo).")

col1, col2, col3 = st.columns(3)

with col1:
    region_seleccionada = st.selectbox("🌎 Región", ["Mexico", "Mundo"])
with col2:
    fecha_inicio = st.date_input("📅 Fecha inicial", datetime.now() - timedelta(days=7))
with col3:
    fecha_fin = st.date_input("📅 Fecha final", datetime.now())

min_magnitud = st.slider("Magnitud mínima", 3.0, 8.0, 4.0)
max_magnitud = st.slider("Magnitud máxima", 4.0, 9.5, 8.0)

# Configuración de coordenadas base por región
region_config = {
    "Mexico": {"centro": [23.6345, -102.5528]},
    "Mundo": {"centro": [0, 0]}
}

# Obtener datos reales
st.subheader("📊 Datos reales obtenidos:")

sismos = obtener_sismos_reales(fecha_inicio, fecha_fin, min_magnitud, max_magnitud, region_seleccionada)
clima = obtener_clima_real_openmeteo(fecha_inicio, fecha_fin,
                                     region_config[region_seleccionada]["centro"][0],
                                     region_config[region_seleccionada]["centro"][1])

colA, colB = st.columns(2)
with colA:
    st.metric("Sismos detectados", len(sismos))
with colB:
    if not clima.empty:
        st.metric("Temperatura promedio (°C)", round((clima["temp_max"].mean() + clima["temp_min"].mean()) / 2, 1))
    else:
        st.metric("Temperatura promedio (°C)", "N/D")

# Mapa de sismos
st.subheader("🗺️ Mapa de sismos")
mapa = mostrar_mapa_sismos(sismos)
if mapa:
    st_folium(mapa, width=900, height=500)

# Gráfica de magnitudes
if not sismos.empty:
    st.subheader("📈 Magnitud de los sismos")
    fig = px.histogram(sismos, x="magnitud", nbins=20, title="Distribución de magnitudes sísmicas", color_discrete_sequence=["red"])
    st.plotly_chart(fig, use_container_width=True)

# Gráfica de clima
if not clima.empty:
    st.subheader("🌤️ Clima histórico")
    fig2 = px.line(clima, x="fecha", y=["temp_max", "temp_min", "precipitacion"],
                   title="Temperaturas y precipitación diarias", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Datos de sismos: USGS Earthquake API | Datos climáticos: Open-Meteo Archive")
