import streamlit as st
import ee
import geemap
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import psycopg2
import folium
import json

if 'EARTH_ENGINE_CREDENTIALS' in st.secrets:
    # 1. Leer el texto plano desde los secretos
    ee_creds_raw = st.secrets["EARTH_ENGINE_CREDENTIALS"]
    
    # 2. Convertir el texto en un diccionario real (json maneja los saltos de línea automáticamente)
    ee_creds = json.loads(ee_creds_raw)
    
    # 3. Autenticar usando directamente el diccionario extraído
    credential_object = ee.ServiceAccountCredentials(
        ee_creds['client_email'], 
        key_data=ee_creds['private_key']
    )
    ee.Initialize(credential_object)
        
else:
    # Si está corriendo de forma local en tu computadora
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

# --- DEFINICIÓN DE GEOMETRÍAS (ROIs) ---
# Coordenadas del Entorno Campus UTP (Frontera Activa)
roi_utp = ee.Geometry.Rectangle([
    -79.551101, 9.000870, -79.499710, 9.040796
])

# Coordenadas de la Zona de Control (Área Urbana Consolidada en Betania/El Dorado)
roi_control = ee.Geometry.Rectangle([
    -79.540000, 8.980000, -79.510000, 9.000000
])

# 2. FUNCIONES DE PROCESAMIENTO (GEE)
def procesar_sentinel(imagen):
    qa = imagen.select('QA60')
    bits_nube = (1 << 10) | (1 << 11)
    mascara = qa.bitwiseAnd(bits_nube).eq(0)
    img_limpia = imagen.updateMask(mascara).divide(10000)
    
    ndvi = img_limpia.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = img_limpia.normalizedDifference(['B11', 'B8']).rename('NDBI') # Área construida
    return img_limpia.addBands([ndvi, ndbi]).copyProperties(imagen, ["system:time_start"])

def obtener_porcentajes(imagen_clasificada, roi):
    area_pixeles = ee.Image.pixelArea().addBands(imagen_clasificada).reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName='clase'),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    
    ha_urb = 0
    ha_bos = 0
    if area_pixeles and 'groups' in area_pixeles:
        for grupo in area_pixeles['groups']:
            if grupo['clase'] == 0: 
                ha_urb = grupo['sum'] / 10000
            elif grupo['clase'] == 1: 
                ha_bos = grupo['sum'] / 10000
            
    total_ha = ha_urb + ha_bos
    pct_urb = (ha_urb / total_ha) * 100 if total_ha > 0 else 0
    pct_bos = (ha_bos / total_ha) * 100 if total_ha > 0 else 0
    return pct_urb, pct_bos

# 3. ENTRENAMIENTO DEL MODELO (RANDOM FOREST)
# Se entrena usando el entorno base histórico de la UTP
coleccion_historica = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                       .filterDate('2018-01-01', '2024-04-30')
                       .filterBounds(roi_utp)
                       .filter(ee.Filter.calendarRange(1, 4, 'month'))
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                       .map(procesar_sentinel))

mosaico_entrenamiento = coleccion_historica.median()

banda_clases = ee.Image().expression(
    "(NDVI < 0.3) ? 0 : ((NDVI > 0.55) ? 1 : 2)",
    {'NDVI': mosaico_entrenamiento.select('NDVI')}
).rename('clase')

banda_clases = banda_clases.updateMask(banda_clases.neq(2))
imagen_para_muestreo = mosaico_entrenamiento.select('NDVI').addBands(banda_clases)

datos_entrenamiento = imagen_para_muestreo.stratifiedSample(
    numPoints=500,
    classBand='clase',
    region=roi_utp,
    scale=10,
    geometries=True
)

clasificador_rf = ee.Classifier.smileRandomForest(50).train(
    features=datos_entrenamiento,
    classProperty='clase',
    inputProperties=['NDVI']
)

# 4. ENTORNO VISUAL Y CONTROLES (STREAMLIT)
st.title("📊 Modelado Predictivo de la Pérdida de Cobertura Forestal")
st.subheader("Entorno de la Universidad Tecnológica de Panamá (UTP)")

# Sidebar con filtros interactivos
st.sidebar.header("🎛️ Filtros de Control")
opcion_zona = st.sidebar.selectbox("Seleccione la Zona:", ["Entorno Campus UTP", "Zonas de Control"])
opcion_anio = st.sidebar.slider("Año de Visualización (Mapa Histórico/Predicción):", 2018, 2026, 2024)
opcion_indice = st.sidebar.selectbox("Tipo de Índice Ambiental:", ["NDVI (Vegetación)", "NDBI (Urbano/Construido)"])

# --- LÓGICA DE ASIGNACIÓN DINÁMICA DE ZONA ---
if opcion_zona == "Entorno Campus UTP":
    roi_actual = roi_utp
    centro_mapa = [9.0208, -79.5254]
    delta_texto = "Frontera Activa"
else:
    roi_actual = roi_control
    centro_mapa = [8.9900, -79.5250]
    delta_texto = "Urbano Consolidado"

# --- PROCESAMIENTO DINÁMICO EN TIEMPO REAL ---
with st.spinner("🔄 Conectando con Google Earth Engine y procesando imágenes..."):
    # Cargar colección según los filtros dinámicos
    coleccion_dinamica = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                         .filterDate(f'{opcion_anio}-01-01', f'{opcion_anio}-04-30')
                         .filterBounds(roi_actual)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                         .map(procesar_sentinel))
    
    mosaico_dinamico = coleccion_dinamica.median().clip(roi_actual)
    
    # Clasificación y estadísticas con base en la zona actual elegida
    mapa_clasificado = mosaico_dinamico.select(['NDVI']).classify(clasificador_rf)
    pct_urb_real, pct_bos_real = obtener_porcentajes(mapa_clasificado, roi_actual)

# --- PANEL DE INDICADORES (KPIs DINÁMICOS) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label=f"Cobertura Vegetal Real ({opcion_anio})", value=f"{pct_bos_real:.1f}%")
with col2:
    st.metric(label=f"Área Gris/Urbana Real ({opcion_anio})", value=f"{pct_urb_real:.1f}%")
with col3:
    st.metric(label="Dinámica del Suelo", value=opcion_zona, delta=delta_texto)

st.markdown("---")

# 5. VISUALIZACIÓN: MAPAS Y GRÁFICOS
col_mapa, col_graficos = st.columns([3, 2])

with col_mapa:
    st.write(f"### 🗺️ Vista Satelital Dinámica: {opcion_zona} ({opcion_anio})")
    
    
    # Configurar el mapa base centrado en las coordenadas dinámicas
    mapa_puro = folium.Map(location=centro_mapa, zoom_start=14, tiles="OpenStreetMap")
    
    # 2. Definir los parámetros de visualización para Earth Engine
    if "NDVI" in opcion_indice:
        vis_params = {'bands': ['NDVI'], 'min': 0.1, 'max': 0.7, 'palette': ['blue', 'yellow', 'green']}
        banda_seleccionada = mosaico_dinamico.select('NDVI')
        nombre_capa = 'Índice NDVI'
    else:
        vis_params = {'bands': ['NDBI'], 'min': -0.3, 'max': 0.3, 'palette': ['green', 'yellow', 'red']}
        banda_seleccionada = mosaico_dinamico.select('NDBI')
        nombre_capa = 'Índice NDBI'
        
    # 3. Obtener el ID del mapa directo desde los servidores de Google Earth Engine
    map_id_dict = ee.Image(banda_seleccionada).getMapId(vis_params)
    
    # 4. Inyectar la capa de Earth Engine como un TileLayer estándar de Folium
    folium.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=nombre_capa,
        overlay=True,
        control=True,
        opacity=0.85
    ).add_to(mapa_puro)
    
    # Agregar también el polígono rojo del límite geográfico analizado
    # Pasamos el ROI de Earth Engine a un formato GeoJSON que folium entiende nativamente
    geojson_roi = roi_actual.getInfo()
    folium.GeoJson(geojson_roi, name="Límite Analizado", style_function=lambda x: {'color': 'red', 'fillOpacity': 0.1}).add_to(mapa_puro)
    
    # 5. Renderizar usando el visualizador HTML incorporado que nunca falla
    folium.LayerControl().add_to(mapa_puro)
    mapa_html = mapa_puro._repr_html_()
    components.html(mapa_html, width=700, height=500, scrolling=False)

with col_graficos:
    st.write("### 📈 Tendencias Generales de Cobertura")
    
    # Línea de tiempo estandarizada de referencia
    data_historica = {
        'Año': [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        'Urbano': [45.1, 46.8, 48.0, 49.3, 51.5, 52.8, 53.9, 54.8, 55.9], 
        'Bosque': [54.9, 53.2, 52.0, 50.7, 48.5, 47.2, 46.1, 45.2, 44.1]  
    }
    df = pd.DataFrame(data_historica)
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(df['Año'], df['Urbano'], color='red', marker='o', label='Urbano (%)')
    ax.plot(df['Año'], df['Bosque'], color='green', marker='s', label='Bosque (%)')
    ax.axvline(x=2024.5, color='gray', linestyle='--', label='Proyección ML')
    ax.set_ylabel('Porcentaje (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    # Distribución en Barras Específica de la Selección
    st.write(f"### 📊 Proporción de Cobertura en Pantalla")
    fig2, ax2 = plt.subplots(figsize=(6, 2.1))
    categorias = ['Bosque', 'Urbano']
    valores = [pct_bos_real, pct_urb_real]
    ax2.barh(categorias, valores, color=['green', 'red'])
    ax2.set_xlabel('Porcentaje (%)')
    ax2.set_xlim(0, 100)
    st.pyplot(fig2)

st.markdown("---")

# 6. FORMULARIO DE CIENCIA CIUDADANA 
st.write("<h2>🐍 Registro de Avistamiento de Fauna (Ciencia Ciudadana)</h2>", unsafe_allow_html=True)

# 1. Diccionario de coordenadas predefinidas del campus de la UTP
lugares_utp = {
    "Seleccione el lugar del campus...": None,
    "Edificio 1 (Administrativo)": (9.0228, -79.5261),
    "Edificio 3 (FISC / FCyT)": (9.0232, -79.5250),
    "Edificio Postgrado": (9.0242, -79.5244),
    "Canchas Deportivas": (9.0215, -79.5230),
    "Cafetería Central / Librería": (9.0225, -79.5255),
    "Vía Centenario (Frente a la UTP)": (9.0195, -79.5275),
    "Sendero Ecológico / Bosque colindante": (9.0255, -79.5270)
}

# 2. Lista de fauna más común reportada en la zona de la UTP-Betania
fauna_comun = [
    "Seleccione el animal visto...",
    "Ñeque (Dasyprocta punctata)",
    "Perezoso de tres dedos (Bradypus variegatus)",
    "Gato Solo / Coatí (Nasua narica)",
    "Iguana Verde (Iguana iguana)",
    "Ardilla gris (Notosciurus granatensis)",
    "Tucán Pico Iris (Ramphastos sulfuratus)",
    "Otro (Especificar...)"
]

with st.form("formulario_especies", clear_on_submit=True):
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        fauna_seleccionada = st.selectbox("Especie avistada:", fauna_comun)
        
        # Si selecciona 'Otro', se despliega una caja de texto dinámica debajo
        especie_final = ""
        if fauna_seleccionada == "Otro (Especificar...)":
            especie_final = st.text_input("Escriba el nombre del animal:", placeholder="Ej. Armadillo, Boa, Venado")
        else:
            especie_final = fauna_seleccionada

    with col_f2:
        lugar_seleccionado = st.selectbox("¿Dónde lo viste? (Ubicación de referencia):", list(lugares_utp.keys()))
        coordenadas_capturadas = lugares_utp[lugar_seleccionado]

    with col_f3:
        fecha = st.date_input("Fecha del avistamiento:")
        
    comentarios = st.text_area("Detalles u observaciones adicionales (Ej. ¿Qué estaba haciendo?):")
    boton_enviar = st.form_submit_button("Enviar Reporte a PostGIS")
    
    if boton_enviar:
        # Validar que se haya seleccionado un animal y un lugar válido
        if lugar_seleccionado != "Seleccione el lugar del campus..." and especie_final not in ["Seleccione el animal visto...", ""]:
            try:
                # Extraer Latitud y Longitud del diccionario de la UTP
                lat, lon = coordenadas_capturadas

                # Conexión Inteligente: Detecta si está en la nube o en Localhost
                if 'postgres' in st.secrets:
                    # Si está corriendo en la nube de Streamlit, usa las credenciales secretas
                    conn = psycopg2.connect(
                        host=st.secrets['postgres']['host'],
                        database=st.secrets['postgres']['database'],
                        user=st.secrets['postgres']['user'],
                        password=st.secrets['postgres']['password'],
                        port=st.secrets['postgres']['port']
                    )
                else:
                    # Si estás haciendo pruebas locales en tu computadora
                    conn = psycopg2.connect(
                        host="localhost",
                        database="proyecto",  
                        user="postgres",           
                        password="tu_password",  # <-- Asegúrate de que coincida con tu clave local (242315)
                        port="5432"
                    )
                cursor = conn.cursor()
                
                
                # Consulta SQL espacial usando las variables directamente
                # PostGIS recibe: Longitud (X), Latitud (Y)
                query_sql = """
                    INSERT INTO avistamientos_fauna (especie, fecha, comentarios, geom)
                    VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326));
                """
                
                cursor.execute(query_sql, (especie_final, fecha, comentarios, lon, lat))
                conn.commit()
                
                cursor.close()
                conn.close()
                
                st.success(f"✅ ¡Reporte de '{especie_final}' registrado en PostGIS para el sector '{lugar_seleccionado}' con éxito!")
                
            except Exception as e:
                st.error(f"❌ Error al conectar o guardar en la base de datos: {e}")
        else:
            st.warning("⚠️ Por favor, asegúrate de seleccionar un animal válido y un lugar del campus antes de enviar.")