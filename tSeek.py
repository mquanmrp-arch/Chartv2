import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

# Configuración de página
st.set_page_config(
    page_title="📈 Trading Pattern Predictor",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📈 Predictor de Tendencias - Modelo Chartista con IA/CNN")
st.markdown("---")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    modelo_tipo = st.selectbox(
        "Tipo de Modelo",
        ["Binario (Alcista/Bajista)", "Multi-clase (Patrones)"]
    )
    
    st.markdown("---")
    st.subheader("📤 Cargar Modelo")
    modelo_file = st.file_uploader(
        "Sube tu modelo .h5",
        type=['h5'],
        help="Modelo entrenado de TensorFlow/Keras"
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Entrena tu modelo en Google Colab y descarga el archivo .h5")

# Área principal
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Subir Imagen del Gráfico")
    uploaded_image = st.file_uploader(
        "Selecciona una imagen (150x150 px recomendado)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_image:
        img = Image.open(uploaded_image).convert('RGB')
        st.image(img, caption="Imagen cargada", use_container_width=True)

with col2:
    st.subheader("🎯 Resultados de Predicción")
    
    if uploaded_image and modelo_file:
        try:
            # Cargar modelo
            with st.spinner("Cargando modelo..."):
                modelo_bytes = modelo_file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                    tmp.write(modelo_bytes)
                    tmp_path = tmp.name
                model = tf.keras.models.load_model(tmp_path)
                os.unlink(tmp_path)

            # Preprocesar imagen
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predicción
            with st.spinner("Analizando..."):
                predictions = model.predict(img_array, verbose=0)
            
            # Resultados según tipo de modelo
            if modelo_tipo == "Binario (Alcista/Bajista)":
                prob_alcista = predictions[0][0]
                prob_bajista = 1 - prob_alcista
                
                if prob_alcista > prob_bajista:
                    tendencia = "📈 ALCISTA"
                    confianza = prob_alcista
                else:
                    tendencia = "📉 BAJISTA"
                    confianza = prob_bajista
                
                st.markdown(f"### {tendencia}")
                st.markdown(f"**Confianza:** {confianza:.1%}")
                st.progress(float(confianza))
                
                with st.expander("Ver probabilidades detalladas"):
                    st.metric("Probabilidad Alcista", f"{float(prob_alcista):.2%}")
                    st.metric("Probabilidad Bajista", f"{float(prob_bajista):.2%}")
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(['Bajista', 'Alcista'], [prob_bajista, prob_alcista], color=['red', 'green'])
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilidad')
                ax.set_title('Distribución de Probabilidades')
                st.pyplot(fig)
                
            else:  # Multi-clase
                patrones = [
                    "Bandera Alcista", "Bandera Bajista",
                    "Canal Alcista", "Canal Bajista",
                    "Cuña Alcista", "Cuña Bajista",
                    "Hombro-Cabeza-Hombro", "Doble Techo", "Doble Piso",
                    "Rectángulo Alcista", "Rectángulo Bajista",
                    "Triángulo Alcista", "Triángulo Bajista",
                    "Triángulo Simétrico Alc", "Triángulo Simétrico Baj"
                ]

                patron_imagenes = {
                    "Bandera Alcista": "Patron/BAlc.png",
                    "Bandera Bajista": "Patron/BBaj.png",
                    "Canal Alcista": "Patron/CNA.png",
                    "Canal Bajista": "Patron/CND.png",
                    "Cuña Alcista": "Patron/CAlc.png",
                    "Cuña Bajista": "Patron/CBaj.png",
                    "Doble Piso": "Patron/DP.png",
                    "Doble Techo": "Patron/DT.png",
                    "Hombro-Cabeza-Hombro": "Patron/HCHB.png",
                    "Rectángulo Alcista": "Patron/RAlc.png",
                    "Rectángulo Bajista": "Patron/RBaj.png",                    
                    "Triángulo Alcista": "Patron/TAlc.png",
                    "Triángulo Bajista": "Patron/TBaj.png",
                    "Triángulo Simétrico Alc": "Patron/TSAlc.png",
                    "Triángulo Simétrico Baj": "Patron/TSBaj.png"
                }   

                num_clases = len(predictions[0])
                if len(patrones) != num_clases:
                    patrones = [f"Patrón {i+1}" for i in range(num_clases)]
                
                # Crear lista de nombres de archivos para el gráfico
                nombres_archivos = []
                for patron in patrones:
                    if patron in patron_imagenes:
                        # Extraer solo el nombre del archivo (sin "Patron/")
                        nombre_archivo = patron_imagenes[patron].split('/')[-1]
                        nombres_archivos.append(nombre_archivo)
                    else:
                        nombres_archivos.append(patron)

                indices_ordenados = np.argsort(predictions[0])[::-1]
                patron_predicho = patrones[indices_ordenados[0]]
                confianza_max = predictions[0][indices_ordenados[0]]
                archivo_patron_predicho = nombres_archivos[indices_ordenados[0]]
                
                st.markdown(f"### 🎯 Patrón Predicho: {archivo_patron_predicho}")
                st.markdown(f"**Confianza:** {confianza_max:.1%}")
                st.progress(float(confianza_max))
                
                st.markdown("---")
                
                # TRES GRÁFICAS PROPORCIONALES
                col_grafica1, col_grafica2, col_grafica3 = st.columns(3)
                
                with col_grafica1:
                    st.markdown("#### 📊 Activo Analizado")
                    st.image(img, caption="Gráfico del Activo", use_container_width=True)
                
                with col_grafica2:
                    st.markdown("#### 📈 Ranking de Probabilidades")
                    # Gráfico de barras con nombres de archivos
                    y_pos = np.arange(num_clases)
                    colors = ['green' if i == indices_ordenados[0] else 'skyblue' for i in range(num_clases)]
                    fig, ax = plt.subplots(figsize=(6, min(8, num_clases * 0.5)))
                    ax.barh(y_pos, predictions[0], color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(nombres_archivos, fontsize=8)
                    ax.set_xlabel('Probabilidad', fontsize=9)
                    ax.set_title('Ranking de Patrones', fontsize=10)
                    ax.set_xlim(0, 1)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col_grafica3:
                    st.markdown("#### 🎯 Patrón Detectado")
                    # Mostrar imagen del patrón predicho
                    if patron_predicho in patron_imagenes:
                        if os.path.exists(patron_imagenes[patron_predicho]):
                            patron_img = Image.open(patron_imagenes[patron_predicho])
                            st.image(patron_img, caption=f"{archivo_patron_predicho}", use_container_width=True)
                        else:
                            st.warning(f"⚠️ Imagen no encontrada")
                            st.info(f"Ruta: {patron_imagenes[patron_predicho]}")
                    else:
                        st.info("Sin imagen de referencia")
                
                # Ranking mejorado con imágenes (colapsable)
                st.markdown("---")
                with st.expander("🏆 Ver Ranking Completo de Patrones"):
                    st.markdown("### Top Patrones Detectados")
                    
                    for i, idx in enumerate(indices_ordenados):
                        patron_nombre = patrones[idx]
                        probabilidad = predictions[0][idx]
                        
                        col_rank1, col_rank2, col_rank3 = st.columns([0.5, 2, 1])
                        
                        with col_rank1:
                            # Medalla para top 3
                            if i == 0:
                                st.markdown("### 🥇")
                            elif i == 1:
                                st.markdown("### 🥈")
                            elif i == 2:
                                st.markdown("### 🥉")
                            else:
                                st.markdown(f"### {i+1}°")
                        
                        with col_rank2:
                            st.markdown(f"**{patron_nombre}**")
                            st.progress(float(probabilidad))
                            st.caption(f"Probabilidad: {probabilidad:.2%}")
                            
                            # Mostrar ruta del archivo
                            if patron_nombre in patron_imagenes:
                                st.caption(f"📁 `{patron_imagenes[patron_nombre]}`")
                        
                        with col_rank3:
                            # Mostrar miniatura de la imagen del patrón
                            if patron_nombre in patron_imagenes:
                                if os.path.exists(patron_imagenes[patron_nombre]):
                                    st.image(patron_imagenes[patron_nombre], width=100)
                                else:
                                    st.caption("🖼️ N/A")
                        
                        st.markdown("---")
            
            st.success("✅ Análisis completado exitosamente")
            
        except Exception as e:
            st.error(f"❌ Error al procesar: {str(e)}")
            st.info("Verifica que el modelo y la imagen sean compatibles")
    
    elif not modelo_file:
        st.warning("⚠️ Por favor, carga un modelo primero")
    elif not uploaded_image:
        st.info("📤 Sube una imagen para comenzar el análisis")

# Footer con instrucciones
st.markdown("---")
with st.expander("📚 ¿Cómo usar esta aplicación?"):
    st.markdown("""
    ### Pasos para usar el predictor:
    
    1. **Entrenar tu modelo en Google Colab:**
       - Usa el código de entrenamiento proporcionado
       - Descarga el archivo `.h5` generado
    
    2. **Cargar el modelo:**
       - En el sidebar, sube el archivo `.h5`
       - Selecciona el tipo de modelo (Binario o Multi-clase)
    
    3. **Subir gráfico:**
       - Carga una imagen del gráfico de velas (150x150 px recomendado)
       - La app redimensionará automáticamente si es necesario
    
    4. **Ver resultados:**
       - La predicción se mostrará automáticamente
       - Puedes ver probabilidades detalladas y gráficos
    
    ### 📊 Tipos de análisis:
    - **Binario:** Determina si la tendencia es alcista o bajista
    - **Multi-clase:** Identifica patrones chartistas específicos
    
    ### 🖼️ Imágenes de patrones:
    - Asegúrate de tener la carpeta `Patron/` con las imágenes correspondientes
    - Las imágenes deben seguir la nomenclatura indicada en el código
    """)

st.markdown("---")
st.caption("🔧 Desarrollado con TensorFlow + Streamlit | 📈 Trading con IA")