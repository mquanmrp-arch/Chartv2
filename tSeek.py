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
                    "Triángulo Simetrico Alc", "Triángulo Simetrico Baj"
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

                indices_ordenados = np.argsort(predictions[0])[::-1]
                patron_predicho = patrones[indices_ordenados[0]]
                confianza_max = predictions[0][indices_ordenados[0]]
                
                st.markdown(f"### 🎯 {patron_predicho}")
                st.markdown(f"**Confianza:** {confianza_max:.1%}")
                st.progress(float(confianza_max))

                if patron_predicho in patron_imagenes:
                    st.image(patron_imagenes[patron_predicho], caption=f"Patrón: {patron_predicho}", width=250)
                
                with st.expander("Ver Top 3 Patrones"):
                    for i in range(min(3, len(indices_ordenados))):
                        idx = indices_ordenados[i]
                        st.metric(patrones[idx], f"{predictions[0][idx]:.2%}")
                
                y_pos = np.arange(num_clases)
                colors = ['green' if i == indices_ordenados[0] else 'skyblue' for i in range(num_clases)]
                fig, ax = plt.subplots(figsize=(10, min(8, num_clases * 0.6)))
                ax.barh(y_pos, predictions[0], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(patrones)
                ax.set_xlabel('Probabilidad')
                ax.set_title('Probabilidades por Patrón')
                ax.set_xlim(0, 1)
                st.pyplot(fig)
            
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
    """)

st.markdown("---")
st.caption("🔧 Desarrollado con TensorFlow + Streamlit | 📈 Trading con IA")