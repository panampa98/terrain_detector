import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import json
import firebase_admin
from firebase_admin import credentials, firestore
import cloudinary
import cloudinary.uploader
import io
import uuid
from datetime import datetime
import hashlib
import time
from datetime import datetime
import random
import string

import google.generativeai as genai

# Cargar API KEY
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", None))

class TerrainDetector:
    def __init__(self):
        st.set_page_config(
                initial_sidebar_state='collapsed',
                layout='wide',
                page_icon='🚜',
                page_title='Terrain detector',
            )
        
        trained_model = self.load_nn_model()

        # models = genai.list_models()

        # for model in models:
        #     print("📦 Nombre:", model.name)
        #     print("🧠 Soporta generate_content:", "generateContent" in model.supported_generation_methods)
        #     print("---")

        # Crear modelo
        genai_model = genai.GenerativeModel(model_name='models/gemini-2.5-flash')

        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = False
        if 'real_terrain' not in st.session_state:
            st.session_state.real_terrain = None
        
        self.db = self.connect_firebase()
        self.bucket = self.connect_cloudinary()

        self.run(trained_model, genai_model)

    def connect_firebase(self):
        cred_dict = json.loads(st.secrets['firebase_service_account'])
        cred = credentials.Certificate(cred_dict)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'storageBucket': cred_dict['project_id'] + '.appspot.com'
            })

        db = firestore.client()
        
        return db

    def connect_cloudinary(self):
        cloudinary.config(
            cloud_name=st.secrets['cloudinary']['cloud_name'],
            api_key=st.secrets['cloudinary']['api_key'],
            api_secret=st.secrets['cloudinary']['api_secret']
        )
    
    def upload_to_cloudinary(self, image: Image.Image, filename: str):
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        buffered.seek(0)

        response = cloudinary.uploader.upload(
            buffered,
            public_id=filename,
            folder='terrain_detector'
            )
        return response['secure_url']

    # Cargar el modelo
    @staticmethod
    @st.cache_resource
    def load_nn_model():
        model = load_model('trained_model.h5', compile=False)
        return model
    
    def run(self, trained_model, genai_model):
        # Diccionario de terrenos
        terr_dict = {0: 'Carretera', 1: 'Tierra Seca', 2: 'Tierra Lodosa', 3: 'Pedregoso'}

        # Prompt de clasificación
        prompt = (
            'Clasifica la siguiente imagen de terreno en una de estas categorías:\n'
            '0: Carretera\n'
            '1: Tierra Seca\n'
            '2: Tierra Lodosa\n'
            '3: Pedregoso\n'
            'Responde solo el número correspondiente.'
        )

        # Función para determinar velocidad recomendada
        def set_speed(terrain):
            return {
                'Carretera': '70 KPH',
                'Pedregoso': '30 KPH',
                'Tierra Seca': '50 KPH',
                'Tierra Lodosa': '20 KPH'
            }.get(terrain, 'N/A')
        
        def generate_custom_id(real_terrain: str):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            terrain = real_terrain.replace(" ", "") if real_terrain else "Desconocido"
            rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            return f"{timestamp}_{terrain}_{rand}"


        cols = st.columns([3, 5])

        with cols[0]:
            st.title('Clasificador de Terreno')
        
        with cols[1]:
            uploaded_file = st.file_uploader('Sube una imagen de terreno', type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            current_image_hash = self.get_file_hash(uploaded_file)

            # Si la imagen cambia, reseteamos el estado
            if st.session_state.get('last_image_hash') != current_image_hash:
                st.session_state.last_image_hash = current_image_hash
                st.session_state.uploaded_data = False
                st.session_state.real_terrain = None
                st.session_state.genai_prediction = None  # limpiar predicción anterior
            
            # Mostrar imagen redimensionada
            image = Image.open(uploaded_file)

            with cols[1]:
                st.image(image, caption='Imagen cargada', use_container_width=True)
            
            # Preprocesamiento de imagen
            img_size_display = (750, 500)
            img_train_size = (300, 450)
            image = image.resize(img_size_display)

            # Botón para forzar nueva predicción
            force_predict = False
            # if not not st.session_state.uploaded_data:
            #     force_predict = st.button("Volver a predecir con Gemini")

            # =========== CLASIFICACION CON GEN AI ===========
            # Solo predecir si no hay una ya guardada o si se fuerza
            if st.session_state.get('genai_prediction') is None or force_predict:
                start = time.perf_counter()
                response = genai_model.generate_content([prompt, image])
                end = time.perf_counter()
                genai_time = round(end - start, 3)

                pred = st.session_state.genai_prediction
                lbl =  response.text.strip()

                try:
                    pred_idx = int(lbl)
                    predicted_label = terr_dict[pred_idx]
                except (ValueError, KeyError):
                    predicted_label = 'No identificado'

                # Guardar en session_state
                st.session_state.genai_prediction = {
                    'label': lbl,
                    'time': genai_time,
                    'predict': predicted_label,
                    'tokens_prompt': response._result.usage_metadata.prompt_token_count,
                    'tokens_output': response._result.usage_metadata.candidates_token_count,
                    'tokens_total': response._result.usage_metadata.total_token_count,
                }

            # Mostrar resultados guardados
            pred = st.session_state.genai_prediction
            geanai_pred = pred['predict']
            genai_time = pred['time']

            cols[0].success(f"Predicción GenAI: {geanai_pred} (en {genai_time}s)")
            with cols[0].expander("Detalles del uso de tokens"):
                st.text(f"Prompt: {pred['tokens_prompt']} tokens")
                st.text(f"Respuesta: {pred['tokens_output']} tokens")
                st.text(f"Total: {pred['tokens_total']} tokens")

            # =========== CLASIFICACION CON DEEP LEARNING ===========
            start = time.perf_counter()
            # Preprocesamiento de imagen
            img_array = np.array(image) / 255.0
            img_array = (img_array - np.min(img_array)) * 255 / (np.max(img_array) - np.min(img_array))
            img_array = img_array.astype(np.uint8)

            img_to_predict = Image.fromarray(img_array).resize(img_train_size).convert('L')
            img_to_predict = np.array(img_to_predict)
            img_to_predict = np.expand_dims(img_to_predict, axis=0)

            # Predicción
            prediction = trained_model.predict(img_to_predict)[0]
            pred_index = np.argmax(prediction)
            detected_terrain = terr_dict[pred_index]

            end = time.perf_counter()
            deep_time = round(end - start, 3)

            with cols[0]:
                cols_2 = st.columns([5, 5], vertical_alignment='bottom')
                with cols_2[0]:
                    self.show_card('Terreno detectado (Deep Learning)', detected_terrain)
                
                with cols_2[1]:
                    self.show_card('Terreno detectado (Generative AI)', geanai_pred)
                
                cols_3 = st.columns([5, 5], vertical_alignment='bottom')
                with cols_3[0]:
                    self.show_card('Tiempo de respuesta', f'{deep_time} segs.')
                
                with cols_3[1]:
                    self.show_card('Tiempo de respuesta', f'{genai_time} segs.')
                    # self.show_card('Velocidad recomendada', set_speed(detected_terrain))
                
                if not st.session_state.uploaded_data:
                    real_terrain = st.selectbox('**Selecciona el tipo de terreno real**',
                                                options=list(terr_dict.values()),
                                                index=None,
                                                placeholder='Escoge una opcion')

                    st.session_state.real_terrain = real_terrain

                    if real_terrain:
                        # Subir imagen y datos automáticamente
                        image_id = str(uuid.uuid4())
                        image_url = self.upload_to_cloudinary(image, f'{image_id}')

                        doc_data = {
                            'timestamp': datetime.now(),
                            'deep_l_terrain': {
                                'label': int(pred_index),
                                'time': deep_time,
                                'predict': detected_terrain,
                                },
                            'gen_ai_terrain': pred,
                            'real_terrain': real_terrain,
                            'imagen_url': image_url
                        }
                        self.db.collection('terrain_detector').add(doc_data, document_id=generate_custom_id(real_terrain))

                        st.session_state.uploaded_data = True
                        st.rerun()
                else:
                    self.show_card('Terreno real', st.session_state.real_terrain)
                # with cols_2[1]:
                #     self.show_card('Velocidad esperada', set_speed(st.session_state.real_terrain))
                
                if st.session_state.real_terrain:
                    st.success('🥳🥳 Gracias por tu colaboración.')
    
    def get_file_hash(self, file):
        return hashlib.md5(file.getvalue()).hexdigest()
    
    def show_card(self, lbl, value):
        card_html = f"""
        <div style="
            background-color: var(--background-secondary);
            padding: 15px;
            width: 100%;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 10px;
        ">
            <div style="
                font-size: 18px;
                font-weight: bold;
                color: #1668CC;
                margin-bottom: 8px;
            ">
                {lbl}
            </div>
            <div style="
                font-size: 24px;
                font-weight: bold;
                color: var(--text-color);
            ">
                {value}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

if __name__ == '__main__':
    TerrainDetector()