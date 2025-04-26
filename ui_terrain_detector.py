import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

class TerrainDetector:
    def __init__(self):
        st.set_page_config(
                initial_sidebar_state='collapsed',
                layout='wide',
                page_icon='🚜',
                page_title='Terrain detector',
            )
        
        trained_model = self.load_nn_model()

        self.run(trained_model)

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

    # Cargar el modelo
    @staticmethod
    @st.cache_resource
    def load_nn_model():
        model = load_model('trained_model.h5', compile=False)
        return model
    
    def run(self, trained_model):
        # Diccionario de terrenos
        terr_dict = {0: 'Carretera', 1: 'Tierra Seca', 2: 'Tierra Lodosa', 3: 'Pedregoso'}

        # Función para determinar velocidad recomendada
        def set_speed(terrain):
            return {
                'Carretera': '70 KPH',
                'Pedregoso': '30 KPH',
                'Tierra Seca': '50 KPH',
                'Tierra Lodosa': '20 KPH'
            }.get(terrain, 'N/A')


        cols = st.columns([3, 5])

        with cols[0]:
            st.title('Clasificador de Terreno')
            uploaded_file = st.file_uploader('Sube una imagen de terreno', type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            # Mostrar imagen redimensionada
            image = Image.open(uploaded_file)

            with cols[1]:
                st.image(image, caption='Imagen cargada', use_container_width=True)

            # Preprocesamiento de imagen
            img_size_display = (750, 500)
            img_train_size = (300, 450)
            
            image = image.resize(img_size_display)
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

            with cols[0]:
                cols_2 = st.columns([5, 6], vertical_alignment='bottom')
                with cols_2[0]:
                    self.show_card('Terreno detectado', detected_terrain)
                
                with cols_2[1]:
                    self.show_card('Velocidad recomendada', set_speed(detected_terrain))
                
                cols_2 = st.columns([5, 6], vertical_alignment='center')
                with cols_2[0]:
                    real_terrain = st.selectbox('**Selecciona el tipo de terreno real**', options=list(terr_dict.values()), index=None)

                with cols_2[1]:
                    self.show_card('Velocidad esperada', set_speed(real_terrain))

if __name__ == '__main__':
    TerrainDetector()