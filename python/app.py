import streamlit as st
from image import *
import numpy as np
from conf import *
from mlp import *
import requests
import os


def main():
    model = None
    img = None
    X_list = None

    st.set_page_config(page_title='Country Guesser', page_icon=":checkered_flag:")
    st.title("Country Guesser")

    with st.expander('Charger une image'):
        img_file = st.file_uploader("", type=['png', 'jpeg', 'jpg'])
        if img_file is not None:
            img = Image.open(img_file)
            X_list = transform_image(img, IMAGE_SIZE)
            X_list = np.array(X_list).flatten()
            X_list = X_list / 255.0
            col1, col2, col3 = st.columns(3)
            col2.image(img, width=220)

    with st.expander("Charger un modèle"):
        mlp_types = ['Sans couche cachée',
                     '1 couche cachée, 8 neurones',
                     '1 couche cachée, 32 neurones',
                     '2 couches cachées, 32 neurones']

        mlp_type_st = st.selectbox('Architecture du PMC', mlp_types)
        mbtn1, mbtn2, mbtn3, mbtn4 = st.columns(4)
        mlp_type_btn = mbtn1.button("Charger")

        if mlp_type_btn:
            if mlp_type_st == "Sans couche cachée":
                model = load_mlp_model(MLP_0HNL)
            elif mlp_type_st == "1 couche cachée, 8 neurones":
                model = load_mlp_model(MLP_1HNL_8N)
            elif mlp_type_st == "1 couche cachée, 32 neurones":
                model = load_mlp_model(MLP_1HNL_32)
            elif mlp_type_st == "2 couches cachées, 32 neurones":
                model = load_mlp_model(MLP_2HNL_32)

        if model is not None:
            mlp_destroy_btn = mbtn4.button("Détruire")
            if mlp_destroy_btn:
                destroy_mlp_model(model)
                st.success('Modèle détruit')

    with st.expander('Prédiction'):
        if model is not None and img is not None:
            if X_list is not None:
                predicted_output = predict_mlp_model_classification(model, X_list, 3)
                output = np.argmax(predicted_output)
                label = CLASSES[output]
                opt_col1, opt_col2, opt_col3 = st.columns(3)
                opt_col2.write(':point_right:     ' + label + '     :point_left:')
        else:
            st.write(":warning: Vous devez avoir chargé un modèle et une image afin de prédire :warning:")


if __name__ == '__main__':
    model = None
    img = None
    main()
