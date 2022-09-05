import streamlit as st
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import joblib

from PIL import Image

# import filename constant-------------------
from carSafe import MODEL_FILENAME, names


@st.cache
def load_dataset(dataset):
    columns = ['buying', 'maint', 'doors',
               'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(dataset, names=columns)
    return df


def load_prediction_models(model_file):
    load_model = joblib.load(open(os.path.join(model_file), "rb"))
    return load_model


# load model -------------------------------
trained_knn_model = joblib.load("knn_model.pkl")

buying_label = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
maint_label = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
doors_label = {'2': 0, '3': 1, '4': 3, '5more': 2}
persons_label = {'1': 0, '2': 1, '3': 2, '4': 3, 'more': 3}
lug_boot_label = {'small': 3, 'med': 2, 'big': 1, 'vbig': 0}
safety_label = {'vhigh': 0, 'high': 1, 'med': 2, 'low': 3}
class_label = {'Unacceptable': 0, 'Acceptable': 1, 'Good': 2, 'Very Good': 3}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def main():

    st.title("Car Safe Project")
    image = Image.open('assets/index.png')
    st.image(image, use_column_width=True)

    st.subheader("Select below")
    buying = st.selectbox("Select Buying Level: ",
                          tuple(buying_label.keys()))
    maint = st.selectbox("Select Maintainence Level: ",
                         tuple(maint_label.keys()))
    doors = st.selectbox("Select Number of Doors: ",
                         tuple(doors_label.keys()))
    persons = st.number_input("Select Number of Persons: ", 1, 10)
    lug_boot = st.selectbox("Select Luggage Boot: ",
                            tuple(lug_boot_label.keys()))
    safety = st.selectbox("Select Safety", tuple(safety_label.keys()))
    k_buying = get_value(buying, buying_label)
    k_maint = get_value(maint, maint_label)
    k_doors = get_value(doors, doors_label)
    k_lug_boot = get_value(lug_boot, lug_boot_label)
    k_safety = get_value(safety, safety_label)

    final_data = {
        "buying": buying,
        "maint": maint,
        "doors": doors,
        "persons": persons,
        "lug_boot": lug_boot,
        "safety": safety,
    }
    st.subheader("options selected")
    st.json(final_data)

    st.subheader("Data Encoding")

    print("F data : ", final_data)

    sample_data = [k_buying, k_maint, k_doors,
                   persons, k_lug_boot, k_safety]
    st.write(sample_data)

    prep_data = np.array(sample_data).reshape(1, -1)
    print(" Prep DAta : ", prep_data)
    model_choices = st.selectbox(
        "Model Type", ['KNNClassifier'])

    if st.button('Evaluate'):

        # using Logis regression start--------------
        # pred = load_prediction_models("logit_model.pkl")
        # y_pred = pred.predict(prep_data)
        # print("logit :", y_pred)
        # st.write(y_pred)
        # final_result = get_key(y_pred, class_label)
        # print("Final result : ", final_result)
        # st.success(final_result)
        # using logi ends---------------

        # knn start------------------
        knn_pred = load_prediction_models("knn_model.pkl")
        y_pred = knn_pred.predict(prep_data)
        print("Knn 1 : ", y_pred)
        st.write(y_pred)
        final_result = get_key(y_pred, class_label)
        print("Final result : ", final_result)
        st.success(final_result)
        # knn ends -----------------------


if __name__ == '__main__':
    main()
