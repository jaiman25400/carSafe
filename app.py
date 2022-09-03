import streamlit as st
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import joblib

from PIL import Image

from carSafe import getKnnData


@st.cache
def load_dataset(dataset):
    columns = ['buying', 'maint', 'doors',
               'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(dataset, names=columns)
    return df


def load_prediction_models(model_file):
    load_model = joblib.load(open(os.path.join(model_file), "rb"))
    return load_model


buying_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
maint_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
doors_label = {'2': 0, '3': 1, '5more': 2, '4': 3}
persons_label = {'2': 0, '4': 1, 'more': 2}
lug_boot_label = {'small': 0, 'big': 1, 'med': 2}
safety_label = {'high': 0, 'med': 1, 'low': 2}
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
    persons = st.number_input("Select Number of Persons: ", 2, 10)
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
    print("Final Data :", final_data, k_buying, buying_label)
    st.subheader("options selected")
    st.json(final_data)

    st.subheader("Data Encoding")

    sample_data = [k_buying, k_maint, k_doors,
                   persons, k_lug_boot, k_safety]
    st.write(sample_data)

    prep_data = np.array(sample_data).reshape(1, -1)
    print(" Prep DAta : ", prep_data)
    model_choices = st.selectbox(
        "Model Type", ['KNNClassifier'])

    if st.button('Evaluate'):
        #         if model_choices == 'Logistic Regression':
        #             pred = load_prediction_models("models/logit_model.pkl")
        #             y_pred = pred.predict(prep_data)
        #             st.write(y_pred)
        #         if model_choices == 'Random Forest':
        #             pred = load_prediction_models("models/rf_model.pkl")
        #             y_pred = pred.predict(prep_data)
        #             st.write(y_pred)
        #         if model_choices == "MLP Classifier":
        #             pred = load_prediction_models("models/nn_model.pkl")
        #             y_pred = pred.predict(prep_data)
        #             st.write(y_pred)
        print("Evaluate")
        getKnnData(prep_data)

        # y_pred = getKnnModel(prep_data)
        # print("y pred :", y_pred, class_label)
        # final_result = get_key(y_pred, class_label)
        # print("Final result : ", final_result)
        # st.success(final_result)


if __name__ == '__main__':
    main()
