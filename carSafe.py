import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# To store model Jaiman----------------------------
import joblib

# Filename in which sk learn model will be saved----------------------------------
MODEL_FILENAME = 'knn_model'

data = pd.read_csv("car.data")  # Read data using pandas

data.drop_duplicates()
# Convert non numerical data into numerical data using Sklearn

# take the labels and convert them into appropriate integer values
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))  # returns array
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["classes"]))

predict = "class"

print("Safety :", safety, " buying :", buying)

X = list(zip(buying, maint, door, persons, lug_boot, safety))  # Features
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1)

knn_model = KNeighborsClassifier(n_neighbors=7)  # Setting K Neighbour value

knn_model.fit(x_train, y_train)

print("Predict : ", x_test)
predicted = knn_model.predict(x_test)  # setting prediction for out data set
# print("Predicted :", predicted)

acc = knn_model.score(x_test, y_test)  # Getting Accuracy of our model
print("score : ", acc)

# save model-------------------------------------------
knn_new_model = open("knn_model.pkl", "wb")
joblib.dump(knn_model, knn_new_model)
knn_new_model.close()

names = ["unacceptable", "acceptable", "good", "Excellet"]
