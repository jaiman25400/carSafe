import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# To store model ----------------------------
import joblib

# Filename in which sk learn model will be saved----------------------------------
MODEL_FILENAME = 'knn_model'

data = pd.read_csv("car.data")  # Read data using pandas

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


X = list(zip(buying, maint, door, persons, lug_boot, safety))  # Features
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1)


knn_model = KNeighborsClassifier(n_neighbors=5)  # Setting K Neighbour value

knn_model.fit(x_train, y_train)


def classifier(model, X_train_res, X_test, y_train_res, y_test):
    clf = model
    clf.fit(x_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.fit(x_train, y_train)
    return model


lr_model = classifier(
    LogisticRegression(), x_train, x_test, y_train, y_test)

logit_model = open("logit_model.pkl", "wb")
joblib.dump(lr_model, logit_model)
logit_model.close()


knn1_model = classifier(
    KNeighborsClassifier(n_neighbors=5), x_train, x_test, y_train, y_test)

# save model-------------------------------------------
knn_new_model = open("knn1_modelData.pkl", "wb")
joblib.dump(knn1_model, knn_new_model)
knn_new_model.close()

acc = knn_model.score(x_test, y_test)  # Getting Accuracy of our model
print("score : ", acc)
# print("Predict : ", x_test)
# predicted = knn_model.predict(x_test)  # setting prediction for out data set
# names for our classifier to classify our data
names = ["unacceptable", "acceptable", "good", "Excellet"]

# for x in range(len(predicted)):
#     print("Predicted: ", names[predicted[x]], "Data: ",
#           x_test[x], "Actual: ", names[y_test[x]])
#     # returning distance and index of neighbours
#     # n = knn_model.kneighbors([x_test[x]], 9, True)
#     # print("N: ", n)


def getKnnData(data):
    print("data to be predicted ", data)

# chart_predicted_class = []
# chart_actual_class = []

# for x in range(len(predicted)):
#     for i in range(4):
#         if names[i] == names[predicted[x]]:
#             chart_predicted_class.append(names[predicted[x]])
#         if names[i] == names[y_test[x]]:
#             chart_actual_class.append(names[y_test[x]])

# chart_predicted_class_freq = []
# chart_actual_class_freq = []

# for name in names:
#     chart_predicted_class_freq.append(chart_predicted_class.count(name))
#     chart_actual_class_freq.append(chart_actual_class.count(name))

# plt.subplot(1, 2, 1)
# plt.pie(chart_predicted_class_freq, autopct='%0.1f%%')
# plt.axis('equal')

# plt.subplot(1, 2, 2)
# plt.pie(chart_actual_class_freq, autopct='%0.1f%%')
# plt.axis('equal')
# plt.legend(names)
# plt.show()

# bar = plt.subplot()
# n = 1
# t = 2
# d = len(names)
# w = 0.8
# x_values = [t * element + w * n for element in range(d)]
# predicted_x = x_values
# plt.bar(predicted_x, chart_actual_class_freq)

# n = 2
# t = 2
# d = len(names)
# w = 0.8
# x_values = [t * element + w * n for element in range(d)]
# actual_x = x_values
# plt.bar(actual_x, chart_actual_class_freq)
# plt.legend(['Predicted', 'Actual'])
# bar.set_xticks(range(8))
# bar.set_xticklabels([' ', 'Unacceptable', ' ', 'Good',
#                     ' ', 'Very Good', ' ', 'Acceptable'])
# plt.title("Bar Chart Comparison of Actual and Predicted classifications of cars")
# plt.show()
