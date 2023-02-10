import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

filename = "../data/processed/heart_disease_processed.csv"

df = pd.read_csv(filename)

df.head()

df = df.drop("Unnamed: 0", axis=1)

df.info()

# ----------------------------------------------------------------
# Getting the statisticcal measure iof the given dataset
# ----------------------------------------------------------------

df.describe()

df.groupby("Outcome").mean()

# ----------------------------------------------------------------
# Separating the data and the labels in the dataset
# ----------------------------------------------------------------

X = df.drop("Outcome", axis=1)
Y = df["Outcome"]

print(X)

print(Y)

# ----------------------------------------------------------------
# Data Standardisation
# ----------------------------------------------------------------

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

print(standardized_data)


X = standardized_data

# ----------------------------------------------------------------
# Implementing train test split
# ----------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)


# ----------------------------------------------------------------
# Training the model
# ----------------------------------------------------------------

classifier = svm.SVC(kernel="linear")

classifier.fit(X_train, Y_train)


# ----------------------------------------------------------------
# Evaluating the model, first on the training set and then the
# testing set
# ----------------------------------------------------------------

X_train_prediction = classifier.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of the training data: {}".format(training_data_accuracy))

X_test_prediction = classifier.predict(X_test)

testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of the testing data: {}".format(testing_data_accuracy))

# ----------------------------------------------------------------
# Creating a rough predicttive system that will allow the user to
# set the values, edit the values ion the input data to show any change
# ----------------------------------------------------------------

input_data = (26.58, 1, 0, 0, 20.0, 30.0, 0, 1, 5, 2, 1, 1, 8.0, 1, 0, 0, 67)

input_np_array = np.asarray(input_data)

reshaped_input_data = input_np_array.reshape(1, -1)

standardized_data_input = scaler.transform(reshaped_input_data)

prediction = classifier.predict(standardized_data_input)
print(prediction)

if prediction[0] == 0:
    print(
        "The person with the given attribuites has a higher probability of not having any heart diseases"
    )
else:
    print(
        "The person with the given attribuites has a higher probability of having any heart diseases"
    )


# ----------------------------------------------------------------
# Saving the Model using pickle
# ----------------------------------------------------------------

import pickle

filename = "heartDiseaseModel.sav"
pickle.dump(classifier, open(filename, "wb"))
