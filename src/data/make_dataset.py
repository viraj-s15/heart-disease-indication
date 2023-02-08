import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename = "../../data/raw/heart_data_2020.csv"

df = pd.read_csv(filename)

df.head()

# ----------------------------------------------------------------
# Processing the data
# ----------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

df["Outcome"] = lb.fit_transform(df["HeartDisease"])
df["Smoking"] = lb.fit_transform(df["Smoking"])
df["AlcoholDrinking"] = lb.fit_transform(df["AlcoholDrinking"])
df["Stroke"] = lb.fit_transform(df["Stroke"])
df["DiffWalking"] = lb.fit_transform(df["DiffWalking"])
df["Sex"] = lb.fit_transform(df["Sex"])
df["Race"] = lb.fit_transform(df["Race"])
df["Diabetic"] = lb.fit_transform(df["Diabetic"])
df["PhysicalActivity"] = lb.fit_transform(df["PhysicalActivity"])
df["GenHealth"] = lb.fit_transform(df["GenHealth"])
df["Asthma"] = lb.fit_transform(df["Asthma"])
df["KidneyDisease"] = lb.fit_transform(df["KidneyDisease"])
df["SkinCancer"] = lb.fit_transform(df["SkinCancer"])

df["AgeCategory"] = df["AgeCategory"].replace("80 or older", "80-99")

# ----------------------------------------------------------------
# Taking the median age from the given data as the age range is not
# easy to work with
# ----------------------------------------------------------------

df.info()

df = df.astype({"AgeCategory": "string"})

age = []
for value in df["AgeCategory"]:
    n1 = int(value[0:2])
    l = len(value)
    n2 = int(value[l - 2 :])
    value = int((n1 + n2) / 2)
    age.append(value)

df["Age"] = age

# ----------------------------------------------------------------
# All the needed data has been processed, the columns that are not
# needed ust be removed from the dataframe
# ----------------------------------------------------------------

df = df.drop(["AgeCategory", "HeartDisease"], axis=1)

df.describe()

# ----------------------------------------------------------------
# The data set is finall processed, exporting the dataset now
# ----------------------------------------------------------------

df.to_csv("../../data/processed/heart_disease_processed.csv")
