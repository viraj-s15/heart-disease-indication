import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = "../../data/processed/heart_disease_processed.csv"
df = pd.read_csv(filename)

df.head()


df = df.drop("Unnamed: 0", axis=1)

# ----------------------------------------------------------------
# Seeing the correlation between all the values in the dataframe
# ----------------------------------------------------------------

correlation = df.corr()

correlation.style.background_gradient(cmap="coolwarm")

# ----------------------------------------------------------------
# Creating some ground rules we cans ay that a positive corrlelation of
# 0.1-0.5 can be considered as a slightly strong correlation however
# a correlation of 0.5-1 can be considered as a stron correlation.
# A negative correlation refers to the two variables in question to
# be inversely related to one another. Using this we can conclude that:
# 1) Physcial Activity and BMI have a slightly negative correlation
# 2) Difficult in walking and physcial health have a slightly strong positive correlation,
# along with diabetes
# 3) None of the correlations are too high which means no one particular variable
# decides the result completely on its own
# 4) Skin cancer and age have a relatively high correlation
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Visualization of outliers
# ----------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(data=df, width=0.6, whis=500, ax=ax, fliersize=5)
plt.xticks(rotation=90)
plt.title("Visualization of outliers")


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(
    y=df["Age"],
    x=df["PhysicalActivity"],
    ax=ax[0],
    width=0.4,
)

sns.boxplot(
    y=df["BMI"],
    x=df["PhysicalActivity"],
    ax=ax[1],
    width=0.4,
)
plt.tight_layout()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(
    y=df["Diabetic"],
    x=df["PhysicalActivity"],
    ax=ax[0],
    whis=500,
    width=0.4,
)

sns.boxplot(
    y=df["GenHealth"],
    x=df["PhysicalActivity"],
    ax=ax[1],
    whis=500,
    width=0.4,
)
plt.tight_layout()


plt.figure(figsize=(14, 7))
sns.countplot(
    x="Age",
    hue="PhysicalActivity",
    data=df,
    palette="colorblind",
    edgecolor=sns.color_palette(as_cmap="RdGy_r", n_colors=5),
)
print("0 = Physical Activity", "1 = No Physical Activity")

plt.figure(figsize=(14, 7))
sns.countplot(
    x="SkinCancer",
    hue="PhysicalActivity",
    data=df,
    palette="colorblind",
    edgecolor=sns.color_palette(as_cmap="RdGy_r", n_colors=5),
)
print("0 = Physical Activity", "1 = No Physical Activity")
