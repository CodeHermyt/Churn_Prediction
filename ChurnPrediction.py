# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing Dataset

dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Data Prerocessing

# Encoding categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X_Geo = LabelEncoder()
X[:,1] = labelEncoder_X_Geo.fit_transform(X[:,1])

labelEncoder_X_Gender = LabelEncoder()
X[:,2] = labelEncoder_X_Gender.fit_transform(X[:,2])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)

X = ct.fit_transform(X)
X = X.tolist()
X = np.array(X)

X = X[:,1:]     #Droping first column to avoid dummy

# splitting data into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scaling X_train and X_test

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


















