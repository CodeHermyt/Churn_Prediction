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

# Importing Keras and Packages
# Sequential - to initialise ANN
# Dense - to create layers

import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing ANN with Sequential()
classifier = Sequential()
# Adding Input Layer And first Hidden layer to ANN
classifier.add(keras.Input(shape=(11,)))
classifier.add(Dense(units= 6, activation= "relu", kernel_initializer='uniform')) 
#kernel_initializer='uniform' initializes weights uniformly near zero
# units are the number of nodes in the layer

#Adding Second hidden layer
classifier.add(Dense(units = 6, activation='relu', kernel_initializer='uniform'))

#Adding Output layer 
classifier.add(Dense(units=1, activation = 'sigmoid', kernel_initializer='uniform'))
# Output layer will have only one node, thus units = 1
# Output is the binary prediction so used sigmoid function as activation function

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training our model
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predition on X_test

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion_metrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)










