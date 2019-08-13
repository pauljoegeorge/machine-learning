# -*- coding: utf-8 -*-

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  #takes all indexes upto index 12 (0..12)(independent variables)
Y = dataset.iloc[:, 13].values   #index value from 0

# Encoding categorical data (converting string values to encoded values 001, 010 etc...)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  #Country
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # 0 or 1 or 2
labelencoder_X_2 = LabelEncoder()  #gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # 0 or 1
#dummy encoding -> convert it to dummy encoding . example: if spain(1,0,0) france (0,1,0) us (0,0,1)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #removing index to avoid confusions


#split data set into training set

#2 set: training set and test set
from sklearn.model_selection import train_test_split
# out of 10 observations 2 will be in test set and 8 in training set
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.25, random_state=0)

#feature scaling -> scale all values to same range(currently salary and age are not in same range for example)
# not applied to Y_train and Y_test : depdendent variables

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Making ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize ANN
classifier = Sequential()  #sequence of layers

# Adding input layer and first hidden layer
                #no of nodes for hidden layer, initalizers, activation mode for hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_shape=(11,)))   
