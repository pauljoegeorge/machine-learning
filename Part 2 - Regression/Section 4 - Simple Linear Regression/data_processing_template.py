# -*- coding: utf-8 -*-
#simple linear regression
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:39:11 2019

@author: pauljoegeroge
"""

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  #takes all columns except the last one (independent variables)
Y = dataset.iloc[:, 1].values   #index value from 0

# taking care of missing data
"""
# replace Nan values by mean value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
"""

# Encoding categorical data
"""
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#convert countries to 0,1,2
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#dummy encoding -> convert it to dummy encoding . example: if spain(1,0,0) france (0,1,0) us (0,0,1)
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
encoding purchased state (yes/no)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
"""

#split data set into training set
#2 set: training set and test set
from sklearn.model_selection import train_test_split
# out of 10 observations 2 will be in test set and 8 in training set
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.3, random_state=0)


#feature scaling -> scale all values to same range(currently salary and age are not in same range for example)
# not applied to Y_train and Y_test : depdendent variables
"""
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

#fitting linear regression to training set
# by simple linear regression, it will learn the correaltion (y = mx + c ) 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
