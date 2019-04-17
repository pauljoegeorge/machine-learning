#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:23:00 2019

@author: pauljoegeroge
"""

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  #takes all columns except the last one (independent variables)
Y = dataset.iloc[:, 4].values   #index value from 0

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#convert countries to 0,1,2
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#dummy encoding -> convert it to dummy encoding . example: if spain(1,0,0) france (0,1,0) us (0,0,1)
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
#taking all columns except the 0 index ( to avoid one dummy variable manually)
X = X[:, 1:]

#split data set into training set
#2 set: training set and test set
from sklearn.model_selection import train_test_split
# out of 10 observations 2 will be in test set and 8 in training set
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.2, random_state=0)

 
#fitting  multiple linear regression to training set
# by multiple linear regression, it will learn the correaltion (y = b0 + b1x1 + b2x2.... ) 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predict the test set results (profit)
y_pred = regressor.predict(X_test)

"""
# Manual implementation
"""
#currently all columns (relavant and least relavant) are used. but eliminating least relavant columns will optimize the prediction
#building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#y = b0 + b1x1 + b2x2...bnxn 
# b0 is a constant which has to be added to dataset (a constant value 1 will be added to each row as a new column)
# no of lines: 50 no of columns: 1, axis: 1(add column), axis: 0 (add line)
X = np.append(arr= np.ones((50,1)).astype(int), values = X, axis=1)
#removed unrelavant independent variables
X_opt = X[:, [0,1,2,3,4,5]]  #all independent variable values 
#step2: fit model with all possible predictors
regressor_OLS = sm.OLS(endog= Y, exog= X_opt).fit()
#see all values of P
regressor_OLS.summary()
#step3: remove the predictor with P > SL (0.05) ----> remove x2
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog= Y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog= Y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog= Y, exog= X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog= Y, exog= X_opt).fit()
regressor_OLS.summary()

"""
# Automatic implementation

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""


