#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:29:26 2019

@author: pauljoegeroge
"""

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  #takes all columns except the last one (independent variables)
Y = dataset.iloc[:, 2].values   #index value from 0


#split data set into training set
#2 set: training set and test set
"""
from sklearn.model_selection import train_test_split
# out of 10 observations 2 will be in test set and 8 in training set
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.3, random_state=0)
"""

#feature scaling -> scale all values to same range(currently salary and age are not in same range for example)
# not applied to Y_train and Y_test : depdendent variables

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = np.array(Y).reshape(-1, 1)
Y = sc_Y.fit_transform(Y)
 
#fitting SVR to training set
#create svr regressor
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

#predict the test set results (salary)
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]))) #scaled prediction
#to get original scale -> apply inverse transform
y_pred = sc_Y.inverse_transform(y_pred)


#visualise the `train set` results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')

plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#visualise the `SVR RESULTS` for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
