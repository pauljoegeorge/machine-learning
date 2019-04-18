#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:33:48 2019

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
"""
#2 set: training set and test set
from sklearn.model_selection import train_test_split
# out of 10 observations 2 will be in test set and 8 in training set
X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.3, random_state=0)
"""

#feature scaling -> scale all values to same range(currently salary and age are not in same range for example)
# not applied to Y_train and Y_test : depdendent variables
"""
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""
 
#fitting decision tree  regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

#predict the results
y_pred = regressor.predict(np.array([[6.5]]))


#visualise the  decision tree results (not good for decision tree regression)
"""
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')

plt.title('Truth or bluff (Decision tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""
#visualise the  decision tree results in higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Truth or bluff (Decision tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()