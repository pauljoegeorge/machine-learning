#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:40:53 2019

@author: pauljoegeroge

program to predict the salary; works like a fake salary detector
"""
#polynomial regression

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # takes all values of index 1  (:2 convert X to matrix)
Y = dataset.iloc[:, 2].values   #index value from 0


"""
Why no training set ???
Accurate prediction required and only less data available 
"""

"""
Build linear regression model and polynomial regression model for comparison
""" 

from sklearn.linear_model import LinearRegression
# Fitting linear regression to dataset
X = X.reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#fitting  polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)   #degree:2   b0 + b1x^1 + b2x^2
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)


#visualise linear regression model
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')

plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, Y, color = 'red')
"""
# increase x value by 0.1
# x_grid = np.arange(min(X), max(X), 0.1)
# x_grid = x_grid.reshape(len(x_grid), 1) # no of columns: 1
# plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)),  color='blue')
"""
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),  color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predicting result with linear regression
lin_reg.predict([[6.5]])  # x_value: 6.5 

# predicting result with polynomial regression
#lin_reg2.predict(poly_reg.fit_transform(6.5))
