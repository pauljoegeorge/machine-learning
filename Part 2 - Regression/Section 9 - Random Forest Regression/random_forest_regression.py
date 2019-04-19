#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:11:01 2019

@author: pauljoegeroge
"""

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


#fitting  random regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, Y)

y_pred = regressor.predict(np.array([[6]]))


#visualise linear regression model
plt.scatter(X, Y, color = 'red')
# increase x value by 0.1
x_grid = np.arange(min(X), max(X), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1) # no of columns: 1
plt.plot(x_grid, regressor.predict(x_grid),  color='blue')
#plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),  color='blue')
plt.title('Truth or Bluff (Random forest regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


"""
# predicting result with linear regression
lin_reg.predict([[6.5]])  # x_value: 6.5 

# predicting result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
"""