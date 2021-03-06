#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 07:41:19 2019

@author: pauljoegeorge
"""
import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

#importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values  #takes all columns except the last one (independent variables)
Y = dataset.iloc[:, 4].values   #index value from 0



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

# Fitting classifier  to training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)   #rbf kernal is used for non-linear
classifier.fit(X_train, Y_train)

"""
rbf converts non linear seperable to linear separable and that too in different dimensional space,
and  data will be categorised
then later dimension has to be converted back
"""

# predicting the test set results
y_pred = classifier.predict(X_test)

"""
manually test whether y_pred is same as y_test 
"""

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)  # left to right diagonal -> correct prediction .. leftt to right -> incorrect prediction.
"""
|64, 4|
|      |   ==> 63 + 25 correct predictions, (5 + 7 )  wrong predictions
|3, 29|
"""

"""
Applying k-fold cross validation
cv - no of cross validations to be performed ie; we will get 10 accuracies
"""
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
accuracies.mean()
accuracies.std() #gives the standard deviation


"""
Applying grid search to find best model and best parameters
C: parameter of SVC , its the penalty, don't make it a very high value
"""
from sklearn.model_selection import GridSearchCV
# check SVC params to find the below
parameters = [{'C': [1,10,100, 1000], 'kernel': ['linear']}, #linear
              {'C': [1,10,100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.0001]} #non-linear
              ] # 2 dictionaries
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10, 
                           n_jobs = -1,
                           iid=True)
grid_search_res = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search_res.best_score_
best_parameters = grid_search_res.best_params_  # will give the best values for c , gamma and kernel 
"""
# To improve the paramters , change the value  of Gamma
# ie; a value better than received in best_paramters
"""
# visualising the Training set results
#green & red region are the predictions. green dot -> they purchased , red dot -> didnt purchase
# straight line in graph -> prediction boundary.   
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
  

# visualising the Test set results
#green & red region are the predictions. green dot -> they purchased , red dot -> didnt purchase
# straight line in graph -> prediction boundary.   
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),   #step = 0.01 => resolution 
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# apply classifier on all pixels 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-

