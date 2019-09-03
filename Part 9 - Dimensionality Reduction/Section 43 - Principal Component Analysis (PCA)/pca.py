# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:08:37 2019

@author: pauljoegeorge
"""

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

#importing dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values  #takes all columns except the last one (independent variables)
Y = dataset.iloc[:, 13].values   #index value from 0



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

"""
Applying PCA
- to reduce the dimensionality 
- selects the most relevant 2 independent variables
- ie; once applied only 2 indpendent variables will be used

n_components: no of prinicipal components expecting as output
"""
from sklearn.decomposition import PCA
pca = PCA(n_components =  2) # intially it was None, then changed it to 2 , by checking the variance sum of first 2 values ( greater than 50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_  # will give 13 values (variance) , first values the best and last is the worst 



# Fitting Logistic regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, Y_train)

# predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)  # left to right diagonal -> correct prediction .. leftt to right -> incorrect prediction.
"""
will have a 3x3 class /prediction
16 - correct number predictions  of segment 1
19 - correct number of predictions of segment 2 
18 - segment 3
"""

# visualising the Training set results
#green & red region are the predictions. green dot -> they purchased , red dot -> didnt purchase
# straight line in graph -> prediction boundary.   
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('PCA (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
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
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Refression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

