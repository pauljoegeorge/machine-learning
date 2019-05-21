# -*- coding: utf-8 -*-

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
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()  # try different available kernels 
classifier.fit(X_train, Y_train)



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
|5, 27|
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
plt.title('Naive Bayes (Training set)')
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
plt.title('Naive Bayes(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

