# -*- coding: utf-8 -*-

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

"""
we need to find the clusters for customers spending score
"""
#importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    """
     max_iter = maximum number of iteration when k-means algo is running, n_init = number of times k means algo will run with centroid 
    """
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init=10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)    
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


"""
from plotted wcss n_clusters = 5 is the recommended value
"""
# applying k-means to mall dataset
kmeans = KMeans(n_clusters = 5,init = 'k-means++', max_iter = 300, n_init=10, random_state = 0 )
y_kmeans =  kmeans.fit_predict(X)

"""
X[y_kmeans == 0, 0 ], X[y_kmeans == 0, 1] 
 ==> X[y_kmeans == 0, 0 ]   =========> X - coordinate
   *. find an X whose cluster = 0 and  value of 1st column of X
 ==> X[y_kmeans == 0,  1 ]  =========> Y - coordinate
   *. find an X whose cluster =0  and  value of 2nd column of X
"""
# visualising the clusters
# plotting the clusters
plt.scatter(X[y_kmeans == 0, 0 ], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0 ], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0 ], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0 ], X[y_kmeans == 3, 1], s = 100, c = 'yellow', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0 ], X[y_kmeans == 4, 1], s = 100, c = 'cyan', label = 'Sensible')
#plotting the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 300, c = 'magenta', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()