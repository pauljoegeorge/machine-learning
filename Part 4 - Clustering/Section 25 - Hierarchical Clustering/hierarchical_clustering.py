# -*- coding: utf-8 -*-

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

# Importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


"""
based on the customers salary and points it will try to draw a graph where each data point will be individual clusters.
and then based on it a dendogram will be drawn by the following code.
"""
# plot dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) # ward - method which tries to minimise the variance.
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian distance')
plt.show()

# fitting the hierarchical clustering to the dataset
from  sklearn.cluster import AgglomerativeClustering
# n_clusters = 5 , optimial number of clusters identified from dendogram
hc = AgglomerativeClustering(n_clusters =  5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_hc == 0, 0 ], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0 ], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0 ], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0 ], X[y_hc == 3, 1], s = 100, c = 'yellow', label = 'Careless')
plt.scatter(X[y_hc == 4, 0 ], X[y_hc == 4, 1], s = 100, c = 'cyan', label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('spending score (1-100)')
plt.legend()
plt.show()