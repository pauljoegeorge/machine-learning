# -*- coding: utf-8 -*-

import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

# Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#prepare input correctly (list of lists)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training apriori on the dataset
from apyori import apriori
"""
=> minimum support can be fixed based on number of times a product is purchased more than 3 times( it can be changed) a day 
=> minimum support , if a product is purchased 3 times a day , in a week its 7*3= 21 times in a week .. so lets fix 21/7500 as the minimum support.
=> so all products will have support higher than that as it is a minimum value
=> minimum confidence = 0.8 means it has to be correct 80% of the time. so lets fix 0.2 ir; 20% which is better as overconfidence is not good :P
=> for lift value 3-6 will be good to go.
"""
rules = apriori(transactions, min_support = 0.003, min_confidence=0.2, min_lift= 3, min_length = 2)
# Visualising the results, top 5 most relevant are displayed
results = list(rules)
for i in range(0, 5): 
  print(results[i])