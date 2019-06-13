# -*- coding: utf-8 -*-


import numpy as np  #contain mathematical tools
import matplotlib.pyplot as plt  #used to plot charts
import pandas as pd #import and manage datasets

# Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
"""
* at each round n for ad i  , consider 
  - number of times the ad i got reward 1 up to round n 
  - number of times the ad i got reward 0 up to round n 
* for each ad i , perform a random draw
* select the ad that has highest value
"""
#step 1
import random 
N = 10000
d = 10
ads_selected = [] 
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

#step 2 
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i] + 1 )
        if random_beta > max_random:
            max_random = random_beta  #updating maximum upper bound
            ad = i  # ad selected with maximum upper bound
    
    ads_selected.append(ad)
    #numbers_of_selections[ad] =  numbers_of_selections[ad] +1 #if ad selected, so increment value from 0 to 1
    reward = dataset.values[n, ad] # will be 0 or 1
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + reward
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0 [ad] + reward
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()