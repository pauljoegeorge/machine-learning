# -*- coding: utf-8 -*-

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts and create corpus
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
"""
line 1: removing numbers, special characters
line 2: replace captial letters by small letters
line 3: splitting review into different words
line 4: remove irrelevant words for prediction( this, that , in , on etc..)
line 5: perform stepping and keep the root value only ( keep root word: `love` rather than keeping loved, loving, etc..)
line 6: join it
corpus : is a collection of text
"""
corpus  = []
for i  in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ',  dataset['Review'][i] )
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in  set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
# creating the bag of words model
"""
step 1: from 1000 reviews add all different words to a bag (without taking duplicates)
step 2:  create one column for each word -> each will have number (numbe of times same word is called.)
matrix containing most of the values as 0 -->  is known as sparse matrix
perform classification model (to differentiate 0 or 1 )
"""
# Applying classification algorithm
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)