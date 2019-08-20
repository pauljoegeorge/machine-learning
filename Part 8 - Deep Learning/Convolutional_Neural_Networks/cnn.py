# -*- coding: utf-8 -*-

# solve image classification problem
# we have set of cat and dogs images and classify it accordingly using CNNN

# Part 1-  Building CNN
# Importing the Keras libraries and packages
from keras.models import Sequential # init neural network
from keras.layers import Convolution2D #to add CNN layers, de-pack images
from keras.layers import MaxPooling2D # helps pooling layers
from keras.layers import Flatten  #flattening 
from keras.layers import Dense   # dense layer

# init CNN
classifier = Sequential()

# adding convolutional layer (Layer 1)
