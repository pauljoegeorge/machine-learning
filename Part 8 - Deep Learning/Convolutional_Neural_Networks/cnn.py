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

"""
# STEP1: adding convolutional layer (Layer 1)
"""
classifier.add(Convolution2D(32,3,3, input_shape=(64, 64, 3), activation = 'relu'))  #64 feature map , 3 rows and 3 cols for feature detector
# we are working on cpu not gpu , so feature map has to be low value.  
# input shape for colored image (3,256,256) => 3 channel (RGB), size 256 pixel each (2D)
# since we are using tensorflow backend input shape input will in (64, 64, 3) format
# if using Theano order will be reverse
# Add activation layer to eliminate non-linearity

"""
# STEP 2 -> POOLING
"""
# Slide with stride of 2 (not one ) and will take the max value
# in step 1 stride applied is 1 
# Why pooling? reduce the size by 75% without removing the unique features
classifier.add(MaxPooling2D(pool_size = (2, 2)))

"""
# STEP 3 -> FLATTENING
"""
# create a single vector using pooling output as input
classifier.add(Flatten())

"""
# STEP 4 -> FULLY CONNECTED LAYER add(input layer, hidden layer, output layer)
"""
classifier.add(Dense(128, kernel_initializer="uniform", activation = 'relu')) # hidden layer
# output_dim = no of hidden layer nodes
classifier.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid')) # for output layer-> sigmoid
# expecting just one output, cat or dog. so output_dim is 1


"""
STEP 5 -> COMPILE WHOLE MODEL
"""
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# loss -> since we have binary outcome -> binary cross entropy