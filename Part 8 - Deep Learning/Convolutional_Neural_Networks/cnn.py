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
# (32, 3,3) => size of feature detector and dimensions
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
Adding second convolution layer to improve the test_set accuracy
"""
#classifier.add(Convolution2D(32,3,3, activation = 'relu')) #input shape is skipped because input to second layer is not images
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
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


"""
    PART 2 - Fitting Images to CNN
"""
# Data augmentation
# is a technique to enrich data to get data performance without adding more images, but by tranforming the existing dataset ( like cropping , zooming , blurring etcc..)

from keras.preprocessing.image import ImageDataGenerator
# check keras documentation: Example of using .flow_from_directory(directory):
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)  #image preprocessing

# creating a training set of reduced size 64x64 and creating batches of 32 images.
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64), #dimensions specified same as above
                                                 batch_size=32,
                                                 class_mode='binary')
# creating a test set of reduced size 64x64 and creating batches of 32 images.
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64), #size of resultant (64 is mentioned above)
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)  