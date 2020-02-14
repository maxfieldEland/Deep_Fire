#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:25:58 2020

CNN model source code

@author: max
"""



import keras
import numpy as np
from keras.layers import Input, LSTM, Dense,Conv2D, MaxPooling2D, AveragePooling2D,Flatten,Dropout
from keras.models import Model
import matplotlib.pyplot as plt



# Raster input, meant to receive 30x30 matrices that represent the neighorhood of pixels of interest
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(30,30,3,), dtype='float32', name='POI_HOOD')
atmospheric_input = Input(shape=(4,), name = 'atmospheric_input')

# A convolutional layer will transform the vector neighborhood into a feature layer 
# 64 filters (number of output filters in convolution)
# 3x3 kernel size
# 
x = AveragePooling2D(pool_size=(2, 2))(main_input)
x = Conv2D(32, (3,3), input_shape=(15,15), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3,3), input_shape=(7,7), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
flat = Flatten()(x)
conv_out = Dense(128,activation = 'relu')(flat)

# concatenate auxillary input to the CNN layers : this makes a 136 dense layer
merged = keras.layers.concatenate([flat, atmospheric_input])

# probability of pixel of interest being on fire
final_out = Dense(1, activation='sigmoid',name = 'final_out')(merged)

model = Model(inputs = [main_input, atmospheric_input], outputs = [ final_out])
model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[0.2], metrics =['accuracy'])
model.summary()


# --------------------------------------------------------------------- data summary ------------------------------------------------------

pixel_hood = np.load('data/x_train.npy')

atmospheric_data = np.tile(np.array([90,32.35,17,0.2]),(100,1))
labels = np.load('data/y_train.npy')
# transform labels to zero and one
labels[labels == 2] = 0

hist  = model.fit([pixel_hood, atmospheric_data], labels,epochs=100, batch_size=10)

history =  hist.history
loss = history['loss']
plt.figure(figsize = (10,10))
plt.plot(range(len(loss)),loss)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.show()


acc = history['accuracy']
plt.figure(figsize = (10,10))
plt.plot(range(len(acc)),acc)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.show()



