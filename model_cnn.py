#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:25:58 2020

CNN model source code

@author: max
"""



import keras
import numpy as np
from keras.layers import Input, LSTM, Dense,Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model
import matplotlib.pyplot as plt

   
# Raster input, meant to receive 30x30 matrices that represent the neighorhoof of pixels of interest
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(30,30,1,), dtype='float32', name='POI_HOOD')

# A convolutional layer will transform the vector neighborhood into a feature layer 
# 64 filters (number of output filters in convolution)
# 3x3 kernel size
# 
x = AveragePooling2D(pool_size=(2, 2))(main_input)
x = Conv2D(32, (3,3), input_shape=(30,30), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3,3), input_shape=(30,30), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dense(128,activation = 'relu')(x)
conv_out = MaxPooling2D(pool_size=(2, 2))(x)


atmospheric_input = Input(shape=(1,1,4,), name = 'atmospheric_input')

# concatenate auxillary input to the CNN layers : this makes a 136 dense layer
merged = keras.layers.concatenate([conv_out, atmospheric_input])

# probability of pixel of interest being on fire
final_out = Dense(1, activation='sigmoid',name = 'final_out')(merged)

model = Model(inputs = [main_input, atmospheric_input], outputs = [ final_out])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[ 0.2])
model.summary()
pixel_hood = np.round(np.abs(np.random.rand(20,30,30,1) * 100))

atmospheric_data = np.random.randn(1,1,20,4)

labels = np.random.randint(0,2,size=20)
labels.resize(20,1,1,4)


model.fit([pixel_hood, atmospheric_data], [labels, labels],
          epochs=50, batch_size=32)
    

    
    

    



