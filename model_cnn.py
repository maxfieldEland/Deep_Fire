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
from keras import optimizers
N_CHANNELS = 6
WIN_SIZE = 30

# Raster input, meant to receive 30x30 matrices that represent the neighorhood of pixels of interest
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(WIN_SIZE,WIN_SIZE,N_CHANNELS,), dtype='float32', name='POI_HOOD')
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
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=adam, loss='binary_crossentropy',loss_weights=[0.2], metrics =['accuracy'])
model.summary()


# --------------------------------------------------------------------- data summary ------------------------------------------------------

pixel_hood = np.load('data/x_train.npy')

atmospheric_data = np.tile(np.array([90,32.35,17,0.2]),(400,1))
labels = np.load('data/y_train.npy')
# transform labels to zero and one
labels[labels == 2] = 0

pixel_hood_val = np.load('data/x_val.npy')
labels_val = np.load('data/y_val.npy')

hist  = model.fit([pixel_hood, atmospheric_data],labels, validation_data = ([pixel_hood_val, atmospheric_data[:100,:]], labels_val),epochs=30, batch_size=32)

history =  hist.history
loss = history['loss']
plt.figure(figsize = (5,5))
plt.plot(range(len(loss)),loss,label = "Training Loss")
plt.plot(range(len(history['val_loss'])),history['val_loss'], label = 'Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Loss over 100 Epochs")
plt.legend()
plt.show()


acc = history['accuracy']
plt.figure(figsize = (5,5))
plt.plot(range(len(acc)),acc,label = 'Training Accuracy')
plt.plot(range(len(history['val_accuracy'])),history['val_accuracy'],label = 'Validation Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Accuracy over 100 Epochs" )
plt.legend()
plt.show()
