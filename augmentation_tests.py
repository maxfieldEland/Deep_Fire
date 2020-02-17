#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:16:54 2020

@author: natshenton
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def data_load():
    X = np.load("data/x_train.npy")
    y = np.load("data/y_train.npy")
    
    return X,y
    

def flip_image(img):
    flip = np.flip(img)
    return flip

def fliplr_image(img):
    return np.flipud(img)
    
def flipud_image(img):
    return np.fliplr(img) 
    
def data_augmentation(X,y):
    num_samples = X.shape[0]
    big_X = np.zeros([num_samples*4,30,30,3])
    big_y = np.zeros(num_samples*4)
    i = 0
    for n, img in enumerate(X):
        big_X[i] = img
        big_y[i] = y[n]
        i += 1
        big_X[i] = flip_image(img)
        big_y[i] = y[n]
        i += 1
        big_X[i] = fliplr_image(img)
        big_y[i] = y[n]
        i += 1
        big_X[i] = flipud_image(img)
        big_y[i] = y[n]
        i += 1
    
    return big_X,big_y
    
   
if __name__ == '__main__':
    X, y = data_load()
    big_X, big_y = data_augmentation(X,y)
        
print(big_y)  
        
            
        
    
    