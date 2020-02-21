#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:54:52 2020

@author: max

landscapes are composed of three layers, the first indicating the state of the fire at a given time.
The second indicating the lidar layer, the third indicating the NDVI index 

"""
import numpy as np
import matplotlib.pyplot as plt
from augmentation_tests import augement_array_driver

N_CHANNELS = 6
WIN_SIZE = 30

def sample_map(map_size):
    """
    Input :
            map_size : size of landscape array
    Output :
            x,y coordinates of sampled point within landscape
    Description:
            Sample the state map for pixels of interest, return coordinates
            make sure that 
    """
    x = np.random.randint(15,map_size-15)
    y = np.random.randint(15,map_size-15)
    
    # only select pixels of interest that are not going to be outside of 

    
    return x,y
 
def generate_sample(x,y,channels_map):
    
    """
    Input :
             x,y  : coordinates of sampled point within landscape
             channel_map : NxNx3 matrix containing three data channels in order [lidar, perimeter, NDVI]
             
             
    Output : 
        sample : 30x30x6 sample 
           
    Description :
        
            subset the matrix via desired boundaries
    """

    max_x = x + 15
    min_x = x - 15
    
    max_y = y + 15
    min_y = y - 15
    
    # now have corners of box 
    
    # trim to sample
    
    sample = channels_map[min_x:max_x, min_y:max_y,:]
    
    # check pad the sample in case the bounding box is cut off by the border of the map
    if sample.shape != (WIN_SIZE,WIN_SIZE, N_CHANNELS):
        print("problem")
    return(sample)
    
# generate 100 neighborhood samples

def generate_samples(channels_map,future_state_map,num_samples = 100):
    # array with dimensions 30x30x3xnum_samples holding the lidar, state and NDVI sampled matrices 
        
    neighborhoods = np.zeros([num_samples,WIN_SIZE,WIN_SIZE,N_CHANNELS])
    labels = np.zeros(num_samples)
    for sample_idx in range(num_samples):
        # generate x,y pairs for pixel of interest (POI)
        map_size = channels_map.shape[0]
        
        x,y = sample_map(map_size)
     
        # test of point is outside of perimiter, channels_map[x,y,1] == 2 
          
        # once valid point has been generated, sample perimeter
        sample_neighborhood = generate_sample(x,y,channels_map)
        
        plt.imshow(sample_neighborhood[:,:,1])
        
        neighborhoods[sample_idx,:,:,:] = sample_neighborhood
        labels[sample_idx] = future_state_map[x,y]
    
    return(neighborhoods, labels)
    
def driver():

    
    # currently hard coded, this will need to change
    state_zero = np.load("landscape_king/burn_0.npy")
    state_one = np.load("landscape_king/burn_1.npy")
    state_two = np.load("landscape_king/burn_2.npy")
    elevations = np.load("landscape_king/topography.npy")
    red = np.load("landscape_king/red_layer.npy")
    blue = np.load("landscape_king/blue_layer.npy")
    green = np.load("landscape_king/green_layer.npy")
    infred = np.load('landscape_king/infred_layer.npy')
    
    # someone needs to collect
    
    channels_map_one = np.zeros([500,500,N_CHANNELS])
    channels_map_one[:,:,0] = elevations
    channels_map_one[:,:,1] = state_zero
    channels_map_one[:,:,2] = red
    channels_map_one[:,:,3] = blue
    channels_map_one[:,:,4] = green
    channels_map_one[:,:,5] = infred
    
    channels_map_one = np.zeros([500,500,N_CHANNELS])
    channels_map_one[:,:,0] = elevations
    channels_map_one[:,:,1] = state_one
    channels_map_one[:,:,2] = red
    channels_map_one[:,:,3] = blue
    channels_map_one[:,:,4] = green
    channels_map_one[:,:,5] = infred


    # duplicate data with augmentation
    
    

    
    X_train,labels_train = generate_samples(channels_map_one,state_one,num_samples = 100)

    X_test,labels_test = generate_samples(channels_map_one,state_two,num_samples = 100)
    
    X_train_aug, labels_train_aug = augement_array_driver(X_train,labels_train)

    
    np.save("data/x_train",X_train_aug)
    np.save("data/y_train",labels_train_aug)
    
    np.save("data/x_val",X_test)
    np.save("data/y_val",labels_test)

    
    return(X,labels)
    

    
samples = driver()

    
    

    
        
        
    
    
        