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
    x = np.random.randint(map_size)
    y = np.random.randint(map_size)
    
    while x > map_size - 16 and y > map_size - 16:
        x = np.random.randint(map_size)
        y = np.random.randint(map_size)

     
    return x,y
 
def generate_sample(x,y,channels_map):
    
    """
    Input :
             x,y  : coordinates of sampled point within landscape
             channel_map : NxNx3 matrix containing three data channels in order [lidar, perimeter, NDVI]
             
             
    Output : 
        sample : 30x30x3 sample 
           
    Description :
        
            subset the matrix via desired boundaries
    """

    max_x = x + 15
    min_x = x- 15
    
    max_y = y + 15
    min_y = y - 15
    
    # now have corners of box 
    
    # trim to sample
    sample = channels_map[min_x:max_x, min_y:max_y,:]
    
    # check pad the sample in case the bounding box is cut off by the border of the map
        
    return(sample)
    
# generate 100 neighborhood samples

def generate_samples(channels_map,num_samples = 100):
    # array with dimensions 30x30x3xnum_samples holding the lidar, state and NDVI sampled matrices 
        
    neighborhoods = np.zeros([30,30,3,num_samples])
    
    for sample_idx in range(num_samples):
        # generate x,y pairs for pixel of interest (POI)
        map_size = channels_map.shape[0]
        
        x,y = sample_map(map_size)
     
        # test of point is outside of perimiter, channels_map[x,y,1] == 2 
        while channels_map[x,y,1] != 1:
            x,y = sample_map(map_size)
            
        # once valid point has been generated, sample perimeter
        sample_neighborhood = generate_sample(x,y,channels_map)
        
        neighborhoods[:,:,:,sample_idx] = sample_neighborhood
        
    return(neighborhoods)
    
def driver():
    #state_zero = np.load("landscape_king/burn_0.npy")
    state_one = np.load("landscape_king/burn_1.npy")
    elevations = np.load("landscape_king/topography.npy")
    NDVI = np.random.rand(300,300)
    
    channels_map = np.zeros([300,300,3])
    channels_map[:,:,0] = elevations
    channels_map[:,:,1] = state_one
    channels_map[:,:,2] = NDVI
    
    
    
    X = generate_samples(channels_map,num_samples = 100)
    
    return(X)
    

    
samples = driver()
    
    

    
        
        
    
    
        