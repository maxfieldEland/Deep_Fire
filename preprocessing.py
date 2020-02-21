#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:42:48 2020

@author: max
"""


from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import datetime 
import os
from rasterio.mask import mask
import rasterio
import fiona 
import cv2
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
from descartes import PolygonPatch
from shapely.geometry import box
import pycrs
import shapely.geometry as geometry
from fiona.crs import from_epsg
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
import copy
import rasterio.plot

def reproject_tif_func(file_path, new_file_path):
    """
    Input:
        file_path : path to file to reproject, must be .tif file
        new_file_path : path to file that will be written to store new projection
    
    Output:
        new_projection : returns new projection raster

    Reproject a raster from original projection to EPSG 4326 projection. Use optimal new heights and widths, apply padding as necessary 
    """
    
    dst_crs = 4326
    with rasterio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        with rasterio.open(new_file_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        return(rasterio.open(new_file_path))
    
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]






def crop_perim(perim_path, tif_file):
    """
    Input :     
        perim_path : path to the perimeter file (shx or shp)
        tif_file : loaded tif file 
        
    Output : 
        state_map_final : numpy nxn array that represents the states of each pixel at a given time
    
    Description : 
        Crop the original perimeter image that is mapped into the pixel space of the tif file
        Apply a 30 pixel buffer to the max and min of the shape after projection into pixel space 
    
    Sample Call : 
        crop_perim('ca_king_20140930_2301_dd83.shx', lidar_raster)

    """
    
    # open the shape file and extract the shape features
    
    with fiona.open(perim_path,'r') as shapefile:
        features = [feature['geometry'] for feature in shapefile]
    
    # handle the case of multipolygon
    coordinates = []
    if features[0]['type'] == 'MultiPolygon':
        for shape in features[0]['coordinates']:
            coordinates.extend(shape[0])
    elif features[0]['type'] == 'Polygon':
        
        coordinates = features[0]['coordinates'][0]
        
    latitude = []
    longitude = []
    for coord in coordinates:
        latitude.append(coord[1])
        longitude.append(coord[0])
    
    # transform the la/lon pairs into pixel coordinates to project them into pixel space
    affine_trans = tif_file.transform
    
    x, y = rasterio.transform.rowcol(affine_trans, xs = longitude, ys = latitude)
    
    img = tif_file.read(1)
    # construct state matrix based on perimeter
    state_final = np.ones([len(img),len(img)],)*2
    
    # switch the value of every coordinate in the x,y pairs to 1, indicating that it is on 
    for idx in range(len(x)):
        state_final[x[idx],y[idx]] = 1
    
    
    
    # crop image to desired size to limit useless computation
    # TODO: make this automated somehow.... 
        
    plt.matshow(state_final)
    # crop images
        
    # crop the elevation matrix
    cropped_elevation = img[500:1000, 1300:1800]
    # crop the state matrix
    cropped_states = state_final[500:1000, 1300:1800]
    
    
    # This section connects the dots when the transformation to pixel space leaves gaps in the perimiter shape
    # TODO : automate this
    # Idea : while hull_convex != true : dilate and erode img
    
    bw = invert(cropped_states)
    kernel = np.ones((1,20), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(bw, kernel, iterations=2)
    e_im = cv2.erode(d_im, kernel, iterations=1) 
    
    kernel = np.ones((20,1), np.uint8)  # note this is a verticle kernel
    d_im = cv2.dilate(e_im, kernel, iterations=2)
    e_im = cv2.erode(d_im, kernel, iterations=1) 
    
    # encode final statemap for fire perimeter
    state_mat_final = np.zeros([500,500])
    
    for idx,row in enumerate(e_im):
        for idy, pixel in enumerate(row):
            if pixel == -1 :
                state_mat_final[idx,idy] = 0
            else:
                state_mat_final[idx,idy] = 1
    plt.matshow(state_mat_final)
    state_mat_filled = ndimage.binary_fill_holes(state_mat_final).astype(int)
    plt.matshow(state_mat_filled)

    return(state_mat_filled,cropped_elevation)
    
    
def crop_tif(tif_path, new_file_path,bb_cords,shape_file_path):
    """
    Input:
        tif_path : path to tif file that you want to crop
        new_file_path : path to new file that will be written
        perim_image : fire perimter outline mapped into pixel space
    
    Output:
        clipped : cropped raster 
    
    Description : 
        Function to crop a tif file to align with an input image of the fire perimeter
        Save arrays 
    
    """
    
    reprojection = reproject_tif_func(tif_path,new_file_path)
    minx_pix, miny_pix, maxx_pix, maxy_pix = bb_cords
    
    # convert these pairs to lat/long cords using the affine transformation
    affine_trans = reprojection.transform

    maxx, maxy = affine_trans*(maxx_pix, maxy_pix)
    minx, miny = affine_trans*(minx_pix, miny_pix)

    # build bounding box for image crop
    bbox = box(minx,miny,maxx,maxy)

    # insert bounding box into geo data frame
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
    geo = geo.to_crs(crs=reprojection.crs.data)

    coords = getFeatures(geo)
    
    # use mask to get cropped image
    out_img, out_transform = mask(dataset=reprojection, shapes=coords, crop=True)
    
    #plt.imshow(out_img[0,:,:])
    # build new raster from image and edit old meta data to reflect new file
    out_meta = reprojection.meta.copy()
    epsg_code = int(reprojection.crs.data['init'][5:])
    
    #pycrs makes an api request requiring wifi

    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                      "transform": out_transform,
                     "crs": '+proj=longlat +datumiu=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +nodef'})
        
    out_tif = "reprojection.TIF"
    
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)
        
    clipped = rasterio.open(out_tif)

    return(clipped)




#- ---------------------------------------------------------------------------------- Load fire perimeter shape file -------------------------------------------------


# PREPROCESSING 

# Load shapefile and lidar tif file
initial_perim_path = 'neural_net_data_raw/King_Fire/King_Fire/ca_king_20140914_0000_dd83/ca_king_20140914_0000_dd83.shx'
e_county = rasterio.open('neural_net_data_raw/USGS_1_n39w121.tif')
state_initial, elevation_final = crop_perim(initial_perim_path, e_county)


final_perim_path = 'neural_net_data_raw/King_Fire/King_Fire/ca_king_20140915_0555_dd83/ca_king_20140915_0555_dd83.shx'
state_final, elevation_final = crop_perim(final_perim_path, e_county)

third_perim_path = 'neural_net_data_raw/King_Fire/King_Fire/ca_king_20140916_0113_dd83/ca_king_20140916_0113_dd83.shx'
state_third, elevation_final = crop_perim(third_perim_path, e_county)

# view initial and final outlines

# --------------------------------------------------------- crop and align spectral layers to cropped lidar -----------------------------------------

minx_pix, miny_pix = 1300, 500
maxx_pix, maxy_pix = 1800, 1000


bb_cords = [minx_pix, miny_pix, maxx_pix, maxy_pix]
blue_layer = crop_tif('neural_net_data_raw/LC08_L1TP_043033_20140818_20170304_01_T1/LC08_L1TP_043033_20140818_20170304_01_T1_B2.TIF','blue_new.tif',bb_cords, initial_perim_path)
green_layer = crop_tif('neural_net_data_raw/LC08_L1TP_043033_20140818_20170304_01_T1/LC08_L1TP_043033_20140818_20170304_01_T1_B2.TIF','green_new.tif',bb_cords, initial_perim_path)
red_layer = crop_tif('neural_net_data_raw/LC08_L1TP_043033_20140818_20170304_01_T1/LC08_L1TP_043033_20140818_20170304_01_T1_B2.TIF','red_new.tif',bb_cords, initial_perim_path)
infred_layer = crop_tif('neural_net_data_raw/LC08_L1TP_043033_20140818_20170304_01_T1/LC08_L1TP_043033_20140818_20170304_01_T1_B2.TIF','infred_new.tif',bb_cords, initial_perim_path)


new_tif_array = green_layer.read(1)
plt.matshow(new_tif_array)

# SAVE ALL CROPPED LAYERS TO EXPIRIMENT FOLDER TO BE LOADED INTO NEURAL NETWORK

np.save('landscape_king/burn_0.npy', state_initial)
np.save('landscape_king/burn_1.npy', state_final)
np.save('landscape_king/burn_2.npy', state_third)
np.save('landscape_king/topography.npy', elevation_final)
np.save('landscape_king/blue_layer.npy', blue_layer.read(1))
np.save('landscape_king/green_layer.npy', green_layer.read(1))
np.save('landscape_king/red_layer.npy',red_layer.read(1))
np.save('landscape_king/infred_layer.npy',infred_layer.read(1))






