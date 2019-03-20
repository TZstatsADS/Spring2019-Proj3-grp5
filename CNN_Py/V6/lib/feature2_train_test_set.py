#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:19:39 2019

@author: elena
"""
import numpy as np
import cv2
from numpy import array 
# Train_test data 

def hr_images(images):
    images_hr = images
    return images_hr

def lr_images(images_to_downscale, n_scale):
    
    images = []
    for i in  range(len(images_to_downscale)):
        images.append(cv2.resize(images_to_downscale[i], (images_to_downscale[i].shape[0]//n_scale, images_to_downscale[i].shape[1]//n_scale), interpolation = cv2.INTER_LINEAR))
    images_lr = images
    return images_lr


def train_test_data(imgs, number_of_images = 2200, train_test_ratio = 0.9):

    number_of_train_images = int(number_of_images * train_test_ratio)
    
    x_train = imgs[:number_of_train_images]
    x_test = imgs[number_of_train_images:number_of_images]
    
    x_train_hr = hr_images(x_train)
    
    x_test_hr = hr_images(x_test)
    x_test_lr = lr_images(x_test, 2)
    
    x_train_lr = lr_images(x_train, 2)
    
    return x_train_lr, x_train_hr, x_test_lr, x_test_hr

def ndarray_to_4dim(ndarray, image_size = 16):
    four_dim = np.empty([len(ndarray), image_size, image_size, 3])
    for i in range(len(ndarray)):
        if ndarray[i].shape == (image_size,image_size,3):
            four_dim[i] = ndarray[i]
        
    return four_dim
