#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:57:08 2019

@author: elena
"""
import os
import glob
import cv2
import numpy as np
from numpy import array

def reshape(string_dir_extention): #"LR" as string
    
    image_size = 32
    color_dim = 3
    step = 21
    im_dir = os.path.join(os.getcwd(), string_dir_extention)
    paths = glob.glob(os.path.join(im_dir, "*.jpg"))
    
    reshaped_sequence = []
    
    for i in range(len(paths)):
        
        im = cv2.imread(paths[i])
        im_resized = cv2.resize(im, None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
        h, w, c = im_resized.shape
        
        nx, ny = 0, 0
        for x in range(0, h - image_size + 1, step):
            nx += 1; ny = 0
            for y in range(0, w - image_size + 1, step):
                ny += 1

        im_resized = im_resized[x: x + image_size, y: y + image_size] 
        im_reshaped = im_resized.reshape([image_size, image_size, color_dim])
        #Normalize
        im_reshaped =  im_reshaped / 255.0
        
        reshaped_sequence.append(im_reshaped)
        
    return reshaped_sequence

def hr_images(images):
    images_hr = array(images)
    return images_hr

def lr_images(images_to_downscale, n_scale):
    
    images = []
    for i in  range(len(images_to_downscale)):
        images.append(cv2.resize(images_to_downscale[i], (images_to_downscale[i].shape[0]//n_scale, images_to_downscale[i].shape[1]//n_scale), interpolation = cv2.INTER_LINEAR))
    images_lr = array(images)
    return images_lr

def train_test_data(imgs, number_of_images = 1470, train_test_ratio = 0.9):

    number_of_train_images = int(number_of_images * train_test_ratio)
        
    x_train = imgs[:number_of_train_images]
    x_test = imgs[number_of_train_images:number_of_images]
    
    x_train_hr = hr_images(x_train)
    x_test_hr = hr_images(x_test)
    
    x_train_lr = lr_images(x_train, 4)
    x_test_lr = lr_images(x_test, 4)
    
    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


