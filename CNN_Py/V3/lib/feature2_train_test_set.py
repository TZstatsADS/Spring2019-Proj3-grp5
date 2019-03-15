#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:19:39 2019

@author: elena
"""

# Train_test data 

def hr_images(images):
    images_hr = array(images)
    return images_hr

def lr_images(images_to_downscale, n_scale):
    
    images = []
    for i in  range(len(images_to_downscale)):
        images.append(cv2.resize(images_to_downscale[i], (images_to_downscale[i].shape[0]//n_scale, images_to_downscale[i].shape[1]//n_scale), interpolation = cv2.INTER_LINEAR))
    images_lr = array(images)
    return images_lr


def train_test_data(imgs, number_of_images = 2200, train_test_ratio = 0.9):

    number_of_train_images = int(number_of_images * train_test_ratio)
    
    x_train = imgs[:number_of_train_images]
    x_test = imgs[number_of_train_images:number_of_images]
    
    x_train_hr = hr_images(x_train)
    
    x_test_hr = hr_images(x_test)
    x_test_lr = lr_images(x_test, 4)
    
    x_train_lr = lr_images(x_train, 4)
    
    return x_train_lr, x_train_hr, x_test_lr, x_test_hr
