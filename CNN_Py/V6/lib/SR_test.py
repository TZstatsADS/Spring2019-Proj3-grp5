#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:42:31 2019

@author: elena
"""
import cv2
import matplotlib.pyplot as plt
import psnr
import feature1_images_for_training as f1
import feature2_train_test_set as f2
import test_transform

def super_resolution(path_string_lr, path_string_truth, generator):
    
    pic = plt.imread(path_string_lr)
    pic_truth = plt.imread(path_string_truth)
    
    pic_norm = pic/255.0
    orig_shape = pic_norm.shape
    
    
    decomposed_picture = test_transform.test_image_decomposition(pic_norm, input_size=16)
    decomposed_picture_4dim = f2.ndarray_to_4dim(decomposed_picture[0], image_size = 16)
    
    pic_predicted = generator.predict(decomposed_picture_4dim)
    
    whole_predicted = test_transform.text_image_composition(pic_predicted, decomposed_picture[2])
    
    predicted_orig_shape = cv2.resize(whole_predicted, (orig_shape[1]*2, orig_shape[0]*2), interpolation = cv2.INTER_CUBIC)

    psnr_item = psnr.psnr(predicted_orig_shape, pic_truth/255.0)
    
    return psnr_item, pic, pic_truth, predicted_orig_shape