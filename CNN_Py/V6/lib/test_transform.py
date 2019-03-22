#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:07:47 2019

@author: elena
"""
import cv2
import os 
import glob
#import matplotlib.pyplot as plt
import numpy as np

def test_image_decomposition(image, input_size):
    dim = (image.shape[1]//input_size*input_size, image.shape[0]//input_size*input_size)
    cubic = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    cubic_sequence = []
    xn = 0
    xy = 0
    for x in range(0,cubic.shape[0],input_size):
        xn+=1
        for y in range(0,cubic.shape[1],input_size):
            patch = cubic[x : x + 16, y : y + 16]
            cubic_sequence.append(patch)
    xy = int(len(cubic_sequence)/xn)
    return cubic_sequence, xn, xy




def divide_chunks(decomposed_list, chunk_size): 
    for i in range(0, len(decomposed_list), chunk_size):  
        yield decomposed_list[i:i + chunk_size]
        
def text_image_composition(reconstructed_image_decomposed, chunk_size_xy):
    decomposed_chunks = list(divide_chunks(reconstructed_image_decomposed, chunk_size_xy)) 
    row_list = []
    for r in range(len(decomposed_chunks)):
        row = np.concatenate(decomposed_chunks[r], axis=1)
        row_list.append(row)
    whole_image = np.concatenate(row_list, axis=0)
    return whole_image


#c = test_image_decomposition(LR, input_size=16)
#c1 = text_image_composition(c[0], c[2])

#plt.imshow(c1)
#line 20, 22 deleted the following: -input_size
