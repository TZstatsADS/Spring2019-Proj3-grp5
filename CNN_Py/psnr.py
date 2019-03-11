#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:58:28 2019

@author: elena
"""

import numpy 
import math
import cv2


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0 #255.0 for normalized images
    return 20 * math.log10(PIXEL_MAX) - 10*math.log10(mse)


#test function
    
im_dir = os.path.join(os.getcwd(), "LR")
paths = glob.glob(os.path.join(im_dir, "*.jpg"))

img1 = cv2.imread(paths[0])
img2 = cv2.imread(paths[1])

PSNR = psnr(img1,img2)

print(PSNR)

