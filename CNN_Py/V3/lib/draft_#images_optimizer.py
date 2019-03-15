# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:30:03 2019

@author: ed2801
"""

number_of_train_images = int(2200 * 0.9)
    
x_train = HR_cropped[:number_of_train_images]
x_test = HR_cropped[number_of_train_images:2200]

x_train_hr = hr_images(x_train)
x_train_lr = lr_images(x_train_1, 4)

if type(x_train_hr) == 'object':
    x_train_hr_1 = []
    for i in range(len(x_train_hr)):
        x_train_hr[i] = img
        x_train_hr_1.append(img)
else:
    x_train_hr = x_train_hr

x_test_hr = hr_images(x_test)
x_test_lr = lr_images(x_test, 4)
