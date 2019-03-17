#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:11:53 2019

@author: elena

Contains ALL COMPUTATIONAL STEPS

"""
#### LIBRARIES

# Utils

import time

start_1 = time.time()

import os
import glob
import numpy as np
from numpy import array
import random
import math

# Images resizing

import cv2

# CNN

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU
from tensorflow.python.keras.layers import add
from tensorflow.python.keras.optimizers import Adam

# Visualization

import matplotlib.pyplot as plt

end_1 = time.time()

#### STEP 0: DIRECTORIES SPECIFIED

random.seed(2019)
os.getcwd()
path = "D:\\training_images\\"
os.chdir(path)

#### STEP 1: CONTROLS FOR EVALUATION EXPERIMENTS

RUN_CV = True
K_FOLDS = 5
RUN_FEATURE_TRAIN = True
RUN_TEST = True
RUN_FEATURE_TEST = True

#### STEP 2: IMPORTING LABELS (omitted, we don't use labels)

#### STEP 3: FEATURES AND IMAGE TRANSFORMATIONS 
start_2 = time.time()
 
hr_paths = get_paths("HR")
HR_cropped = cropped_imgs(hr_paths)
x_train_lr, x_train_hr, x_test_lr, x_test_hr = train_test_data(HR_croppedS)

x_train_lr = ndarray_to_4dim(x_train_lr, image_size = 8)
x_train_hr = ndarray_to_4dim(x_train_hr, image_size = 32)
x_test_lr = ndarray_to_4dim(x_test_lr, image_size = 8)
x_test_hr = ndarray_to_4dim(x_test_hr, image_size = 32)

end_2 = time.time()
#ex = x_train_hr[0]
#### STEP 4a: TRAIN MODEL
start_3 = time.time()

epochs = 5
batch_size = 10

generator, generated_train, loss, val_loss = train(epochs = epochs, batch_size = batch_size)

end_3 = time.time()

image_shape = (8,8,3)
generator = Generator(image_shape).generator()
adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
generator.compile(loss='mean_squared_error', optimizer=adam)
generator.load_weights("D:\\models\\model50.h5")

#### STEP 4b: CROSS-VALIDATION AND PARAMETER TUNING (performance results visualized, best model chosed)
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r+', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#### STEP 5: TEST ON HOLDOUT SET

start_4 = time.time()



test_im = generator.evaluate(x_test_lr, x_test_hr)

type(test_im)

plt.figure()
plt.plot(test_im, 'bo', label='Trest loss')
plt.title('Test loss')
plt.legend()
plt.show()

gen_imgs = generator.predict(x_test_lr)


type(gen_imgs)
gen_img = gen_imgs[20]
result = plt.imshow(gen_img, interpolation='nearest')

plt.imshow(x_test_hr[20], interpolation='nearest')
plt.imshow(x_test_lr[20], interpolation='nearest')

psnr_list = []
for i in range(len(gen_imgs)):
    psnr_item = psnr(gen_imgs[i], x_test_hr[i])
    psnr_list.append(psnr_item)
    
plt.figure()
plt.plot(psnr_list, 'bo', label='Test(PSNR)')
plt.title('Test(PSNR)')
plt.legend()
plt.show()

np.mean(psnr_list)
    
#psnr_1 = psnr(gen_imgs[0], x_test_hr[0])

end_4 = time.time()

#### SUMMARISE RUNNING TIME

lib_download_time = end_1 - start_1
data_prep_time = end_2 - start_2
train_time = end_3 - start_3
test_time = end_4 - start_4

print("Loading libraries takes: ", lib_download_time)
print("Preparing data for training the model takes: ", data_prep_time)
print("Training time: ", train_time)
print("Testing time: ", test_time)
