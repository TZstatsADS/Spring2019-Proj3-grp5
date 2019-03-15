#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:11:53 2019

@author: elena

Contains ALL COMPUTATIONAL STEPS

"""
#### LIBRARIES

# Utils

import os
import glob
import numpy as np
from numpy import array
import random
import math

# Images resizing

import cv2

# CNN

from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add
from keras.optimizers import Adam

# Visualization

import matplotlib.pyplot as plt


#### STEP 0: DIRECTORIES SPECIFIED

random.seed(2019)
os.getcwd()
#path = "~/doc"
#os.chdir(path)

#### STEP 1: CONTROLS FOR EVALUATION EXPERIMENTS

RUN_CV = True
K_FOLDS = 5
RUN_FEATURE_TRAIN = True
RUN_TEST = True
RUN_FEATURE_TEST = True

#### STEP 2: IMPORTING LABELS (omitted, we don't use labels)

#### STEP 3: FEATURES AND IMAGE TRANSFORMATIONS 

hr_paths = get_paths("HR")
HR_cropped = cropped_imgs(hr_paths)
x_train_lr, x_train_hr, x_test_lr, x_test_hr = train_test_data(HR_cropped)

#### STEP 4a: TRAIN MODEL

train(epochs = 1, batch_size = 10)


#### STEP 4b: CROSS-VALIDATION AND PARAMETER TUNING (performance results visualized, best model chosed)

plt.imshow(x_train_lr[0], interpolation='nearest')
plt.imshow(x_train_hr[0], interpolation='nearest')

#### STEP 5: TEST ON HOLDOUT SET

#### SUMMARISE RUNNING TIME


