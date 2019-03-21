#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:43:46 2019

@author: elena
"""


LR=cv2.imread('/Users/elena/Desktop/AppDS/Project_3_SR/Spider_local_project/LR/img_1279.jpg')

type(LR)


LR.shape
plt.imshow(LR)

dim = (LR.shape[1]//16*16, LR.shape[0]//16*16)

cubic = cv2.resize(LR, dim, interpolation = cv2.INTER_CUBIC)
cubic.shape
LR_sequence = []
xn = 0
xy = 0
for x in range(0,cubic.shape[0]-16,16):
    xn+=1
    for y in range(0,cubic.shape[1]-16,16):
        iimg = cubic[x : x + 16, y : y + 16]
        LR_sequence.append(iimg)

xy = int(len(LR_sequence)/xn)
chunks = list(divide_chunks(LR_sequence, xy)) 

row_list = []
for r in range(len(chunks)):
    row = np.concatenate(chunks[r], axis=1)
    row_list.append(row)

plt.imshow(row_list[0])
v3 = np.concatenate(row_list, axis=0)
plt.imshow(v3)
_____