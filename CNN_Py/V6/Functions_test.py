#!/usr/bin/env python
# coding: utf-8


import cv2
import os 
import glob
import matplotlib.pyplot as plt
import numpy as np


#path to test images folder
#path = '/home/ed2801/train_img'
path = '/Users/elena/Desktop/AppDS/Project_3_SR/Spyder_local_project'

# In[4]:


os.chdir(path)


# In[18]:


input_size = 16


# In[10]:


im_dir = os.path.join(os.getcwd(), "LR")
paths = glob.glob(os.path.join(im_dir, "*.jpg"))
paths = list(paths)


# In[77]:


paths[444]


# In[78]:


LR = cv2.imread(paths[444])


# In[154]:


plt.imshow(LR)


# In[155]:


LR.shape


# In[156]:


#height = LR.shape[0]//input_size
#width = LR.shape[1]//input_size
dim = ((LR.shape[1]//input_size)*input_size, (LR.shape[0]//input_size)*input_size)


# In[147]:


dim


# In[157]:


cubic = cv2.resize(LR, dim, interpolation = cv2.INTER_CUBIC)


# In[158]:


LR_sequence = []
xn = 0
for x in range(0,cubic.shape[1]-input_size,input_size):
    xn+=1
    for y in range(0,cubic.shape[0]-input_size,input_size):
        patch = cubic[x : x + 16, y : y + 16]
        LR_sequence.append(patch)


# In[159]:


xy = int(len(LR_sequence)/xn)


# In[164]:


def test_image_decomposition(image, input_size):
    dim = ((image.shape[1]//input_size)*input_size, (image.shape[0]//input_size)*input_size)
    cubic = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    cubic_sequence = []
    xn = 0
    for x in range(0,cubic.shape[1]-input_size,input_size):
        xn+=1
        for y in range(0,cubic.shape[0]-input_size,input_size):
            patch = cubic[x : x + 16, y : y + 16]
            cubic_sequence.append(patch)
            xy = int(len(cubic_sequence)/xn)
    return cubic_sequence, xn, xy


# In[165]:


test = test_image_decomposition(LR, 16)


# In[210]:


test[2]


# In[ ]:


plt.imshow(test[0][0])


# In[133]:


#transform the sequence to 4d array, feed to generator.predict to get 32*32


# In[168]:


def divide_chunks(decomposed_list, n_chunks): 
    for i in range(0, len(decomposed_list), n_chunks):  
        yield decomposed_list[i:i + n_chunks]


# In[213]:


decomposed_chunks = list(divide_chunks(test[0], test[1])) 


# In[214]:


len(decomposed_chunks)


# In[216]:


len(decomposed_chunks[1])


# In[217]:


row_list = []
for r in range(len(decomposed_chunks)):
    row = np.concatenate(decomposed_chunks[r], axis=1)
    row_list.append(row)


# In[202]:


row_list[9]


# In[208]:


plt.imshow(row_list[5])


# In[171]:


whole_image = np.concatenate(row_list, axis=0)


# In[179]:


def text_image_composition(reconstructed_image_decomposed, xy):
    decomposed_chunks = list(divide_chunks(reconstructed_image_decomposed, xy)) 
    row_list = []
    for r in range(len(decomposed_chunks)):
        row = np.concatenate(decomposed_chunks[r], axis=1)
        row_list.append(row)
    whole_image = np.concatenate(row_list, axis=0)
    return whole_image


# In[181]:


test1 = text_image_composition(test[0], test[2])


# In[182]:


plt.imshow(test1)


# In[175]:


dim_orig = (LR.shape[1], LR.shape[0])
output = cv2.resize(test1, dim_orig, interpolation = cv2.INTER_CUBIC)


# In[176]:


plt.imshow(output)


# In[153]:


output.shape


# In[ ]:




