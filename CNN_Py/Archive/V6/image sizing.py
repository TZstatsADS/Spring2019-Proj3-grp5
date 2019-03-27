#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os 
import glob
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


os.getcwd()


# In[4]:


path = '/home/ed2801/train_img'


# In[5]:


os.chdir(path)


# In[6]:


im_dir = os.path.join(os.getcwd(), "LR")
paths = glob.glob(os.path.join(im_dir, "*.jpg"))
paths = list(paths)


# In[7]:


#paths = get_paths("LR")


# In[8]:


img = paths[0]


# In[9]:


img


# In[10]:


LR=cv2.imread(img)


# In[11]:


type(LR)


# In[12]:


LR.shape


# In[13]:


plt.imshow(LR)


# In[14]:


height = LR.shape[0]//16
height


# In[15]:


width = LR.shape[1]//16
width


# In[16]:


dim = (width*16, height*16)


# In[227]:


dim


# In[17]:


cubic = cv2.resize(LR, dim, interpolation = cv2.INTER_CUBIC)


# In[18]:


cubic.shape


# In[19]:


plt.imshow(cubic)


# In[20]:


image_size = 16
color_dim = 3
step = 1
padding = 0

LR_sequence = []
h, w, c = cubic.shape


# In[21]:


nx, ny = 0, 0
for x in range(0, h - image_size, image_size):
    nx += 1; ny = 0
    for y in range(0, w - image_size, image_size):
        ny += 1


# In[22]:


nx, ny #13-1, 17-1


# In[23]:


LR_sequence


# In[24]:


len(LR_sequence)


# In[25]:


12*16


# In[26]:


LR_sequence = []
nx, ny = 0,0
for x in range(2):
    x = nx
    ny = 0
    for y in range(2):
        y = ny
        iimg = cubic[x : x + 104, y : y + 136]
        nx =+ 104
        ny =+ 136
    #iimg = iimg/255.0
        LR_sequence.append(iimg)


# In[27]:


len(LR_sequence)


# In[28]:


type(LR_sequence)


# In[29]:


test = LR_sequence[0]


# In[30]:


type(test)


# In[31]:


test.shape


# In[32]:


test = LR_sequence[2]
plt.imshow(test)
#test


# In[33]:


test = LR_sequence[1]
plt.imshow(test)
#test


# In[34]:


test = LR_sequence[2]
plt.imshow(test)
#test


# In[42]:


v = np.concatenate(LR_sequence[0:2], axis=1)
v2 = np.concatenate(LR_sequence[2:4], axis=1)


# In[43]:


l = [v,v2]


# In[45]:


v3 = np.concatenate(l, axis=0)


# In[46]:


plt.imshow(v3)


# In[80]:


cubic.shape


# In[159]:


LR_sequence = []
xn = 0
xy = 0
for x in range(0,208,104):
    xn+=1
    for y in range(0,272,68):
        iimg = cubic[x : x + 104, y : y + 68]
        

        LR_sequence.append(iimg)


# In[160]:


len(LR_sequence)


# In[161]:


xn


# In[162]:


xy


# In[178]:


test = LR_sequence[7]
plt.imshow(test)


# In[195]:


def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]  
n = 4 
x = list(divide_chunks(LR_sequence, n)) 
len(x)


# In[197]:


row_list = []
for r in range(len(x)):
    row = np.concatenate(x[r], axis=1)
    row_list.append(row)


# In[198]:


len(row_list)


# In[199]:


plt.imshow(row_list[1])


# In[200]:


v3 = np.concatenate(row_list, axis=0)


# In[201]:


plt.imshow(v3)


# SO WE HAVE

# In[207]:


LR_sequence = []
xn = 0
xy = 0
for x in range(0,208-16,16):
    xn+=1
    for y in range(0,272-16,16):
        iimg = cubic[x : x + 16, y : y + 16]
        

        LR_sequence.append(iimg)


# In[209]:


len(LR_sequence)


# In[210]:


xn


# In[215]:


xy = int(len(LR_sequence)/xn)
xy


# In[219]:


test = LR_sequence[2]
plt.imshow(test)


# In[220]:



x = list(divide_chunks(LR_sequence, xy)) 
len(x)


# In[221]:


row_list = []
for r in range(len(x)):
    row = np.concatenate(x[r], axis=1)
    row_list.append(row)


# In[222]:


len(row_list)


# In[224]:


plt.imshow(row_list[0])


# In[225]:


v3 = np.concatenate(row_list, axis=0)


# In[226]:


plt.imshow(v3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


v1 = LR_sequence[0]
v2 = LR_sequence[1]

indexes = ((x1, w1), (x2, w2))
v = np.concatenate([img[y: y+h , v1: v1+v2] for v1,v2 in indexes], axis=1)


# In[48]:


v = img[y:y+h, list(range(x1, x1+w1)) + list(range(x2, x2 + w2))]


# In[39]:


from PIL import Image


def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths = 2
    heights = 2

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


# In[40]:




combo_1 = append_images(LR_sequence[0:2], direction='horizontal')


# In[ ]:


combo_2 = append_images(images, direction='horizontal', aligment='top',
                        bg_color=(220, 140, 60))
combo_3 = append_images([combo_1, combo_2], direction='vertical')
combo_3.save('combo_3.png')


# In[ ]:



from PIL import Image

def crop(path, input, height, width, k, page, area):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                o = a.crop(area)
                o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
            except:
                pass
            k +=1


# In[ ]:


from PIL import Image

def crop(path, input, height, width, k, page, area):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                o = a.crop(area)
                o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
            except:
                pass
            k +=1


# In[ ]:


image = Image.open(img)
type(image)
sequence = []
boxes = []
for x0 in range(0,272,68):
   for y0 in range(0,272,68):
      box = (x0, y0, x0+68, y0+104)
      boxes.append(box)
for l in range(len(boxes)):
    cropped_image = cubic.crop(box)
    sequence.append(cropped_image)
    
boxes

