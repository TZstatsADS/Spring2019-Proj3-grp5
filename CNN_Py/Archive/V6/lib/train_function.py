#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:23:43 2019

@author: elena
"""
from tensorflow.python.keras.models import load_model
import model_CNN
import csv
from tensorflow.python.keras.optimizers import Adam

def train(x_train_lr, x_train_hr, epochs, batch_size):
    
    image_shape = (16,16,3)
    downscale_factor = 2
    batch_count = int(x_train_hr.shape[0] / batch_size)
    #shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    
    generator = model_CNN.Generator(image_shape).generator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    generator.compile(loss='mean_squared_error', optimizer=adam)
    generator.summary()
    generated_train = generator.fit(x_train_lr, x_train_hr, batch_size = batch_size, validation_split = 0.2, epochs = epochs)
    loss = generated_train.history['loss']
    val_loss = generated_train.history['val_loss']
    
    generator.save('/content/drive/My Drive/SR_images/model.h5')
    generator.save_weights('/content/drive/My Drive/SR_images/weights_model.h5')

    return generator, generated_train, loss, val_loss

#with open('/home/ed2801/Spring2019-Proj3-spring2019-proj3-grp5/CNN_Py/V3/doc/loss.csv', 'w') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#    wr.writerow(loss)

#with open('/home/ed2801/Spring2019-Proj3-spring2019-proj3-grp5/CNN_Py/V3/doc/loss_val.csv', 'w') as myfile:
#    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#   wr.writerow(val_loss)    

#generator.save("D:\\models\\model%d.h5" %epochs)
#generator.save_weights("D:\\models\\weights_model%d.h5" %epochs)

#generator.save("V3/output/model%d.h5" %epochs)
#generator.save_weights("V3/output/weights_model%d.h5" %epochs)
        
