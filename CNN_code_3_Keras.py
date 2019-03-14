# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:58:14 2019

@author: Geoffroy Leconte

architecture avec Keras, couches successives de tailles qui diminuent
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from time import time
import h5py
import keras
from keras import backend as K 
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import add, concatenate
from keras.models import load_model, Model
from keras.losses import MSE
from keras.callbacks import TensorBoard

path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/'
#path = '/home/felixgontier/data/PROJET AUDIO/'
data_path = os.path.join(path,'quelques sons/')


h5f_train_data = h5py.File(os.path.join(data_path,'train_data.h5'),'r')
train_data = h5f_train_data['train_data'][:]
h5f_train_data.close()

h5f_test_data = h5py.File(os.path.join(data_path,'test_data.h5'),'r')
test_data = h5f_test_data['test_data'][:]
h5f_test_data.close()

h5f_train_obj = h5py.File(os.path.join(data_path,'train_obj.h5'),'r')
train_obj = h5f_train_obj['train_obj'][:]
h5f_train_obj.close()

h5f_test_obj = h5py.File(os.path.join(data_path,'test_obj.h5'),'r')
test_obj = h5f_test_obj['test_obj'][:]
h5f_test_obj.close()

h5f_train_phase_data = h5py.File(os.path.join(data_path,'train_phase_data.h5'),'r')
train_phase_data = h5f_train_phase_data['train_phase_data'][:]
h5f_train_phase_data.close()

h5f_test_phase_data = h5py.File(os.path.join(data_path,'test_phase_data.h5'),'r')
test_phase_data = h5f_test_phase_data['test_phase_data'][:]
h5f_test_phase_data.close()

h5f_train_phase_obj = h5py.File(os.path.join(data_path,'train_phase_obj.h5'),'r')
train_phase_obj = h5f_train_phase_obj['train_phase_obj'][:]
h5f_train_phase_obj.close()

h5f_test_phase_obj = h5py.File(os.path.join(data_path,'test_phase_obj.h5'),'r')
test_phase_obj = h5f_test_phase_obj['test_phase_obj'][:]
h5f_test_phase_obj.close()

m,n = np.shape(train_data)


# architecture
def cnn_model(width=256, height=256):
    k_size = (5,5)
    k_init = 'he_normal'
    # input layer:
    inp = Input(shape=(width, height,1))
    # pointeur pour les couches successives
    p = inp
    p = Conv2D(64, k_size, activation='relu', 
               kernel_initializer=k_init, padding='same')(p)
    p = Conv2D(32, k_size, activation='relu',
               kernel_initializer=k_init, padding='same')(p)
    outp = Conv2D(1, k_size, activation='relu',
                  kernel_initializer=k_init, padding='same')(p)
    model = Model(inputs=[inp], outputs=[outp])
    return model
            
# paramètres du modèle:
    
epochs=1
batch_size=5
learning_rate=0.001

adam_opt = keras.optimizers.Adam(lr=learning_rate)
optimizer = adam_opt
loss = MSE
metrics = [MSE]        

model = cnn_model()
model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


#tb_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/summaries/'
#tensorboard = TensorBoard(log_dir=os.path.join(tb_dir,'logs/{}') .format(time()),
#                          update_freq='batch')

train_data = np.reshape(train_data, (m, 256, 256,1))
train_obj = np.reshape(train_obj, (m, 256, 256,1))

m,n = np.shape(test_data)
test_data = np.reshape(test_data, (m, 256, 256,1))
test_obj = np.reshape(test_obj, (m, 256, 256,1))


model.fit(x=train_data, y=train_obj, 
          batch_size=batch_size, epochs=epochs)

model.evaluate(test_data, test_obj, batch_size=batch_size)

pred_obj = model.predict(test_data, batch_size=batch_size)
pred_obj = np.reshape(pred_obj, (m,256**2))

h5f_pred_obj = h5py.File(os.path.join(data_path, 'pred_obj.h5'), 'w')
h5f_pred_obj.create_dataset('pred_obj', data=pred_obj)
h5f_pred_obj.close()      