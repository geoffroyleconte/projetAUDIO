# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:58:14 2019

@author: Geoffroy Leconte
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# on importe une partie du jeu de données avec pandas
train_data = pd.read_csv('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/train_data.csv')
train_obj = pd.read_csv('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/train_obj.csv')
train_data = train_data.values # spectrogramme bf en ligne pour l'entrainement
train_obj = train_obj.values # valeurs hf objet pour l'entrainement

# données test ...


# architecture
graph1 = tf.Graph()
with graph1.as_default():
    # entrées et sorties avec des placeholders
    x_data = tf.placeholder(tf.float32, shape=(None, 256**2))
    y_data = tf.placeholder(tf.float32, shape=(None, 256**2))
    x_im_data = tf.reshape(x_data, shape=(-1, 256, 256, 1))
    # peut-être inutile:
    y_im_data = tf.reshape(x_data, shape=(-1, 256, 256, 1))
    
    # on teste une convolution 5*5 activation relu
    conv1 = tf.layers.conv2d(inputs=x_im_data, filters=64, kernel_size=[5,5],
                            padding='same', activation=tf.nn.relu)
    # pooling
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    
    # 2e couche
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5],
                             padding='same', activation=tf.nn.relu)
    
    
    