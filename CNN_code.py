# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:58:14 2019

@author: Geoffroy Leconte
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# on importe une partie du jeu de données avec pandas
data_d = pd.read_csv('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/train_data.csv')
obj_d = pd.read_csv('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/train_obj.csv')
data = data_d.values # spectrogramme bf en ligne pour l'entrainement
data = data[:,1:] # on enlève la premère colonne rajoutée par le format csv
obj = obj_d.values # valeurs hf objet pour l'entrainement
obj = obj[:,1:]
train_data, test_data, train_obj, test_obj = train_test_split(data,
                                                              obj, 
                                                              test_size=0.33)

# données test ...


# architecture
graph1 = tf.Graph()
with graph1.as_default():
    # entrées et sorties avec des placeholders
    x_data = tf.placeholder(tf.float32, shape=(None, 256**2))
    y_data = tf.placeholder(tf.float32, shape=(None, 256**2))
    x_im_data = tf.reshape(x_data, shape=(-1, 256, 256, 1))
    # peut-être inutile:
    y_im_data = tf.reshape(y_data, shape=(-1, 256, 256, 1))
    
    # on teste une convolution 5*5 activation relu
    conv1 = tf.layers.conv2d(inputs=x_im_data, filters=64, kernel_size=[5,5],
                            padding='same', activation=tf.nn.relu)
    
    # 2e couche
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5,5],
                             padding='same', activation=tf.nn.relu)
    
    # dernière couche:
    # on applatit l'image:
    conv2_flat = tf.reshape(conv2, [-1, 256**2 * 64])
    # on somme
    logits = tf.layers.dense(inputs=conv2_flat, units=256**2)
    
# hyperparamètres:
LEARNING_RATE = 0.05
TRAIN_STEPS = 300
DISPLAY_STEPS = 20

# opérations du modèle:
with graph1.as_default():
    #init_op:
    init = tf.global_variables_initializer()
    # loss: entropie croisée:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_data, 
                                                                     logits=logits))
    # otimiser: descente de gradient
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    # opération d'entrainement:
    train_op = optimizer.minimize(loss)
    
    
# entrainement et test:
with tf.Session(graph=graph1) as sess: 
    # run initialisation:
    sess.run(init)
    for i in range(TRAIN_STEPS+1):
        # train op
        sess.run(train_op, feed_dict={x_data:train_data, y_data: train_obj})
        # affichage fonction de perte
        if i%DISPLAY_STEPS == 0:
            print('loss entropy = ', 
                  sess.run(loss,
                           feed_dict={x_data:train_data, y_data: train_obj}))
    # phase de test:
    pred_obj = sess.run(logits, feed_dict={x_data:test_data})
    
            
    
## OOM when allocating tensor with shape[4194304,65536]   
    