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
import numpy.random as rnd
import os
import h5py

# on importe une partie du jeu de données avec h5py
data_path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/'

h5f_1 = h5py.File(os.path.join(data_path,'data.h5'),'r')
train_data = h5f_1['data'][:]
h5f_1.close()

h5f_2 = h5py.File(os.path.join(data_path,'obj.h5'),'r')
train_obj = h5f_2['obj'][:]
h5f_2.close()

train_data, test_data, train_obj, test_obj = train_test_split(train_data,
                                                              train_obj, 
                                                              test_size=0.33)



m,n = np.shape(train_data)

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
    #pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=[2,2],
    #                                     stride=2, padding='same')
    # 2e couche
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5,5],
                             padding='same', activation=tf.nn.relu)
    #pool2 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[2,2],
    #                                     stride=4, padding='same')
    #print(pool2.shape)
    # dernière couche:
    # on applatit l'image:
    #conv2_flat = tf.reshape(pool2, [-1, 32**2 * 64])
    # on somme
    #logits = tf.layers.dense(inputs=conv2_flat, units=256**2)
    
    # sommation manuelle
    biais = tf.Variable(tf.zeros([1,256**2]))
    logits_im = tf.reduce_sum(conv2, 3)/64 
    print("im", logits_im.shape)
    logits = tf.reshape(logits_im, [-1, 256**2]) + biais
    
# hyperparamètres:
LEARNING_RATE = 0.05
n_epochs = 2
batch_size = 8
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)
    indices = rnd.randint(m, size=batch_size)
    X_batch = train_data[indices]
    y_batch = train_obj[indices]
    return X_batch, y_batch

summaries_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/summaries'
# opérations du modèle:
with graph1.as_default():
    # loss:
    print(np.shape(logits))
    loss = tf.losses.absolute_difference(labels=y_data, predictions=logits)
    
    # otimiser: Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    # opération d'entrainement:
    train_op = optimizer.minimize(loss)
    
    
    #init_op:
    init = tf.global_variables_initializer()
 
    tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()
    
    
# entrainement et test:
with tf.Session(graph=graph1) as sess: 
    # run initialisation:
    sess.run(init)
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    for epoch in range(n_epochs):
        
        # train op
        for batch_index in range(n_batches):
            X_batch, Y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(train_op, feed_dict={x_data:X_batch, y_data:Y_batch })
        # affichage fonction de perte
            loss_a, summary = sess.run([loss, merged],
                                       feed_dict={x_data:X_batch, y_data: Y_batch})
            train_writer.add_summary(summary, epoch * n_batches + batch_index)
        print('loss abs diff = ', loss_a)
            
    # phase de test:
    #pred_obj = sess.run(logits, feed_dict={x_data:test_data})
    
            
    
## OOM when allocating tensor with shape[4194304,65536]   
    