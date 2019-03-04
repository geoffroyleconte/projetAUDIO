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
import fonctions_utilitaires as f_uti
import librosa
# on importe une partie du jeu de données avec h5py

path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/'
#path = '/home/felixgontier/data/PROJET AUDIO/'
data_path = os.path.join(path,'quelques sons/')


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

# nombre de lignes pour les bf/hf 
nb_bf = 64
nb_hf = 512-64

# architecture
graph1 = tf.Graph()
with graph1.as_default():
    # entrées et sorties avec des placeholders
    x_data = tf.placeholder(tf.float32, shape=(None, 256*nb_bf))
    y_data = tf.placeholder(tf.float32, shape=(None, 256*nb_hf))
    x_im_data = tf.reshape(x_data, shape=(-1, nb_bf, 256, 1))
    x_im_data_conc = tf.concat([x_im_data]*7, axis=1)
    # peut-être inutile:
    y_im_data = tf.reshape(y_data, shape=(-1, nb_hf, 256, 1))
    # on teste une convolution 5*5 activation relu
    conv1 = tf.layers.conv2d(inputs=x_im_data_conc, filters=64, kernel_size=[5,5],
                            padding='same', activation=tf.nn.relu)


    #pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=[2,2],
    #                                     stride=2, padding='same')
    # 2e couche
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5,5],
                             padding='same', activation=tf.nn.relu)
    #pool2 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[2,2],
    #                                     stride=4, padding='same')
    print(conv2.shape)
    # sommation manuelle
    biais = tf.Variable(tf.zeros([1,256*nb_hf]))
    logits_im = tf.reduce_sum(conv2, 3)/64 
    print("im", logits_im.shape)
    logits = tf.reshape(logits_im, [-1, 256*nb_hf]) + biais
    
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


summaries_dir = os.path.join(path, 'summaries/')

# opérations du modèle:
with graph1.as_default():
    # loss:
    print(np.shape(logits))
    loss = tf.losses.mean_squared_error(labels=y_data, predictions=logits)
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
            print('epoch',epoch+1,'/', n_epochs, '    batch', 
                  batch_index+1, '/', n_batches,
                  '    loss abs diff = ', loss_a)
            
    # phase de test:
    pred_obj = sess.run(logits, feed_dict={x_data:test_data})
    
            
    
## OOM when allocating tensor with shape[4194304,65536]  
# tests d'écoute    
    
spec_high = np.reshape(pred_obj[3,:], (nb_hf,256))
spec_low = np.reshape(test_data[3,:], (nb_bf,256))
spec = np.zeros((513,256))
spec[0:nb_bf,:] = spec_low
sig_low = librosa.istft(spec)
sig_low_gl = f_uti.reconstruct_sig_griffin_lim(spec,len(sig_low), 100, 1024, 256)
# son avec seulement les bf
path_out = os.path.join(path, 'out_sounds/')
path_bf = os.path.join(path_out, 'test_sound_low_f1.wav')
librosa.output.write_wav(path_bf, sig_low_gl, sr)
                         
spec = np.zeros((513,256))
spec[0:nb_bf,:] = spec_low                         
spec[nb_bf:512,:] = spec_high
sig = librosa.istft(spec)
sr=16000
sig_gl = f_uti.reconstruct_sig_griffin_lim(spec,len(sig), 100, 1024, 256)
# son reconstruit
path_recons = os.path.join(path_out, 'test_sound1.wav')
librosa.output.write_wav(path_recons, sig_gl, sr)
                         
spec_high_gt = np.reshape(test_obj[3,:], (nb_hf,256))
spec_gt = np.zeros((513,256))
spec_gt[0:nb_bf,:] = spec_low 
spec_gt[nb_bf:512,:] = spec_high_gt
sig_gl_gt = f_uti.reconstruct_sig_griffin_lim(spec_gt,len(sig), 100, 1024, 256) 
path_gt = os.path.join(path_out, 'test_sound_gt1.wav')                        
librosa.output.write_wav(path_gt, sig_gl_gt, sr)

snr1 = f_uti.snr2(sig_gl_gt, sig_gl)