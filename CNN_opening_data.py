# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:14:36 2019

@author: Geoffroy Leconte
"""

#import tensorflow as tf
#import fonctions_utilitaires as f_uti
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import librosa
import pandas as pd
import h5py
    
# train_data:
#audio_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/LibriSpeech'
audio_dir = '/home/felixgontier/data/PROJET AUDIO/LibriSpeech'



l_audio_sr = [] # regarder si les fréquences d'échantillonnage sont identiques


# données d'entrainement: (partie basse et partie haute)
train_data = [[0]*(256**2)]
train_obj = [[0]*(256**2)]
c = 0 # compteur pour le nombre max de fichiers à traiter.


# ouverture
for root, dirname, filenames in os.walk(audio_dir):
    for filename1 in filenames:
        audio_file = os.path.join(root, filename1)
        # enlever la condition sur le compteur c pour faire sur tous les
        # fichiers audio
        if audio_file.endswith('.flac') and c<2000:
            c+=1
            y_f, sr_f = sf.read(audio_file)
            sri = 5000
            yi = librosa.resample(y_f, sr_f, sri)
            l_audio_sr.append(sri)
            # stft
            # regarder pour l'overlap 1/4
            Di = librosa.stft(yi, n_fft=1024)
            m,n = np.shape(Di)
            # amplitude bf
            mag_low = np.abs(Di[0:256, :])
            # amplitude hf, on chosiit de ne pas s'occuper de la dernière ligne
            mag_high = np.abs(Di[256:512,:])
            # découpage des signaux (256 trames)
            
            #### on découpe en tranches de taille 256, on rallonge si signal 
            # trop court
            
            i=0
            while i<n:
                if n-i<256:
                    mag_lowi = np.zeros((256,256))
                    mag_lowi[0:256,0:(n-i)] = mag_low[0:256,i:n]
                    mag_highi = np.zeros((256,256))
                    mag_highi[0:256,0:(n-i)] = mag_high[0:256,i:n]
                else:
                    mag_lowi = mag_low[:, i:i+256]
                    mag_highi = mag_high[:, i:i+256]
                train_data.append( np.array(mag_lowi.flatten()))
                train_obj.append(np.array(mag_highi.flatten()))
                i+=256
            if c%10==0:
                print('iter', c)
            
# données d'entrainement dans un fichier h5 pour les ouvrir rapidement  
train_data = train_data[1:][:] # on enlève la première ligne qui était
# seulement pour pouvoir ajouter facilement des éléments dans le tableau np.
train_obj = train_obj[1:][:]

#store_path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/'
store_path = '/home/felixgontier/data/PROJET AUDIO/quelques sons/'

h5f_1 = h5py.File(os.path.join(store_path,'data.h5'), 'w')
h5f_1.create_dataset('data', data=train_data)
h5f_1.close()

h5f_2 = h5py.File(os.path.join(store_path, 'obj.h5'), 'w')
h5f_2.create_dataset('obj', data=train_obj)
h5f_2.close()

      
            
            
            
                