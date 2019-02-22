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
audio_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/LibriSpeech'

l_audio_dir = []
l_audio_files = []
l_audio_sr = []
l_stft = []
# données d'entrainement: (partie basse et partie haute)
train_data = np.zeros((1,256**2))
train_obj = np.zeros((1,256**2))
c = 0
# ouverture
for root, dirname, filenames in os.walk(audio_dir):
    for filename1 in filenames:
        audio_file = os.path.join(root, filename1)
        # enlever la condition sur le compteur c pour faire sur tous les
        # fichiers audio
        if audio_file.endswith('.flac') and c<30:
            c+=1
            l_audio_dir.append(audio_file)
            yi, sri = sf.read(audio_file)
            #l_audio_files.append(yi)
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
            for i in range(0, n-256, 256):
                mag_lowi = mag_low[:, i:i+256]
                train_data = np.append(train_data, np.array([mag_lowi.flatten()]),
                                       axis=0)
                mag_highi = mag_high[:, i:i+256]
                train_obj = np.append(train_obj, np.array([mag_highi.flatten()]),
                                      axis=0)
# données d'entrainement dans un fichier csv pour les ouvrir rapidement  
train_data = train_data[1:,:] # on enlève la première ligne qui était
# seulement pour pouvoir ajouter facilement des éléments dans le tableau np.
train_obj = train_obj[1:,:]

store_path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/'

h5f_1 = h5py.File(os.path.join(store_path,'data.h5'), 'w')
h5f_1.create_dataset('data', data=train_data)
h5f_1.close()

h5f_2 = h5py.File(os.path.join(store_path, 'obj.h5'), 'w')
h5f_2.create_dataset('obj', data=train_obj)
h5f_2.close()


#td = pd.DataFrame(train_data)  
#tobj = pd.DataFrame(train_obj) 
#td.to_csv('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/train_data.csv')
#tobj.to_csv('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/train_obj.csv')      
            
            
            
                