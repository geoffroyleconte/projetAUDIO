# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:10:34 2019

@author: Geoffroy Leconte
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import librosa
import pandas as pd
import h5py
from random import shuffle

# train_data:
audio_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/LibriSpeech/dev-clean/84'
#audio_dir = '/home/felixgontier/data/PROJET AUDIO/LibriSpeech/dev-clean/84'

#67-2 fichiers audio en théorie


l_audio_sr = [] # regarder si les fréquences d'échantillonnage sont identiques


# données d'entrainement: (partie basse et partie haute)
data = [[0]*(256**2)]
obj = [[0]*(256**2)]
phase_data = [[0]*(256**2)]
phase_obj = [[0]*(256**2)]
c = 0 # compteur pour le nombre max de fichiers à traiter.
nb_files = 30 # nombre de fichiers traités

# ouverture
for root, dirname, filenames in os.walk(audio_dir):
    for filename1 in filenames:
        audio_file = os.path.join(root, filename1)
        # enlever la condition sur le compteur c pour faire sur tous les
        # fichiers audio
        if audio_file.endswith('.flac') and c<nb_files:
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
            phase_low = np.angle(Di[0:256,:])
            phase_high = np.angle(Di[256:512,:])
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
                    phase_lowi = np.zeros((256,256))
                    phase_lowi[0:256,0:(n-i)] = phase_low[0:256,i:n]
                    phase_highi = np.zeros((256,256))
                    phase_highi[0:256,0:(n-i)] = phase_high[0:256,i:n]
                    
                else:
                    mag_lowi = mag_low[:, i:i+256]
                    mag_highi = mag_high[:, i:i+256]
                    phase_lowi = phase_low[:,i:i+256]
                    phase_highi = phase_high[:,i:i+256]
                data.append( np.array(mag_lowi.flatten()))
                obj.append(np.array(mag_highi.flatten()))
                phase_data.append(np.array(phase_lowi.flatten()))
                phase_obj.append(np.array(phase_highi.flatten()))
                i+=256
            if c%10==0:
                print('iter', c)
            
# données d'entrainement dans un fichier h5 pour les ouvrir rapidement  
data = np.array(data[1:][:]) # on enlève la première ligne qui était
# seulement pour pouvoir ajouter facilement des éléments dans le tableau np.
obj = np.array(obj[1:][:])
phase_data = np.array(phase_data[1:][:])
phase_obj = np.array(phase_obj[1:][:])

# on permute les éléments au hasard
index_spec = [i for i in range(len(data))]
shuffle(index_spec)
data_shuffle = data[index_spec][:]
obj_shuffle = obj[index_spec][:]
phase_data_shuffle = phase_data[index_spec][:]
phase_obj_shuffle = phase_obj[index_spec][:]

test_ratio = 0.1
# séparation données entrainement et test
i_sep = int(nb_files*(1-test_ratio))
train_data = data_shuffle[:i_sep][:]
test_data = data_shuffle[i_sep:][:]
train_obj = obj_shuffle[:i_sep][:]
test_obj = obj_shuffle[i_sep:][:]
train_phase_data = phase_data_shuffle[:i_sep][:]
test_phase_data = phase_data_shuffle[i_sep:][:]
train_phase_obj = phase_obj_shuffle[:i_sep][:]
test_phase_obj = phase_obj_shuffle[i_sep:][:]

store_path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/one_speaker'
#store_path = '/home/felixgontier/data/PROJET AUDIO/quelques sons/one_speaker'

h5f_train_data = h5py.File(os.path.join(store_path,'train_data.h5'), 'w')
h5f_train_data.create_dataset('train_data', data=train_data)
h5f_train_data.close()

h5f_test_data = h5py.File(os.path.join(store_path,'test_data.h5'), 'w')
h5f_test_data.create_dataset('test_data', data=test_data)
h5f_test_data.close()

h5f_train_obj = h5py.File(os.path.join(store_path, 'train_obj.h5'), 'w')
h5f_train_obj.create_dataset('train_obj', data=train_obj)
h5f_train_obj.close()

h5f_test_obj = h5py.File(os.path.join(store_path, 'test_obj.h5'), 'w')
h5f_test_obj.create_dataset('test_obj', data=test_obj)
h5f_test_obj.close()

h5f_train_phase_data = h5py.File(os.path.join(store_path, 'train_phase_data.h5'), 'w')
h5f_train_phase_data.create_dataset('train_phase_data', data=train_phase_data)
h5f_train_phase_data.close()

h5f_test_phase_data = h5py.File(os.path.join(store_path, 'test_phase_data.h5'), 'w')
h5f_test_phase_data.create_dataset('test_phase_data', data=test_phase_data)
h5f_test_phase_data.close()
      
h5f_train_phase_obj = h5py.File(os.path.join(store_path, 'train_phase_obj.h5'), 'w')
h5f_train_phase_obj.create_dataset('train_phase_obj', data=train_phase_obj)
h5f_train_phase_obj.close()

h5f_test_phase_obj = h5py.File(os.path.join(store_path, 'test_phase_obj.h5'), 'w')
h5f_test_phase_obj.create_dataset('test_phase_obj', data=test_phase_obj)
h5f_test_phase_obj.close()           
            