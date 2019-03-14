# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:16:54 2019

@author: Geoffroy Leconte
"""
import h5py
import numpy as np
import os
import librosa
import fonctions_utilitaires as f_uti

#path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/'
path = '/home/felixgontier/data/PROJET AUDIO/'
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

h5f_pred_obj = h5py.File(os.path.join(data_path,'pred_obj.h5'),'r')
pred_obj = h5f_pred_obj['pred_obj'][:]
h5f_pred_obj.close()

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

def write_out_sounds(mod_low,mod_high,phase_low,phase_high,pred_mod_high, 
                     out_dir, iterations):
    mod_low = np.reshape(mod_low, (256,256))
    mod_high = np.reshape(mod_high, (256,256))
    phase_low = np.reshape(phase_low, (256,256))
    phase_high = np.reshape(phase_high, (256,256))
    pred_mod_high = np.reshape(pred_mod_high, (256,256))
    # write gt sound:
    D = np.zeros((513,np.shape(mod_low)[1]), dtype=np.complex_)
    D[:256,:] = mod_low*np.exp(1.0j*phase_low)
    D[256:512,:] = mod_high*np.exp(1.0j*phase_high)
    s_gt = librosa.istft(D, hop_length=256)
    l_sig = len(s_gt)
    librosa.output.write_wav(out_dir+'/full1.wav', s_gt, 5000)
    
    D_low = np.copy(D)
    D_low[256:,:] = 0
    s_low = librosa.istft(D_low, length=l_sig, hop_length=256)
    librosa.output.write_wav(out_dir+'/low1.wav', s_low, 5000)
    
    D_pred = np.zeros((513,np.shape(mod_low)[1]), dtype=np.complex_)
    D_pred[:256,:] = mod_low*np.exp(1.0j*phase_low)
    D_pred[256:512,:] = pred_mod_high
    s_recons = f_uti.reconstruct_sig_griffin_lim(D_pred, l_sig, iterations, 
                                                 1024, 256)
    D_recons = librosa.stft(s_recons, n_fft=1024)
    librosa.output.write_wav(out_dir+'/recons_CNN1.wav', s_recons, 5000)
    
    D_low_padded = np.copy(D)
    D_low_padded[256:,:] = 0
    print(' snr signal bf vs signal gt: ', f_uti.snr2(np.abs(D), np.abs(D_low_padded)), '\n',
      'snr signal recons vs signal gt: ', f_uti.snr2(np.abs(D), np.abs(D_recons)))
    return None

#out_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/out_sounds/test2'
out_dir = '/home/felixgontier/data/PROJET AUDIO/out_sounds/test3'
i=1 # indice prediction
write_out_sounds(test_data[i],test_obj[i],test_phase_data[i],
                 test_phase_obj[i], pred_obj[i], out_dir, 100)

D1_low = np.reshape(test_data[i], (256,256))*np.exp(1.0j*np.reshape(test_phase_data[i],(256,256)))
D1_high = np.reshape(test_obj[i], (256,256))*np.exp(1.0j*np.reshape(test_phase_obj[i],(256,256)))
D1 = np.zeros((513,np.shape(D1_low)[1]), dtype=np.complex_)
D1[:256,:] = D1_low
D1[256:512,:] = D1_high          
test_out_dir = '/home/felixgontier/data/PROJET AUDIO/out_sounds/test3'
f_uti.pipeline_recons_sig_sbr(np.abs(D1_low),np.abs(D1[256:512,:]),np.angle(D1_low),
                        np.angle(D1[256:512,:]), 100, test_out_dir)
                 
