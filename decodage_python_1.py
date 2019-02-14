# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:04:09 2018

@author: Geoffroy Leconte
"""

# ouverture fichier wave:
import librosa
import librosa.display, librosa.core, librosa.output
import matplotlib.pyplot as plt
import numpy as np
from math import *
import fonctions_utilitaires as f_uti

#import audio_utilities
# open file, stft
DIR = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/i-dont-understand.wav'


y1, sr1 = f_uti.load_signal(DIR) #len(y1) = 36052
# comment régler fenêtre à 25ms, et 512 valeurs différentes pour le spectrogramme?
D = librosa.stft(y1, n_fft=1024)#shape: (513,141) = (f,t)
#win_length = n_fft = 1024. Default hop_length = 1024/4 = 256
# 36052/256 = 141  => t = len(y1)/hop_length
#associated frequencies:
freqs = librosa.core.fft_frequencies(sr=sr1, n_fft=1024)
# freqs = (0, sr/n_fft, 2*sr/n_fft, …, sr/2)
mod, phase = np.abs(D), np.angle(D)

# display function with f_uti.spectrogram

# set phase to 0: magnitude reconstruction
y1_recons = np.real(librosa.istft(mod))

#write new audio file:
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/new.wav'
                         , y1_recons, sr1)

# phase reconstruction with griffin and lim algorithm
#we have the magnitude mod, we want to have the phase.
y1_gl_recons = f_uti.reconstruct_sig_griffin_lim(mod,len(y1), 100, 1024, 256)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/new_gl.wav'
                         , y1_gl_recons, sr1)


### reconstruction hautes fréquences.
## données sur le signal y1 (+sample rate sr1):
# spectrogram
D_low = D[0:256,:]
D_high = D[256:, :]
# magnitude
mod_low = mod[0:256,:]
mod_high = mod[256:,:]
# phase
phase_low = phase[0:256,:]
phase_high = phase[256:,:]

## reconstruction
# on recopie 2 fois la dernière ligne
# on met les hautes fréquences égales aux basses fréquences
mod_low_new = np.concatenate((mod_low, [mod_low[255]]), axis=0)

# calcul du niveau moyen des hf puis mise à l'échelle:
#on multiplie par l'enveloppe spectrale: spectral_env +recons_sig
  
### test:
# signal sans reconstruction hautes fréquences
l_sig = len(y1)
sig_low = librosa.istft(D_low, length=l_sig, hop_length=256)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/sig_sans_recons_hf.wav'
                         , sig_low, sr1)
#signal avec reconstruction hautes fréquences, 100 iter griffin and lim:
spec_env = f_uti.spectral_env(mod_low, mod_high)
sig_recons, D_recons = f_uti.recons_sig(mod_low,D_low, spec_env, l_sig, 100)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/sig_avec_recons_hf.wav'
                         , sig_recons, sr1)


snr1 = f_uti.snr2(y1, sig_recons, sr1)
# créer une pipeline globale et tester sur d'autres données.


