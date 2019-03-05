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
import os

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
sig_recons, D_recons = f_uti.recons_sig(mod_low,D_low, spec_env, l_sig, 100, 256)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/sig_avec_recons_hf.wav'
                         , sig_recons, sr1)


snr1 = f_uti.snr2(y1, sig_recons)
# créer une pipeline globale et tester sur d'autres données.

# test pour sep bf/hf:
audio_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/LibriSpeech/dev-clean/84/121123/'
sound_f = os.path.join(audio_dir, '84-121123-0000.flac')
import soundfile as sf
y_t, sr_t = sf.read(sound_f)
l_sig = len(y_t)
D_t = librosa.stft(y_t, n_fft=1024)



cut=64


s_low,s_full, s_rec_t = f_uti.pipeline_sig_recons(D_t, l_sig, sr_t, cut)

## writing:
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_low.wav'
                         , s_low, sr_t)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_full.wav'
                         , s_full, sr_t)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_rec_sbr.wav'
                         , s_rec_t, sr_t)




audio_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/LibriSpeech/dev-clean/84/121123/'
sound_f = os.path.join(audio_dir, '84-121123-0000.flac')
import soundfile as sf
y_t, sr_t = sf.read(sound_f)
l_sig = len(y_t)
D_t = librosa.stft(y_t, n_fft=1024)
D_t_low256 = np.abs(D_t[0:256, :])
D_t_high256 = np.abs(D_t[256:512, :])
s_low256 = librosa.istft(D_t_low256, length=l_sig, hop_length=256)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_low256.wav'
                         , s_low256, sr_t)
# 256 pour comparer
cut=256
print(cut)
# test reconstruction méthode classique:
sp_env_t256 = f_uti.spectral_env(np.abs(D_t_low256), np.abs(D_t_high256))
s_rec_t256, D_rec_t256 = f_uti.recons_sig(np.abs(D_t_low256),D_t_low256, 
                                    sp_env_t256, l_sig, 150, cut)

librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_rec_sbr256.wav'
                         , s_rec_t256, sr_t)