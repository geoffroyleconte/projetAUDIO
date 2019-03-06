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



# test pour sep bf/hf:
audio_dir = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/LibriSpeech/dev-clean/84/121123/'
sound_f = os.path.join(audio_dir, '84-121123-0000.flac')
import soundfile as sf


y, sr = sf.read(sound_f)
l_sig = len(y)




freqs = librosa.core.fft_frequencies(sr=sr, n_fft=1024)
# freqs = (0, sr/n_fft, 2*sr/n_fft, …, sr/2)

cut=51


#freqs[50] = 781.25 => on test 1600*2=3200 Hz pour l'échantillonnage

freqs1 = librosa.core.fft_frequencies(sr=3200, n_fft=1024)
# freqs1[255] = 800 environ


# rééchantillonnage:
sr1 = 3200
y1 = librosa.resample(y, sr, sr1)
l_sig1 = len(y1)
D1 = librosa.stft(y1, n_fft=1024)
s_gt = librosa.istft(D1,length=l_sig1, hop_length=256)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_full.wav'
                         , s_gt, sr1)
D_low = D1
D_low[256:,:] = 0
s_low = librosa.istft(D_low, length=l_sig1, hop_length=256)

## writing:
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_low.wav'
                         , s_low, sr1)
D1_low = D1[:256,:]

# enveloppe spectrale (moyenne de chaque trame (colonne) du spectrogramme)
sp_env1 = f_uti.spectral_env(D1)

# reconstruction du signal: pour les hf, on utilise les bf, chaque trame (colonne)
# des bf est multipliée par le coefficient correspondant de l'enveloppe spectrale
# puis GL sur les hf uniquement (on connait la phase bf).
s1_recons, D1_recons = f_uti.recons_sig(D1_low, sp_env1, l_sig1, 100)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/s_rec_sbr.wav'
                         , s1_recons, sr1)


