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


#import audio_utilities
# open file, stft
DIR = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/i-dont-understand.wav'

def load_signal(path):
    y, sr = librosa.load(path)
    return y, sr

y1, sr1 = load_signal(DIR) #len(y1) = 36052
# comment régler fenêtre à 25ms, et 512 valeurs différentes pour le spectrogramme?
D = librosa.stft(y1, n_fft=1024)#shape: (513,141) = (f,t)
#win_length = n_fft = 1024. Default hop_length = 1024/4 = 256
# 36052/256 = 141  => t = len(y1)/hop_length
#associated frequencies:
freqs = librosa.core.fft_frequencies(sr=sr1, n_fft=1024)
# freqs = (0, sr/n_fft, 2*sr/n_fft, …, sr/2)
mod, phase = np.abs(D), np.angle(D)

# display function
def spectrogram(D):
    plt.figure(figsize=(15,10))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),ref=np.max),
                             y_axis='log', x_axis='time', hop_length=256)
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return None

# set phase to 0: magnitude reconstruction
y1_recons = np.real(librosa.istft(mod))

#write new audio file:
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/new.wav'
                         , y1_recons, sr1)

# phase reconstruction with griffin and lim algorithm

def reconstruct_sig_griffin_lim(magnitude_spectrogram, len_init_sig, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.

    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.

    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
        and the columns correspond to frequency bins.
            (fft_size (int): The FFT size, which should be a power of 2.
             hopsamp (int): The hope size in samples.)
        len_init_sig: length of the signal we want to reconstruct
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
        is sufficient.

    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    len_samples = int(len_init_sig)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = librosa.stft(x_reconstruct, n_fft=1024)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = librosa.istft(proposal_spectrogram, length=len_init_sig)
        diff = sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        #print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct

#we have the magnitude mod, we want to have the phase.
y1_gl_recons = reconstruct_sig_griffin_lim(mod,len(y1), 100)
librosa.output.write_wav('C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/quelques sons/new_gl.wav'
                         , y1_gl_recons, sr1)


### reconstruction hautes fréquences.
## données sur le signal y1 (+sample rate sr1):
# spectrogram
D_low = D[0:255,:]
D_high = D[255:, :]
# magnitude
mod_low = mod[0:255,:]
mod_high = mod[255:,:]
# phase
phase_low = phase[0:255,:]
phase_high = phase[255:,:]

## reconstruction
# on recopie 2 fois la dernière ligne
# on met les hautes fréquences égales aux basses fréquences
mod_low_new = np.concatenate((mod_low, [mod_low[254]]), axis=0)

# calcul du niveau moyen des hf puis mise à l'échelle:
def spectral_env(mod_low, mod_high):#même taille
    #enveloppe spectrale:  rapport ratio intensité high/low pour chaque bande
    n = len(mod_low)
    spec_env = np.array([])
    for i in range(n):
        low_mean = np.mean(mod_low[i])
        high_mean = np.mean(mod_high[i])
        spec_env = np.append(spec_env, [low_mean/high_mean])
    return spec_env
        
#on multiplie par l'enveloppe spectrale
def recons_sig(mod_low, mod_high, iterations):# iter pour griffin and lim
    spec_env = spectral_env(mod_low, mod_high)
    mod_high_recons = (mod_low.T * spec_env).T
    mod_recons = np.concatenate((mod_low, mod_high_recons), axis=0)
    #griffin and lim (seulement partie haute fréquence:
    #implémenter griffin and lim pour qu'il renvoie plutôt la phase.
    

    







