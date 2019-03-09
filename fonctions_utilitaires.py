# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:28:30 2019

@author: Geoffroy Leconte
"""

import librosa
import librosa.display, librosa.core, librosa.output
import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy import signal



def spectrogram(D):
    #### fonction pour l'affichage du spectrogramme ####
    plt.figure(figsize=(15,10))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),ref=np.max),
                             y_axis='log', x_axis='time', hop_length=256)
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return None

def reconstruct_sig_griffin_lim(magnitude_spectrogram, len_init_sig, iterations, n_fft, hop_length):
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
        reconstruction_spectrogram = librosa.stft(x_reconstruct, n_fft=n_fft, hop_length = hop_length)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = librosa.istft(proposal_spectrogram, length=len_init_sig)
        diff = sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        #print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct

def spectral_env(D):#même taille
    """Calcul de l'enveloppe spectrale"""
    spec_env = np.array([])
    m,n = np.shape(D)
    D = D[0:(m-1),:]
    mod_low = np.abs(D[:(m-1)//2,:])
    mod_high = np.abs(D[(m-1)//2:,:])
    for i in range(n):
        low_mean_trame = np.sum(mod_low[:,i])
        high_mean_trame = np.sum(mod_high[:,i])
        if low_mean_trame == 0:
            low_mean_trame = 1e-4
        spec_env = np.append(spec_env, [high_mean_trame/low_mean_trame])
    return spec_env


def recons_sig(D_low, spec_env, l_sig, iterations):
    """reconstruction du signal par SBR"""
    # l_sig et iter pour griffin and lim
    
    mod_low = np.abs(D_low)
    mod_high_recons = mod_low 
    # colonnes de mod_low multipliées par spec env.
    for i in range(np.shape(mod_high_recons)[1]):
        mod_high_recons[:,i] = mod_high_recons[:,i] * spec_env[i]
        
    #griffin and lim (seulement partie haute fréquence):
    x_recons_high = reconstruct_sig_griffin_lim(mod_high_recons,
                                                l_sig, iterations, 510, 256)
    
    D_recons_high = librosa.stft(x_recons_high,
                                 n_fft=510, hop_length=256)
    D_recons = np.zeros((512, np.shape(D_low)[1]), dtype=np.complex_)
    D_recons[:256,:] = D_low
    #print(np.shape(D_recons_high))
    ##D_recons[cut:512, :] = D_recons_high
    D_recons[256:, :] = D_recons_high
    # signal reconstruit final (avec la phase):
    x_recons = librosa.istft(D_recons, length=l_sig, hop_length=256)
    return x_recons, D_recons


def snr2(original_sig, recons_sig):
    noise = original_sig-recons_sig
    P_noise = np.sum(np.square(noise))
    P_original = np.sum(np.square(original_sig))
    return 20*log10(P_original/P_noise)


    
    
    