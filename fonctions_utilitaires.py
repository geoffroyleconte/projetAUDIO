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

def reconstruct_sig_griffin_lim(spectro, len_init_sig, iterations, n_fft, hop_length):
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
    magnitude_spectrogram = np.abs(spectro)
    phase_low = np.angle(spectro[0:256,:])
    len_samples = int(len_init_sig)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = librosa.stft(x_reconstruct, n_fft=n_fft, hop_length = hop_length)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        reconstruction_angle[0:256,:] = phase_low
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = librosa.istft(proposal_spectrogram, length=len_init_sig)
        diff = sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        #print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    print('shape angle',np.shape(reconstruction_angle))
    print('shape recons_spec',np.shape(reconstruction_spectrogram))
    return x_reconstruct

def spectral_env(D):#même taille
    """Calcul de l'enveloppe spectrale"""
    spec_env = []
    m,n = np.shape(D)
    D = D[0:(m-1),:]
    mod_low = np.abs(D[:(m-1)//2,:])
    mod_high = np.abs(D[(m-1)//2:,:])
    for i in range(n):
        low_mean_trame = np.sum(mod_low[:,i])
        high_mean_trame = np.sum(mod_high[:,i])
        if low_mean_trame == 0:
            low_mean_trame = 1e-4
        spec_env.append(high_mean_trame/low_mean_trame)
    return np.array(spec_env)


def recons_sig(D_low, spec_env, l_sig, iterations):
    """reconstruction du signal par SBR"""
    # l_sig et iter pour griffin and lim
    
    mod_low = np.abs(D_low)
    mod_high_recons = mod_low 
    # colonnes de mod_low multipliées par spec env.
    for i in range(np.shape(mod_high_recons)[1]):
        mod_high_recons[:,i] = mod_high_recons[:,i] * spec_env[i]
        
    print(np.shape(mod_high_recons))
    #griffin and lim (seulement partie haute fréquence)
    ##### griffin & lim sur tout le signal et pas seulement hf
    D_sig = np.zeros((513, np.shape(D_low)[1]), dtype=np.complex_)
    D_sig[0:256, :] = D_low
    D_sig[256:512,:] = mod_high_recons
    x_recons = reconstruct_sig_griffin_lim(D_sig,
                                                l_sig, iterations, 1024, 256)
    
    D_recons = librosa.stft(x_recons, n_fft=1024)
    return x_recons, D_recons

def snr1(original_sig, recons_sig):
    noise = original_sig-recons_sig
    P_noise = np.sum(np.square(noise))
    P_original = np.sum(np.square(original_sig))
    return 20*log10(P_original/P_noise)


def snr2(original_mag, recons_mag):
    noise = original_mag-recons_mag
    P_noise = np.sum(np.square(noise))
    P_original = np.sum(np.square(original_mag))
    return 20*log10(P_original/P_noise)

def pipeline_recons_sig_sbr(mod_low,mod_high,phase_low,phase_high, 
                            iterations,out_dir):
    # on connait la sortie phase high et mod_high (supposés inconnus sauf pour sp_env)
    """arguments autorisés: D,
    iterations pour g&l, nom signal de sortie, direction de sortie """
    """récrit les signaux dans la direction souhaitée, affiche le snr """
    D = np.zeros((513,np.shape(mod_low)[1]), dtype=np.complex_)
    D[:256,:] = mod_low*np.exp(1.0j*phase_low)
    D[256:512,:] = mod_high*np.exp(1.0j*phase_high)
    s_gt = librosa.istft(D, hop_length=256)
    l_sig = len(s_gt)
    librosa.output.write_wav(out_dir+'/full.wav', s_gt, 5000)
    
    D_low = np.copy(D)
    D_low[256:,:] = 0
    s_low = librosa.istft(D_low, length=l_sig, hop_length=256)
    librosa.output.write_wav(out_dir+'/low.wav', s_low, 5000)
    
    D_low = D[:256,:]
    sp_env = spectral_env(D)
    s_recons, D_recons = recons_sig(D_low, sp_env, l_sig, iterations)
    librosa.output.write_wav(out_dir+'/recons_sbr.wav', s_recons, 5000)
    
    D_low_padded = np.copy(D)
    D_low_padded[256:,:] = 0
    print(' snr signal bf vs signal gt: ', snr2(np.abs(D), np.abs(D_low_padded)), '\n',
      'snr signal recons vs signal gt: ', snr2(np.abs(D), np.abs(D_recons)))
    
    return None

import soundfile as sf    
path = 'C:/Users/Geoffroy Leconte/Documents/cours/projet AUDIO/out_sounds/test3/'
sig, sr = sf.read(path+'full1.wav')
D=librosa.stft(sig, n_fft=1024, hop_length=256)
sig_low, sr = sf.read(path+'low1.wav')
D_low=librosa.stft(sig_low, n_fft=1024, hop_length=256)
sig_cnn, sr = sf.read(path+'recons_CNN1.wav')
D_cnn=librosa.stft(sig_cnn, n_fft=1024, hop_length=256)
sig_sbr, sr = sf.read(path+'recons_sbr1.wav')
D_sbr=librosa.stft(sig_sbr, n_fft=1024, hop_length=256)
