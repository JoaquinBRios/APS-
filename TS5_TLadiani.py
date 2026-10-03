#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:49:41 2026

@author: tomy
"""

#%% Librerías
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import square, unit_impulse
from scipy.fft import fft,  fftfreq

import scipy.io as sio
from scipy.io.wavfile import write

#%%Definiciones
N = 1000
fs = 1000

##################
# Lectura de ECG #
##################

fs_ECG = 1000 # Hz

##################
## ECG sin ruido#
##################

ECG = np.load(Path(__file__).parent / 'pdstestbench' / 'ecg_sin_ruido.npy')

plt.figure()
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.plot(ECG)
plt.grid(True)

NE = len(ECG)
print(f"{NE}")
#%%
frec, DEP_ECG_welch = signal.welch(ECG,fs=fs_ECG,nperseg=(max(1, NE//2))) #nperseg mide la longitud de cada periodograma para calcular la DEP 
frec2, DEP_PPG_welch2 = signal.welch(ECG, fs=fs_ECG, nperseg=max(1, NE//4))
frec3, DEP_PPG_welch3 = signal.welch(ECG, fs=fs_ECG, nperseg=max(1, NE//16))
frec4, DEP_PPG_welch4 = signal.welch(ECG, fs=fs_ECG, nperseg=max(1, NE//64))

plt.figure()
plt.title('ECG - Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [W/Hz]')
plt.plot(frec,DEP_ECG_welch, label = "L=N/2")
plt.plot(frec2,DEP_PPG_welch2, label = "L=N/4")
plt.plot(frec3,DEP_PPG_welch3, label = "L=N/16")
plt.plot(frec4,DEP_PPG_welch4, label = "L=N/64")
plt.legend()
plt.show()

#%%
####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# Cargar el archivo CSV como un array de NumPy
#ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppg = np.load(Path(__file__).parent / 'pdstestbench' / 'ppg_sin_ruido.npy')

plt.figure()
plt.title('Señal PPG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.plot(ppg)
plt.grid(True)

N = len(ppg)
print(f"{N}")

L=N/2

frec1, x_ppg_welch1 = signal.welch(ppg, fs=fs_ppg, nperseg=max(1, N//2))
frec2, x_ppg_welch2 = signal.welch(ppg, fs=fs_ppg, nperseg=max(1, N//4))
frec3, x_ppg_welch3 = signal.welch(ppg, fs=fs_ppg, nperseg=max(1, N//16))
frec4, x_ppg_welch4 = signal.welch(ppg, fs=fs_ppg, nperseg=max(1, N//64))

plt.figure()
plt.title('ppg - Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [W/Hz]')
plt.plot(frec1,x_ppg_welch1, label = "L=N/2")
plt.plot(frec2,x_ppg_welch2, label = "L=N/4")
plt.plot(frec3,x_ppg_welch3, label = "L=N/16")
plt.plot(frec4,x_ppg_welch4, label = "L=N/64")
plt.legend()
plt.xlim(0,15)
plt.show()

#%%

####################
# Lectura de audio #
####################

from scipy.io import wavfile
import sounddevice as sd

# Carga el audio; rate es la frecuencia de muestreo, data es el vector
fs_LCC, data = wavfile.read(Path(__file__).parent / 'pdstestbench' / 'la cucaracha.wav')   
sd.play(data, fs_LCC)
#plt.figure()
#plt.plot(wav_data)
sd.wait()
