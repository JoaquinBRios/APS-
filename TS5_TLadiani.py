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
"""
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
frec2, DEP_ECG_welch2 = signal.welch(ECG, fs=fs_ECG, nperseg=max(1, NE//4))
frec3, DEP_ECG_welch3 = signal.welch(ECG, fs=fs_ECG, nperseg=max(1, NE//16))
frec4, DEP_ECG_welch4 = signal.welch(ECG, fs=fs_ECG, nperseg=max(1, NE//64))

plt.figure()
plt.title('ECG - Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [W/Hz]')
plt.plot(frec,DEP_ECG_welch, label = "L=N/2")
plt.plot(frec2,DEP_ECG_welch2, label = "L=N/4")
plt.plot(frec3,DEP_ECG_welch3, label = "L=N/16")
plt.plot(frec4,DEP_ECG_welch4, label = "L=N/64")
plt.legend()
plt.show()

df = frec[1] - frec[0]
potencia_acum = np.cumsum(DEP_ECG_welch) * df
limite = np.searchsorted(potencia_acum, 0.99 * potencia_acum[-1])
ancho_banda_99 = frec[limite]

print(f"Ancho de banda del 99 % de potencia: {ancho_banda_99:.2f} Hz")

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

"""
#%%

####################
# Lectura de audio #
####################

from scipy.io import wavfile
import sounddevice as sd
import scipy.signal as sig

#%%
##Introduzco función para estimar vía Blackman-Tukey


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
####################
# Lectura de audio #
####################

from scipy.io import wavfile
import sounddevice as sd

def blackman_tukey(x, fs_audio, M=None):
    x = np.asarray(x)

    # Convertir estéreo a mono, si hace falta
    if x.ndim == 2:
        x = x.mean(axis=1)

    x = x.astype(float)
    x -= np.mean(x)
    N = len(x)

    if N == 0:
        raise ValueError("El audio está vacío.")

    if M is None:
        M = max(1, N // 5)
    M = min(int(M), N)

    # Autocorrelación sesgada para retardos -(M-1) ... (M-1)
    r_completa = sig.correlate(x, x, mode="full", method="fft")
    r = r_completa[N - M:N + M - 1] / N

    # Ventana Blackman centrada en el retardo cero
    r *= sig.windows.blackman(2 * M - 1)

    # Reordenar retardos para calcular la FFT
    r_fft = np.zeros(N)
    r_fft[:M] = r[M - 1:]
    if M > 1:
        r_fft[-(M - 1):] = r[:M - 1]

    # DEP unilateral, en unidades de amplitud²/Hz
    Pxx = np.real(np.fft.rfft(r_fft)) / fs_audio
    if N % 2 == 0:
        Pxx[1:-1] *= 2
    else:
        Pxx[1:] *= 2

    frec = np.fft.rfftfreq(N, d=1 / fs_audio)
    return frec, np.maximum(Pxx, 0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

fs_LCC, data = wavfile.read(
    Path(__file__).parent / "pdstestbench" / "la cucaracha.wav"
)

ax1.plot(data)
ax1.set_title("Audio de La Cucaracha")
ax1.set_xlabel("Muestras")
ax1.set_ylabel("Amplitud [cuentas]")
ax1.grid(True)

frec, DEP_AUDIO_BT = blackman_tukey(data, fs_LCC)

ax2.plot(frec, DEP_AUDIO_BT, label="Blackman–Tukey")
ax2.set_title("Estimación de audio - Blackman–Tukey")
ax2.set_xlabel("Frecuencia [Hz]")
ax2.set_ylabel("DEP [cuentas²/Hz]")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()