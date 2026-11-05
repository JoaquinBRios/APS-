"""
Respuesta en frecuencia de un filtro digital a partir de h[n].
Ejecutar en Spyder. Requiere scipy, numpy y matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

wp = 4/100
ws = 3/ 100
G_p = 1
G_s = 80

x = np.arange(0,np.pi,0.001)
"""
b, a = signal.iirdesign(wp, ws, G_p, G_s, ftype='butter')

print(len(b))
w, H = signal.freqz(b, a, worN=2048)
modulo = np.abs(H)                 # módulo en veces (escala lineal)
fase = np.unwrap(np.angle(H))      # fase en radianes

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(w, modulo)
ax[0].set_ylabel('Módulo (veces)')
ax[0].set_title('Respuesta en frecuencia')
ax[0].grid(True)

# Bordes de paso y rechazo. freqz expresa w en radianes/muestra;
# iirdesign recibe wp y ws normalizados respecto de Nyquist.
wp_rad = wp * np.pi
ws_rad = ws * np.pi
for axes in ax:
    axes.axvline(wp_rad, color='orange', linestyle='--', label='wp')
    axes.axvline(ws_rad, color='red', linestyle='--', label='ws')

# Límites de módulo correspondientes a Gp y Gs expresados en dB.
ax[0].axhline(10 ** (-G_p / 20), color='orange', linestyle=':', label='Límite Gp')
ax[0].axhline(10 ** (-G_s / 20), color='red', linestyle=':', label='Límite Gs')
ax[0].legend()

ax[1].plot(w, fase)

ax[1].set_ylabel('Fase (rad)')
ax[1].set_xlabel('Frecuencia angular omega (rad/muestra)')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()

print('b (numerador) =', b)
print('a (denominador) =', a)
"""
# %%
ws = 60
wp = 40
G_p = 3
G_s = 60
fs = 1000

sosb = signal.iirdesign(wp, ws, G_p, G_s, ftype='butter',output='sos',fs=fs)
sosc = signal.iirdesign(wp, ws, G_p, G_s, ftype='cheby1',output='sos',fs=fs)
sose = signal.iirdesign(wp, ws, G_p, G_s, ftype='elip',output='sos',fs=fs)

W, H = signal.freqz_sos(sosb,fs=fs)

modulo = np.abs(H)                 
fase = np.unwrap(np.angle(H))      

#Solo con filtro Butter
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(W, 20*np.log(modulo))
ax[0].set_ylabel('Módulo (veces)')
ax[0].set_title('Respuesta en frecuencia')
ax[0].grid(True)

# Bordes de paso y rechazo. freqz expresa w en radianes/muestra;
# iirdesign recibe wp y ws normalizados respecto de Nyquist.
wp_rad = wp
ws_rad = ws
for axes in ax:
    axes.axvline(wp_rad, color='orange', linestyle='--', label='wp')
    axes.axvline(ws_rad, color='red', linestyle='--', label='ws')

# Límites de módulo correspondientes a Gp y Gs expresados en dB.
ax[0].axhline( -G_p , color='orange', linestyle=':', label='Límite Gp')
ax[0].axhline( -G_s , color='red', linestyle=':', label='Límite Gs')
ax[0].legend()

ax[1].plot(W, fase)

ax[1].set_ylabel('Fase (rad)')
ax[1].set_xlabel('Frecuencia angular omega (rad/muestra)')
ax[1].grid(True)
ax[1].legend()

dem = - np.gradient(fase)

plt.figure(2)
plt.plot(W,dem)
plt.ylabel ("Demora [segundos]")
plt.xlabel("Frecuencia angular omega (rad/muestra)")
plt.grid(True)

"""
y = signal.sosfilt(sos, x, zi=None)


fig2, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(x, y)
ax[0].set_ylabel('Módulo (veces)')
ax[0].set_title('Respuesta en frecuencia')
ax[0].grid(True)

# Bordes de paso y rechazo. freqz expresa w en radianes/muestra;
# iirdesign recibe wp y ws normalizados respecto de Nyquist.
wp_rad = wp * np.pi
ws_rad = ws * np.pi
for axes in ax:
    axes.axvline(wp_rad, color='orange', linestyle='--', label='wp')
    axes.axvline(ws_rad, color='red', linestyle='--', label='ws')

# Límites de módulo correspondientes a Gp y Gs expresados en dB.
ax[0].axhline(10 ** (-G_p / 20), color='orange', linestyle=':', label='Límite Gp')
ax[0].axhline(10 ** (-G_s / 20), color='red', linestyle=':', label='Límite Gs')
ax[0].legend()

plt.tight_layout()
plt.show()
"""
import scipy.io as sio
fs_ecg = 1000

sio.whosmat('ECG_TP4.mat')
mat_struct=sio.loadmat('./ECG_TP4.mat')

ecg_one_lead_audio =mat_struct ['ecg_lead'] * 1.0
ecg_one_lead_audio = ecg_one_lead_audio.flatten()

ecg_butter = signal.sosfilt(sosb,ecg_one_lead_audio)
ecg_coward = signal.sosfilt(sosc,ecg_one_lead_audio)
ecg_ellip = signal.sosfilt(sosc,ecg_one_lead_audio)


plt.figure(3)
plt.title("Señales")
plt.plot(ecg_butter[5000:12000],label='Señal filtrada por SOS')
plt.plot(ecg_coward[5000:12000],label='Señal filtrada por SOS')
plt.plot(ecg_ellip[5000:12000],label='Señal filtrada por SOS')
plt.plot(ecg_one_lead_audio[5000:12000],label='Señal de ECG')
plt.plot
plt.grid()
plt.legend()
plt.show()

plt.figure(4)
plt.title("Señales")
plt.plot(ecg_butter[5000:12000],label='Señal filtrada por SOS')
plt.plot(ecg_coward[5000:12000],label='Señal filtrada por SOS')
plt.plot(ecg_ellip[5000:12000],label='Señal filtrada por SOS')
plt.plot(ecg_one_lead_audio[5000:12000],label='Señal de ECG')
plt.plot
plt.grid()
plt.legend()
plt.show()

x=np.zeros(len(ecg_one_lead_audio))
x[0] = 1

t= signal.sosfilt(sosb,x)

plt.figure(6)
plt.title("Señales")
plt.plot(t,label='Señal filtrada por SOS')
plt.plot(x,label='Señal de ECG')
plt.grid()
plt.legend()
plt.show()
