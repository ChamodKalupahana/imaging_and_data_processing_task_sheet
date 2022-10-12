import numpy as np
import matplotlib.pyplot as plt

signal = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal.mat")
t = np.arange(0, 48000)

fft_signal = np.fft.fft(signal)
freq = np.arange(0, 48000)

fig, ax = plt.subplots(2, figsize=(15, 8))

ax[0].plot(t, signal, 'b*')
ax[1].plot(freq, fft_signal, 'r*')

plt.show()



print('Hello World')