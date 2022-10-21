from signal import signal
import numpy as np
import matplotlib.pyplot as plt


def time_freq_decom(signal_brain_num):
    if signal_brain_num == 1:
        signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain.mat")
    if signal_brain_num == 2:
        signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain2.mat")

    t = np.arange(0, np.size(signal_brain))
    fft_signal_1 = np.abs(np.fft.fft(signal_brain))

    freq = np.arange(0, np.size(fft_signal_1))

    fig, ax = plt.subplots(2, figsize=(15, 8))

    ax[0].plot(t, signal_brain, 'b-')
    ax[1].plot(freq, fft_signal_1, 'r-')
    
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Frequency')
    ax[0].set_title('Unprocessed Signal '+ str(signal_brain_num))

    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Power strectrum')
    ax[1].set_title('Fourier Transformed Signal')
    
    plt.savefig('Task 3 Images/signal_figure.jpeg')

    plt.show()

time_freq_decom(signal_brain_num=2)