import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci

def freq_filter():
    signal = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal.mat")
    t = np.arange(0, 48000)

    fft_signal = np.fft.fft(signal)
    freq = np.arange(0, 48000)

    fig, ax = plt.subplots(5, figsize=(15, 8))

    ax[0].plot(t, signal, 'b-')
    ax[1].plot(freq, fft_signal, 'r-')

    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Signal')

    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Power strectrum')

    x = t
    mu_1 = 6
    mu_2 = 49000
    sig = 1000 # sigma regluates the half width full maximum nik says <3 <3
    #guassian = np.exp(-np.power(x - mu_1, 2.) / (2 * np.power(sig, 2.))) + np.exp(-np.power(x - mu_2, 2.) / (2 * np.power(sig, 2.)))
    guassian = np.exp(-np.power(x - mu_1, 2.) / (2 * np.power(sig, 2.))) + np.exp(-np.power(x - mu_2, 2.) / (2 * np.power(sig, 2.)))

    ax[2].plot(freq, guassian, 'g-')
    
    ax[2].set_xlabel('Frequency')
    ax[2].set_ylabel('Power strectrum')

    filtered_signal = guassian * fft_signal

    ax[3].plot(freq, filtered_signal, 'r-')
    
    ax[3].set_xlabel('Frequency')
    ax[3].set_ylabel('Power strectrum')

    inverse_filtered_signal = np.fft.ifft(filtered_signal)

    ax[4].plot(t, inverse_filtered_signal, 'r-')
    
    ax[4].set_xlabel('Time')
    ax[4].set_ylabel('Signal')

    plt.show()


def hilbert_signal():
    signal = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal.mat")
    
    imag_analytic_sginal = np.imag(sci.hilbert(signal))
    real_analytic_sginal = np.real(sci.hilbert(signal))
    abs_analytic_sginal = np.abs(sci.hilbert(signal))
    t = np.arange(0, 48000)
    
    fig, ax = plt.subplots(4, figsize=(15, 8))
    
    ax[0].plot(t, signal, 'b-')

    ax[1].plot(t, imag_analytic_sginal, 'r-')
    ax[2].plot(t, real_analytic_sginal, 'g-')
    ax[3].plot(t, abs_analytic_sginal, 'k-')

    plt.show()

    pass


#freq_filter()
hilbert_signal()