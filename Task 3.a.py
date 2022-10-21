import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.signal as sci

def freq_filter():
    """ 
    Applies fourier transfrom to signal.mat and separates the frequencies using a guassian filter
    # IMPORTANT # my fourier transform  is not correct becasue the inverse fourier transfrom doesn't show the orginal signal
    """

    #------------------------------------------------------------------------
    #--------------------- Fourier Transform ---------------------
    #------------------------------------------------------------------------

    # extract data from signal.mat
    signal = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal.mat")
    
    # number of data points
    n = np.size(signal)

    # plot the signal against 40s of time 
    t = np.linspace(0, 40, n)
    
    # create a subplot for rest of the plots
    fig, ax = plt.subplots(3, 2, figsize=(15, 8))
    ax[0, 0].plot(t, signal, 'b-')

    # take the fourier transfrom of the signal
    fft_signal = np.abs(np.fft.fft(signal))
    
    # plot the fouier transfrom against 50hz of frequency
    freq = np.linspace(0, 50, n)
    ax[0, 1].plot(freq, fft_signal, 'r-')

    #------------------------------------------------------------------------
    #--------------------- Frequency Filter ---------------------
    #------------------------------------------------------------------------

    # create a guassian function
    x = t
    mu_1 = 6
    mu_2 = 10
    sig = 0.4 # sigma regluates the half width full maximum nik says <3 <3
    guassian = np.exp(-np.power(x - mu_1, 2.) / (2 * np.power(sig, 2.)))
    #guassian = np.exp(-np.power(x - mu_1, 2.) / (2 * np.power(sig, 2.))) + np.exp(-np.power(x - mu_2, 2.) / (2 * np.power(sig, 2.)))

    # plot the guassian onto a subplot (could plot the gusssian function onto the fliter signal)
    ax[1, 0].plot(freq, guassian, 'g-')

    # apply the frequency filter to the fourier transfromed signal
    filtered_signal = guassian * fft_signal

    ax[1, 1].plot(freq, filtered_signal, 'r-')

    #------------------------------------------------------------------------
    #--------------------- Inverse Frequency Filter ---------------------
    #------------------------------------------------------------------------
    
    #inverse_filtered_signal = np.fft.ifft(filtered_signal)
    inverse_filtered_signal = np.fft.ifft(fft_signal)
    t_inverse = t = np.linspace(0, 40, n)
    
    ax[2, 0].plot(t_inverse, inverse_filtered_signal, 'r-')

    #------------------------------------------------------------------------
    #--------------------- Hilbert Method ---------------------
    #------------------------------------------------------------------------

    abs_analytic_sginal = sci.hilbert(np.abs(inverse_filtered_signal))
    
    ax[2, 1].plot(t, abs_analytic_sginal, 'b-')
    
    # plotting infomation (could simplify)
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Signal')
    ax[0, 0].set_title('Unprocessed Signal')
    ax[0, 1].set_xlabel('Frequency')
    ax[0, 1].set_ylabel('Power strectrum')
    ax[0, 1].set_title('Fourier Transform')
    ax[1, 0].set_xlabel('Frequency')
    ax[1, 0].set_ylabel('Power strectrum')
    ax[1, 0].set_title('Guassian Filter')
    ax[1, 1].set_xlabel('Frequency')
    ax[1, 1].set_ylabel('Power strectrum')
    ax[1, 1].set_title('Filtered Signal')
    ax[2, 0].set_xlabel('Time')
    ax[2, 0].set_ylabel('Signal')
    ax[2, 0].set_title('Inverse Filtered Signal')
    ax[2, 1].set_title('Analytic Signal')
    fig.subplots_adjust(hspace=0.6)
    plt.savefig(r"Task 3 Images\filtered_signal.jpeg")
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


def test_fft():
    
    sample_rate = 600 # in Hz
    sample_interval = 1 / sample_rate
    total_time = 10

    #t = np.linspace(0, 3, 100)
    t = np.linspace(0, total_time, sample_rate * total_time)
    f = 0.5
    signal = np.sin(2*np.pi*f*t)

    plt.plot(t, signal)

    n = len(signal)
    k = np.arange(n)
    T = n / sample_rate
    frq = k / T
    freq = frq[int(n/2):n]

    fft_signal = np.imag(np.fft.fft(signal)) / n
    #freq = np.linspace(0, 1, np.size(fft_signal))
    #freq = np.linspace(0, 1, )

    plt.figure()
    plt.plot(freq, fft_signal[int(n/2):n])
    plt.show()

    pass

#freq_filter()
test_fft()
#freq_filter_10_sec()
#hilbert_signal()