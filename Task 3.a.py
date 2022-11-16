import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import scipy.signal as sci

def freq_filter(signal ,guassian_filter_centre, guassian_filter_sigma, show_plot):
    """ 
    Applies fourier transfrom to signal.mat and separates the frequencies using a guassian filter
    # IMPORTANT # my fourier transform  is not correct becasue the inverse fourier transfrom doesn't show the orginal signal
    # Fixed this error # had to use np.fft.fftfreq rather than create my own frequency array
    """

    #------------------------------------------------------------------------
    #--------------------- Fourier Transform ---------------------
    #------------------------------------------------------------------------
    # number of data points
    n = np.size(signal)

    # plot the signal against 40s of time 
    t = np.linspace(0, 40, n)

    # take the fourier transfrom of the signal
    fft_signal = np.fft.fft(signal)
    
    # tried using np.linspace for this part but np.fft.fftfreq is correct here
    # freq = np.linspace(0, 50, n)

    sample_rate = 1200
    sample_interval = 1 / sample_rate
    freq = np.fft.fftfreq(np.size(fft_signal), sample_interval)

    #------------------------------------------------------------------------
    #--------------------- Frequency Filter ---------------------
    #------------------------------------------------------------------------

    # create a guassian function
    x = freq
    #mu_1 = 5 for 0-10s
    mu_1 = guassian_filter_centre
    sig = guassian_filter_sigma # sigma regluates the half width full maximum
    #sig = 0.5
    guassian = np.exp(-np.power(x - mu_1, 2.) / (2 * np.power(sig, 2.)))

    # apply the frequency filter to the fourier transfromed signal
    filtered_signal = guassian * fft_signal

    #------------------------------------------------------------------------
    #--------------------- Inverse Fourier Transform ---------------------
    #------------------------------------------------------------------------
    
    inverse_filtered_signal = np.fft.ifft(filtered_signal)
    #inverse_filtered_signal = np.fft.ifft(fft_signal)
    

    #------------------------------------------------------------------------
    #--------------------- Hilbert Method ---------------------
    #------------------------------------------------------------------------

    abs_analytic_sginal = sci.hilbert(np.abs(inverse_filtered_signal))
    
    
    if show_plot == True:
        # create a subplot for rest of the plots
        fig, ax = plt.subplots(3, 2, figsize=(15, 8))
        # plot the fouier transfrom against 50hz of frequency
        ax[0, 0].plot(t, signal, 'b-')

        ax[0, 1].plot(freq, np.abs(fft_signal), 'r-')
        
        # plot the guassian onto a subplot (could plot the gusssian function onto the fliter signal)
        ax[1, 0].plot(freq, guassian, 'g-')
        
        ax[1, 1].plot(freq, np.abs(filtered_signal), 'r-')

        ax[2, 0].plot(t, inverse_filtered_signal, 'r-')

        ax[2, 1].plot(t, np.abs(abs_analytic_sginal), 'b-')

        # plotting infomation
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
        #ax[1, 0].set_xlim([mu_1 - sig, mu_1 + sig])
        #ax[1, 0].set_xlim([4 , 8])
        fig.subplots_adjust(hspace=0.6)
        plt.savefig(r"Task 3 Images\filtered_signal.jpeg")
        plt.show()

    print('Frequency {freq:.2f} done'.format(freq=guassian_filter_centre))

    return np.abs(abs_analytic_sginal)

def hilbert_signal(signal):

    #num_of_freq = int(580 / 2)
    num_of_freq = int(580)
    max_freq = int(100)
    num_of_data_points = 48000

    freq = np.linspace(1, max_freq, num_of_freq)
    total_signal = np.zeros([num_of_freq, num_of_data_points])

    # filter through 580 frequencies
    for i in range(0, num_of_freq):
        temp_signal = freq_filter(signal=signal, guassian_filter_centre=freq[i], guassian_filter_sigma=0.1, show_plot=False)
        total_signal[i] = temp_signal
    
    plt.imshow(total_signal, aspect='auto')
    plt.xticks(ticks = [np.linspace(0, 40, num_of_data_points)])
    plt.show()

    return


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
    #freq = frq[int(n/2):n]
    #freq = frq
    freq = np.fft.fftfreq(n, sample_interval)

    fft_signal = np.fft.fft(signal) / n
    #freq = np.linspace(0, 1, np.size(fft_signal))
    #freq = np.linspace(0, 1, )

    plt.figure()
    #plt.plot(freq, fft_signal[int(n/2):n])
    plt.plot(freq, fft_signal)
    plt.show()

    pass

def example_code():
    #fs is sampling frequency
    fs = 100.0
    time = np.linspace(0,10,int(10*fs),endpoint=False)

    #wave is the sum of sine wave(1Hz) and cosine wave(10 Hz)
    wave = np.sin(np.pi*time)+ np.cos(np.pi*time)
    #wave = np.exp(2j * np.pi * time )

    # Compute the one-dimensional discrete Fourier Transform.

    fft_wave = np.fft.fft(wave)

    # Compute the Discrete Fourier Transform sample frequencies.

    fft_fre = np.fft.fftfreq(n=wave.size, d=1/fs)

    plt.subplot(211)
    plt.plot(fft_fre, fft_wave.real, label="Real part")
    plt.xlim(-50,50)
    plt.ylim(-600,600)
    plt.legend(loc=1)
    plt.title("FFT in Frequency Domain")

    plt.subplot(212)
    plt.plot(fft_fre, fft_wave.imag,label="Imaginary part")
    plt.legend(loc=1)
    plt.xlim(-50,50)
    plt.ylim(-600,600)
    plt.xlabel("frequency (Hz)")

    plt.show()
    pass


# extract data from signal.mat
signal = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal.mat")

#freq_filter()
#freq_filter(guassian_filter_centre=5, show_plot=True)
#test_fft()
#example_code()
#freq_filter_10_sec()
hilbert_signal(signal=signal)