import numpy as np
import matplotlib.pyplot as plt


def time_freq_decom(signal_brain_num):
    """_summary_

    Args:
        signal_brain_num (int): 1 or 2, decribes which file to extract
    """

    #extract signals from either signal_brain.mat or signal_brain2.mat)
    if signal_brain_num == 1:
        signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain.mat")
        # for one repeat (shouldn't need to do FT of each 6.5 sec repeat and stack because the total FT does the same thing)
        #signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain.mat")[0:3900]
    
    if signal_brain_num == 2:
        signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain2.mat")

    # plot the signal against 617.5s of time 
    t = np.linspace(0, 617.5 ,np.size(signal_brain))
    
    # create subplots
    fig, ax = plt.subplots(2, figsize=(15, 8))
    ax[0].plot(t, signal_brain, 'b-')
    
    # calculate and plot fourier transfrom against 600hz of freq
    fft_signal_1 = np.abs(np.fft.fft(signal_brain))
    fft_signal_1 = np.abs(np.fft.ifft(fft_signal_1))
    freq = np.linspace(0, 600, np.size(fft_signal_1))
    
    ax[1].plot(freq, fft_signal_1, 'r-')
    
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_title('Unprocessed Signal '+ str(signal_brain_num))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power strectrum')
    ax[1].set_title('Fourier Transformed Signal')
    plt.savefig('Task 3 Images/signal_figure.jpeg')
    plt.show()

time_freq_decom(signal_brain_num=1)