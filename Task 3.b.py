from random import sample
import numpy as np
import matplotlib.pyplot as plt


def time_freq_decom(signal_brain_num):
    """    
    # tried using np.linsapce
    #freq = np.linspace(0, 600, np.size(fft_signal_1))

    # the fourier transfrom shows the peak frequency to be at 7.80Hz to 7.82Hz

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
    
    # calculate the fourier transfrom
    fft_signal_1 = np.abs(np.fft.fft(signal_brain))
    #fft_signal_1 = np.abs(np.fft.ifft(fft_signal_1))

    # plot the fourier transform against 600Hz
    

    sample_rate = 600 # in Hz
    sample_interval = 1 / sample_rate
    freq = np.fft.fftfreq(np.size(fft_signal_1), sample_interval)
    
    ax[1].plot(freq, fft_signal_1, 'r-')
    
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_title('Unprocessed Signal '+ str(signal_brain_num))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power strectrum')
    ax[1].set_title('Fourier Transformed Signal')
    plt.savefig('Task 3 Images/signal_figure.jpeg')
    plt.show()

def find_baseline(signal_brain_num, last_0_5_secs):
    """_summary_

    Args:
        signal_brain_num (int): 1 or 2, decribes which file to extract
        last_0_5_secs (boolan): Tells the program to only plot the last 0.5s of the 1st task
    """
    
    #------------------------------------------------------------------------
    #--------------------- Fourier Transform ---------------------
    #------------------------------------------------------------------------

    #extract signals from either signal_brain.mat or signal_brain2.mat)
    if signal_brain_num == 1:
        signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain.mat")[0:3900]
        if last_0_5_secs == True:
            signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain.mat")[3600:3900]
        # for one repeat (shouldn't need to do FT of each 6.5 sec repeat and stack because the total FT does the same thing)
    
    if signal_brain_num == 2:
        signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain2.mat")[0:3900]
        if last_0_5_secs == True:
            signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain2.mat")[3600:3900]

    # The 1st task takes 6.5s and the signal_brain array is 370500 long => 1st tasks takes place within 3900 values
    
    # From the 6.5s figure, the task appears to end at 6.0s and the signal for the last 0.5s seems to be the background noise
    
    # The last 0.5s is 300 array values of signal_brain
    # from inspection, the peak frequency of the background noise is from 14Hz to 22Hz
    

    # plot the signal against 6.5s of time 
    t = np.linspace(0, 6.5 ,np.size(signal_brain))
    if last_0_5_secs == True:
        t = np.linspace(0, 0.5 ,np.size(signal_brain))
    
    # create subplots
    fig, ax = plt.subplots(3, figsize=(15, 8))
    ax[0].plot(t, signal_brain, 'b-')
    
    # calculate the fourier transfrom
    fft_signal_1 = np.abs(np.fft.fft(signal_brain))

    # plot the fourier transform against 600Hz
    
    # tried using np.linsapce
    #freq = np.linspace(0, 600, np.size(fft_signal_1))

    sample_rate = 600 # in Hz
    sample_interval = 1 / sample_rate

    freq = np.fft.fftfreq(np.size(fft_signal_1), sample_interval)
    
    ax[1].plot(freq, fft_signal_1, 'r-')

    #------------------------------------------------------------------------
    #--------------------- Frequency Filter ---------------------
    #------------------------------------------------------------------------

    x = freq
    #mu_1 = 5 for 0-10s
    mu_1 = 18
    sig = 0.5 # sigma regluates the half width full maximum nik says <3 <3
    guassian = np.exp(-np.power(x - mu_1, 2.) / (2 * np.power(sig, 2.)))

    filtered_signal = fft_signal_1 * fft_signal_1
    inverse_filtered_signal = np.fft.ifft(filtered_signal)
    
    ax[2].plot(t, inverse_filtered_signal, 'b-')

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_title('Unprocessed Signal '+ str(signal_brain_num))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power strectrum')
    ax[1].set_title('Fourier Transformed Signal')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Frequency (Hz)')
    ax[2].set_title('Filtered Signal '+ str(signal_brain_num))
    if last_0_5_secs == False:
        plt.savefig('Task 3 Images/signal_figure_6.5.jpeg')
    if last_0_5_secs == True:
        plt.savefig('Task 3 Images/signal_figure_last_0.5s.jpeg')
        
    fig.subplots_adjust(hspace=0.6)
    plt.show()

#time_freq_decom(signal_brain_num=1)
find_baseline(signal_brain_num=1, last_0_5_secs=True)