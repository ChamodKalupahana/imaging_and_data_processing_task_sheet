import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sci

def time_freq_decom():
    signal_brain = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain.mat")
    signal_brain_2 = np.loadtxt(r"Task sheet files-20221008\Fourier_Filtering\Fourier_Filtering\signal_brain2.mat")

    t = np.arange(0, np.size(signal_brain))

    fig, ax = plt.subplots(2, figsize=(15, 8))

    ax[0].plot(t, signal_brain, 'b-')
    ax[1].plot(t, signal_brain_2, 'r-')

    plt.show()

time_freq_decom()