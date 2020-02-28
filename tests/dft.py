# dft.py

import numpy as np
from scipy.fftpack import fft
from math import sqrt, log, erfc


def dft(bits):
    """
    Discrete Fourier Transform (Spectral) Test
    ------------------------------------------
    Evaluate the peak heights in DFT of the sequence.

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    percentile : float
        Percentage of DFT peaks below threshold.
    """
    ref = 0.95*bits.size / 2
    bits = 2*bits - 1
    magnitude = np.abs(fft(bits)[:bits.size // 2])
    threshold = sqrt(log(1/0.05)*bits.size)
    count = np.count_nonzero(magnitude < threshold)
    percentile = 100*count / (bits.size/2)

    p_value = erfc(abs((count-ref)/sqrt(bits.size*0.95*0.05/4)) / sqrt(2))

    return p_value, percentile


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=10**6)
    results = dft(bits)
    print("\np-value = {}\n".format(results[0]))
