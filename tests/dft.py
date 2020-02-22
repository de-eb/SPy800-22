# dft.py

import math
import numpy as np
from scipy import fftpack


def dft(bits):
    """
    Discrete fourier transform (spectral) Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    percentile : float
        Probability that peaks after DFT are below the reference.
    n : list of int
        n[0] is expected theoretical number of peaks.
        n[1] is the number of observed peaks below the reference.
    d : float
        Numerator of the argument of the complementary error function.
    """
    if bits.size & bits.size -1 != 0:
        pad = 2**math.ceil(math.log2(bits.size)) - bits.size
        bits = np.pad(bits, [math.floor(pad/2), math.ceil(pad/2)])
    bits = 2*bits - 1

    s = fftpack.fft(bits)
    m = np.abs(s[:bits.size // 2])
    t = math.sqrt(math.log(1./0.05)*bits.size)  # upper bound
    n = [0., 0.]
    n[0] = 0.95*bits.size / 2.
    n[1] = np.count_nonzero(m < t)
    percentile = 100*n[1] / (bits.size/2)
    d = (n[1] - n[0]) / math.sqrt(bits.size*0.95*0.05/4.)

    p_value = math.erfc(abs(d)/math.sqrt(2.))

    return p_value, percentile, n, d


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=1024)
    results = dft(bits)
    print("\np-value = {}\n".format(results[0]))