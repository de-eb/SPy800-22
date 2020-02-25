# complexity.py

import numpy as np
from scipy.special import gammaincc


def bma(bits):
    """
    Berlekamp Massey Algorithm using NumPy
    """
    c, b = np.zeros(bits.size, dtype=int), np.zeros(bits.size, dtype=int)
    c[0], b[0] = 1, 1
    l, m, i = 0, -1, 0
    for i in range(bits.size):
        if (bits[i] + np.dot(bits[i-l:i][::-1], c[1:1+l])) % 2 == 1:
            t = c.copy()
            c[np.where(b[:l]==1)[0] + i-m] += 1
            c = c % 2
            if l <= i>>1:
                l = i + 1 - l
                m = i
                b = t
    return l

def complexity(bits, blk_size=500):
    """
    Linear complexity Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    blk_size : int
        Sequence length (bits) after being divided into blocks.
    
    Returns
    -------
    p_value : float
        Test result.
    chi_square : float
        Computed statistics.
    """
    blk_num = bits.size // blk_size
    blk = np.resize(bits, (blk_num, blk_size))
    mu = blk_size/2 + ((-1)**(blk_size+1)+9)/36 - (blk_size/3+2/9)/2**blk_size
    pi = np.array([0.01047,0.03125,0.12500,0.50000,0.25000,0.06250,0.020833])

    l = np.empty(blk_num)
    for i in range(blk_num):
        l[i] = bma(blk[i])
    t = (-1)**blk_size * (l-mu) + 2/9
    hist = np.histogram(
        t, bins=[-bits.size, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, bits.size])[0]
    chi_square = np.sum((hist-blk_num*pi)**2 / (blk_num*pi))

    p_value = gammaincc(6/2.0 , chi_square/2.0)

    return p_value, chi_square


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = complexity(bits)
    print("\np-value = {}\n".format(results[0]))
