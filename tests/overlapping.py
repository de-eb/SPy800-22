# overlapping.py

import math
import numpy as np
import cv2
from scipy.special import gammaincc, loggamma


def overlapping(bits, tpl_size=9):
    """
    Overlapping template matching Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    tpl_size : int
        Bit length of template.
    
    Returns
    -------
    p_value : float
        Test result.
    chi_square : float
        Computed statistics.
    nu : 1d-ndarray
        Histogram of the number of template matches in each block.
    """
    k = 5
    blk_size = 1032
    blk_num = bits.size // blk_size
    lmd = (blk_size - tpl_size + 1) / 2**tpl_size
    eta = lmd/2.

    pi = np.zeros(k+1)
    pi[0] = math.exp(-eta)
    for i in range(1, k):
        pi[i] = 0.
        for j in range(1, i+1):
            pi[i] += math.exp(-eta - i*math.log(2) + j*math.log(eta)
                - loggamma(j+1) + loggamma(i) - loggamma(j) - loggamma(i-j+1))
    pi[k] = 1 - np.sum(pi)

    tpl = np.ones((1,tpl_size), dtype='uint8')
    blk = np.resize(bits, (blk_num, blk_size)).astype('uint8')
    nu = np.zeros(k+1)
    res = cv2.matchTemplate(blk, tpl, cv2.TM_SQDIFF)
    matches = np.count_nonzero(res <= 0.5, axis=1)
    for i in range(k):
        nu[i] = np.count_nonzero(np.logical_and(matches > i-1, matches <= i))
    nu[k] = np.count_nonzero(matches > i)
    chi_square = np.sum((nu - blk_num*pi)**2 / (blk_num*pi))

    p_value = gammaincc(k/2.0, chi_square/2.0)

    return p_value, chi_square, nu


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = overlapping(bits)
    print("\np-value = {}\n".format(results[0]))
