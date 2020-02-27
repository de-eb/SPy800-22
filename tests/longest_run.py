# longest_run.py

import numpy as np
from scipy.special import gammaincc


def longest_run(bits):
    """
    Test for the Longest Run of Ones in a Block
    -------------------------------------------
    Evaluate the longest "Run" of 1s for each M-bit subsequence,
    where a "Run" is a continuation of the same bit.

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    chi_square : float
        Computed statistic.
    hist : 1d-ndarray
        Histogram of the longest "Run" in each block.
    """
    if bits.size < 128:
        return 0.0, None, None
    elif bits.size < 6272:
        blk_size = 8
        v = np.arange(1, 5)
        pi = np.array([0.21484375, 0.3671875, 0.23046875, 0.1875])
    elif bits.size < 750000:
        blk_size = 128
        v = np.arange(4, 10)
        pi = np.array([ 0.1174035788, 0.242955959, 0.249363483,
                        0.17517706,   0.102701071, 0.112398847])
    else:
        blk_size = 10000
        v = np.arange(10, 17)
        pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    
    blk_num = bits.size // blk_size
    blk = np.pad(np.resize(bits, (blk_num, blk_size)), [(0,0),(1,1)])
    longest = np.zeros(blk_num, dtype=int)
    for i in range(blk_num):
        longest[i] = np.max(np.diff(np.where(blk[i]==0)[0]) - 1)
    longest[longest < v[0]] = v[0]
    longest[longest > v[-1]] = v[-1]
    hist = np.histogram(longest, bins=v.size, range=(v[0], v[-1]+1))[0]
    chi_square = np.sum((hist-blk_num*pi)**2 / (blk_num*pi))
    
    p_value = gammaincc((v.size-1)/2 , chi_square/2)

    return p_value, chi_square, hist


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=10**6)
    results = longest_run(bits)
    print("\np-value = {}\n".format(results[0]))
