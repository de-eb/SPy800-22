# entropy.py

import math
import numpy as np
from scipy.special import gammaincc


def phi_m(x, m):
    p, k = np.zeros(2**(m+1)-1), np.ones(x.size, dtype=int)
    j = np.arange(x.size)
    for i in range(m):
        k *= 2
        k[x[(i+j) % x.size] == 1] += 1
    uniq, counts = np.unique(k, return_counts=True)
    p[uniq-1] = 1*counts
    ref = p[2**m-1 : 2**(m+1)-1]
    ref = ref[np.nonzero(ref)[0]]
    s = np.sum(ref*np.log(ref/x.size)) / x.size
    return s

def entropy(bits, blk_size=10):
    """
    Approximate entropy Test

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
    apen = phi_m(bits, blk_size) - phi_m(bits, blk_size+1)
    chi_square = 2*bits.size*(math.log(2) - apen)

    p_value = gammaincc(2**(blk_size-1), chi_square/2.0)
    
    return p_value, chi_square


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = entropy(bits)
    print("\np-value = {}\n".format(results[0]))
