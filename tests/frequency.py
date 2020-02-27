# frequency.py

import numpy as np
from scipy.special import gammaincc


def frequency(bits, blk_size=128):
    """
    Frequency Test within a Block.
    Evaluate the uniformity of 0s and 1s for each M-bit subsequence.

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    blk_size : int
        Bit length of each block.
    
    Returns
    -------
    p_value : float
        Test result.
    chi_square : float
        Computed statistic.
    """
    blk_num = bits.size // blk_size
    blk = np.resize(bits, (blk_num, blk_size))
    sigma = np.sum((np.sum(blk, axis=1)/blk_size - 0.5)**2)
    chi_square = 4 * blk_size * sigma

    p_value = gammaincc(blk_num/2 , chi_square/2)
        
    return p_value, chi_square


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=10**6)
    results = frequency(bits)
    print("\np-value = {}\n".format(results[0]))
