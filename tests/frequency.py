# frequency.py

import numpy as np
from scipy.special import gammaincc
from fractions import Fraction


def frequency(bits, block_size):
    """ Frequency Test within a block

    Parameters
    ----------
    bits (ndarray, uint8, 1d) : Binary sequence to be tested.
    block_size (int) : Sequence length after being divided into blocks.
    
    Returns
    -------
    p_value (float) : Test result.
    chi_squared (float) : Computed statistics.
    """
    block_num, mod = divmod(bits.size, block_size)  # Number of blocks
    
    s = Fraction(0,1)
    for i in range(block_num):
        block_sum = np.count_nonzero(bits[i*block_size : (i+1)*block_size])
        pi = Fraction(block_sum, block_size)
        v = pi - Fraction(1,2)
        s += v**2
    chi_squared = 4.0 * block_size * s

    p_value = gammaincc(block_num/2.0 , chi_squared/2.0)
        
    return (p_value, chi_squared)
