# frequency.py

import numpy as np
from scipy.special import gammaincc


def frequency(bits, block_size=128):
    """
    Frequency Test within a block

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    block_size : int
        Sequence length after being divided into blocks.
    
    Returns
    -------
    p_value : float
        Test result.
    chi_squared : float
        Computed statistics.
    """
    block_num, mod = divmod(bits.size, block_size)  # Number of blocks
    
    blocks = np.resize(bits, (block_num, block_size))
    block_sums = np.sum(blocks, axis=1)
    v = block_sums / block_size - 0.5
    s = np.sum(np.power(v,2))
    chi_squared = 4.0 * block_size * s

    p_value = gammaincc(block_num/2.0 , chi_squared/2.0)
        
    return p_value, chi_squared


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=1000)
    results = frequency(bits)
    print("\np-value = {}\n".format(results[0]))
