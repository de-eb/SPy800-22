# monobit.py

import numpy as np
from math import sqrt, erfc


def monobit(bits):
    """
    Frequency (Monobit) Test.
    Evaluate the proportion of 0s and 1s for the entire sequence.

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    counts : list of int
        Number of occurrences of 0s and 1s for the entire sequence.
    """
    ones = np.count_nonzero(bits)
    zeros = bits.size - ones

    p_value = erfc((abs(ones-zeros)/sqrt(bits.size))/sqrt(2))

    return p_value, [zeros, ones]


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=10**6)
    results = monobit(bits)
    print("\np-value = {}\n".format(results[0]))
