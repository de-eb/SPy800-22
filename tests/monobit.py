# monobit.py


import math
import numpy as np


def monobit(bits):
    """ Frequency (monobit) Test

    Parameters
    ----------
    bits (ndarray, uint8, 1d) : Binary sequence to be tested.
    
    Returns
    -------
    p_value (float) : Test result.
    ones (int) : Number of 1s in the sequence.
    zeros (int) : Number of 0s in the sequence.
    """
    ones = np.count_nonzero(bits)
    zeros = bits.size - ones

    s_obs = abs(ones-zeros) / math.sqrt(bits.size)
    f = s_obs / math.sqrt(2.0)

    p_value = math.erfc(f)

    return p_value, ones, zeros


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=1024)
    results = monobit(bits)
    print("\np-value = {}\n".format(results[0]))
