# runs.py

import numpy as np
from math import sqrt, erfc


def runs(bits):
    """
    Runs Test.
    Evaluate the total number of "Run"s for the entire sequence,
    where a "Run" is a continuation of the same bit.

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    run : int
        Total of "Run"s.
    """
    pi = np.count_nonzero(bits) / bits.size
    if abs(pi-0.5) > 2/sqrt(bits.size):
        return 0.0, None
    run = np.count_nonzero(np.diff(bits))

    p_value = erfc(
        abs(run-2*bits.size*pi*(1-pi)) / (2*pi*(1-pi)*sqrt(2*bits.size)))

    return p_value, run


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=10**6)
    results = runs(bits)
    print("\np-value = {}\n".format(results[0]))
