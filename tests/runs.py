# runs.py

import math
import numpy as np


def runs(bits):
    """
    Runs Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    erfc_arg : float
        Argment of complementary error function.
    vobs : int
        Total of Runs.
    pi : float
        Ratio of 1 to all bits.
    """
    pi = np.count_nonzero(bits) / bits.size

    if abs(pi - 0.5) > (2.0 / math.sqrt(bits.size)):
        return (0.0, None, None, pi)
    
    vobs = np.count_nonzero(np.diff(bits))
    erfc_arg = (abs(vobs - 2.0*bits.size*pi*(1-pi))
                / (2.0*pi*(1-pi)*math.sqrt(2*bits.size)))

    p_value = math.erfc(erfc_arg)

    return p_value, erfc_arg, vobs, pi


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=1000)
    results = runs(bits)
    print("\np-value = {}\n".format(results[0]))
