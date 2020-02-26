# cusum.py

import math
import numpy as np
from scipy.stats import norm


def sigma(n, z, term=False):
    a, b, c = 1, 1, -1
    if term:
        a, b, c = -3, 3, 1
    st = int((-n/z+1)//4)
    k = np.arange(int((-n/z+a)//4), int((n/z-1)//4)+1)
    s = np.sum(norm.cdf((4*k+b)*z/math.sqrt(n))
                - norm.cdf((4*k+c)*z/math.sqrt(n)))
    return s

def cusum(bits):
    """
    Cumulative sums (cusum) Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : list of float
        Test result.
    sums : list of int
        The largest absolute values of partial sums.
    """
    p_value = [0., 0.]
    sums = [0, 0]

    bits = 2*bits - 1
    sums[0] = np.max(np.abs(np.cumsum(bits)))
    p_value[0] = (1 - sigma(bits.size, sums[0])
                    + sigma(bits.size, sums[0], term=True))
    sums[1] = np.max(np.abs(np.cumsum(bits[::-1])))
    p_value[1] = (1 - sigma(bits.size, sums[1])
                    + sigma(bits.size, sums[1], term=True))
    
    return p_value, sums


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = cusum(bits)
    print("\np-value (forward) = {}".format(results[0][0]))
    print("p-value (reverse) = {}\n".format(results[0][1]))
