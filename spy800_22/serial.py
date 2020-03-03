# serial.py

import numpy as np
from scipy.special import gammaincc


def psi_square(x, m):
    p, k = np.zeros(2**(m+1)-1), np.ones(x.size, dtype=int)
    j = np.arange(x.size)
    for i in range(m):
        ref = x[(i+j) % x.size]
        k[ref==0] *= 2
        k[ref==1] = 2*k[ref==1] + 1
    uniq, counts = np.unique(k, return_counts=True)
    p[uniq-1] = 1*counts
    s = np.sum(p[2**m-1 : 2**(m+1)-1]**2) * 2**m/x.size - x.size
    return s

def serial(bits, blk_size=16):
    """
    Serial Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    blk_size : int
        Sequence length (bits) after being divided into blocks.
    
    Returns
    -------
    p_value : list of float
        Test results.
    psi : list of float
        Computed statistics.
    """
    psi = [0., 0., 0.]
    p_value = [0., 0.]

    for i in range(len(psi)):
        psi[i] = psi_square(bits, blk_size-i)
    
    p_value[0] = gammaincc(2**(blk_size-1)/2.0, (psi[0]-psi[1])/2.0)
    p_value[1] = gammaincc(2**(blk_size-2)/2.0, (psi[0]-2*psi[1]+psi[2])/2.0)

    return p_value, psi


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = serial(bits)
    print("\np-values = {}, {}\n".format(results[0][0], results[0][1]))
