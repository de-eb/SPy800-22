# rank.py

import math
import numpy as np


def rank(bits, shape=(32,32)):
    """
    Binary matrix rank Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    shape : tuple of int
        Matrix shape used for Rank calculation.
    
    Returns
    -------
    p_value : float
        Test result.
    chi_square : float
        Computed statistics.
    prob : 1d-ndarray
        Estimated probability of occurrence of a particular rank.
    freq : 1d-ndarray
        Frequency of occurrence of a calculated particular rank.
    """
    m, q = shape[0], shape[1]
    block_num = bits.size // (m*q)
    if block_num == 0:
        return 0.0

    rs = [m, m-1]
    prob = np.zeros(3)
    for idx, r in enumerate(rs):
        i = np.arange(r, dtype=float)
        prod = np.prod((1 - 2**(i-m)) * (1 - 2**(i-q)) / (1 - 2**(i-r)))
        prob[idx] = 2**(r*(m+q-r) - m*q) * prod
    prob[2] = 1 - prob[0] + prob[1]

    freq = np.zeros(3)
    blocks = np.resize(bits, (block_num, m*q))
    for i in range(block_num):
        rnk = np.linalg.matrix_rank(blocks[i].reshape((m,q)))
        if rnk == m:
            freq[0] += 1
        if rnk == m - 1:
            freq[1] += 1
    freq[2] = block_num - freq[0] + freq[1]

    chi_square = np.sum((freq - block_num*prob)**2 / (block_num*prob))

    p_value = math.exp(-chi_square/2.0)

    return p_value, chi_square, prob, freq


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=1024)
    results = rank(bits)
    print("\np-value = {}\n".format(results[0]))