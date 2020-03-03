# rank.py

import numpy as np
from math import exp


def matrix_rank(mat):
    """
    Calculate Rank by elementary row operations.
    Implementation to avoid using "numpy.linalg.matrix_rank()".

    Parameters
    ----------
    mat : 2d-ndarray int
        Binary matrix to be calculated.
    
    Returns
    -------
    rank : int
        Rank of the matrix.
    """
    i, j, k = 0, 0, 0
    for _ in range(mat.shape[1]):
        ref = np.nonzero(mat[j:mat.shape[1],k])[0]
        if ref.size != 0:
            i = ref[0] + j
            if i != j:
                mat[[i,j]] = mat[[j,i]]
            mat[np.nonzero(mat[j+1:mat.shape[0],k])[0]+j+1] ^= mat[j]
            j += 1
            k += 1
        else:
            k += 1
    rank = np.count_nonzero(np.count_nonzero(mat, axis=1))
    return rank

def rank(bits, shape=(32,32)):
    """
    Binary Matrix Rank Test
    -----------------------
    Evaluate the rank of disjoint sub-matrices of the entire sequence.

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
        Computed statistic.
    freq : 1d-ndarray
        Number of occurrences of M and M-1 and other Ranks,
        where M is the length of a matrix row or column.
    """
    m, q = shape[0], shape[1]
    blk_num = bits.size // (m*q)
    if blk_num == 0:
        return 0.0, None, None

    prob = np.zeros(3)
    for idx, r in enumerate([m, m-1]):
        j = np.arange(r, dtype=float)
        prod = np.prod((1-2**(j-m)) * (1-2**(j-q)) / (1-2**(j-r)))
        prob[idx] = 2**(r*(m+q-r) - m*q) * prod
    prob[2] = 1 - (prob[0] + prob[1])

    freq = np.zeros(3, dtype=int)
    rank = np.zeros(blk_num, dtype=int)
    blk = np.resize(bits, (blk_num, m*q))
    for i in range(blk_num):
        rank[i] = matrix_rank(blk[i].reshape((m,q)))
    freq[:2] = np.count_nonzero(rank == m), np.count_nonzero(rank == m-1)
    freq[2] = blk_num - (freq[0] + freq[1])
    chi_square = np.sum((freq-blk_num*prob)**2 / (blk_num*prob))

    p_value = exp(-chi_square/2)

    return p_value, chi_square, freq


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=10**6)
    results = rank(bits)
    print("\np-value = {}\n".format(results[0]))
