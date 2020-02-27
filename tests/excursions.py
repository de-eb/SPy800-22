# excursions.py

import numpy as np
from scipy.special import gammaincc

def excursions(bits):
    """
    Random excursions Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : 1d-ndarray float
        Test results.
    chi_square : 1d-ndarray float
        Computed statistics.
    """
    pi = np.array(
        [[0.8750000000,  0.8333333333,  0.7500000000,  0.5000000000],
         [0.01562500000, 0.02777777778, 0.06250000000, 0.25000000000],
         [0.01367187500, 0.02314814815, 0.04687500000, 0.12500000000],
         [0.01196289063, 0.01929012346, 0.03515625000, 0.06250000000],
         [0.01046752930, 0.01607510288, 0.02636718750, 0.03125000000],
         [0.0732727051,  0.0803755143,  0.0791015625,  0.0312500000]])
    pi = np.hstack((pi, np.fliplr(pi)))
    stat = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    
    bits = 2*bits - 1
    s = np.pad(np.cumsum(bits), (1,1))
    idx = np.where(s==0)[0]
    cycle = idx.size - 1
    
    hist = np.zeros((cycle, stat.size), dtype=int)
    freq = np.zeros((6, stat.size), dtype=int)
    for i in range(cycle):
        hist[i] = np.histogram(
            s[idx[i]:idx[i+1]], bins=9, range=(-4,5))[0][stat+4]
    for i in range(6):
        freq[i] = np.count_nonzero(hist==i, axis=0)
    freq[i] = np.count_nonzero(hist>i, axis=0)
    chi_square = np.sum((freq-cycle*pi)**2/(cycle*pi), axis=0)

    p_value = gammaincc(2.5 , chi_square/2.0)
    
    return p_value, chi_square


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = excursions(bits)
    for i, s in enumerate([-4, -3, -2, -1, 1, 2, 3, 4]):
        print("p-value ({:2}) = {}".format(s, results[0][i]))
    print()
