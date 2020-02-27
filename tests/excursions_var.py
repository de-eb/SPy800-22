# excursions_var.py

import numpy as np
from scipy.special import erfc

def excursions_var(bits):
    """
    Random excursions variant Test

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : 1d-ndarray float
        Test results.
    xi : 1d-ndarray float
        Number of occurrences of each state in the entire random walk.
    """
    stat = np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9])
    
    bits = 2*bits - 1
    s = np.pad(np.cumsum(bits), (1,1))
    idx = np.where(s==0)[0]
    cycle = idx.size - 1

    xi = np.zeros_like(stat)
    for i in range(stat.size):
        xi[i] = np.count_nonzero(s==stat[i])
    p_value = erfc(np.abs(xi-cycle)/np.sqrt(2*cycle*(4*np.abs(stat)-2)))
    
    return p_value, xi


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = excursions_var(bits)
    print()
    for i, s in enumerate([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9]):
        print("p-value ({:2}) = {}".format(s, results[0][i]))
    print()
