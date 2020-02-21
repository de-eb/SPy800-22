# longest_run.py

import numpy as np
from scipy.special import gammaincc


def longest_run(bits):
    """
    Test for the Longest Run of Ones in a Block

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    chi_square : float
        Computed statistics.
    hist : 1d-ndarray
        Histogram of the longest Runs in each block.
    """
    if bits.size < 128:
        return 0.0, None, None
    elif bits.size < 6272:
        block_size = 8
        V = np.arange(1,5)  # 1,2,3,4,5
        pi = np.array([0.21484375, 0.3671875, 0.23046875, 0.1875])
    elif bits.size < 750000:
        block_size = 128
        V = np.arange(4, 10)  # 4,5,6,7,8,9
        pi = np.array([ 0.1174035788, 0.242955959, 0.249363483,
                        0.17517706,   0.102701071, 0.112398847])
    else:
        block_size = 10000
        V = np.arange(10, 17)  # 10,11,12,13,14,15,16
        pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
    
    block_num = bits.size // block_size
    blocks = np.pad(np.resize(bits, (block_num, block_size)), [(0,0),(1,1)])

    longest = np.zeros(block_num)
    for i in range(block_num):
        runs = np.diff(np.where(blocks[i]==0)[0]) - 1
        longest[i] = np.max(runs)
    longest[longest < V[0]] = V[0]
    longest[longest > V[-1]] = V[-1]
    hist = np.histogram(longest, bins=V.size, range=(V[0], V[-1]+1))[0]
    chi_square = np.sum(np.power((hist - block_num*pi), 2) / (block_num*pi))
    
    p_value = gammaincc((V.size - 1)/2.0 , chi_square/2.0)

    return p_value, chi_square, hist


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=1000)
    results = longest_run(bits)
    print("\np-value = {}\n".format(results[0]))
