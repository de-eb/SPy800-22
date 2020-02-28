# universal.py

import numpy as np
from math import sqrt, erfc


def packbits(x, reverse=True):
    """
    Converts a binary matrix to a decimal value.
    Implementation to avoid using "numpy.packbits()".

    Parameters
    ----------
    x : 1d-ndarray int
        Binary matrix to be converted.
    reverse : bool
        Conversion mode. If true, little endian. Default is big endian.
    """
    p = np.power(2, np.arange(x.shape[-1]))
    if reverse:
        p = p[::-1]
    return np.dot(x, p)

def universal(bits):
    """
    Maurer's "Universal Statistical" Test
    -------------------------------------
    Evaluate the distance (bits) between L-bit patterns
    repeatedly observed in the sequence.

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    
    Returns
    -------
    p_value : float
        Test result.
    phi : float
        Computed statistic.
    """
    blk_size = 5
    thresh = [387840, 904960, 2068480, 4654080, 10342400, 22753280,
                49643520, 107560960, 231669760, 496435200, 1059061760]
    for i in thresh:
        if bits.size >= i:
            blk_size += 1
    if blk_size < 6:
        return 0.0, None
    var = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311,
            3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
    expected = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507,
                7.1836656, 8.1764248, 9.1723243, 10.170032,
                11.168765, 12.168070, 13.167693, 14.167488, 15.167379]
    q = 10 * 2**blk_size
    k = bits.size // blk_size - q
    sigma = ((0.7 - 0.8/blk_size + (4+32/blk_size)/15 * k**(-3/blk_size))
            * sqrt(var[blk_size]/k))

    t = np.zeros(2**blk_size)
    blk = packbits(np.resize(bits, (k+q, blk_size)))
    uniq, idx = np.unique(blk[:q][::-1], return_index=True)
    t[uniq] = idx
    s = 0
    for i in range(t.size):
        s += np.sum(np.log2(np.diff(np.append(-t[i], np.where(blk[q:]==i)))))
    phi = s / k
    
    p_value = erfc(abs(phi-expected[blk_size]) / (sqrt(2)*sigma))

    return p_value, phi


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = universal(bits)
    print("\np-value = {}\n".format(results[0]))
