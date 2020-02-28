# non_overlapping.py

import numpy as np
import cv2
from scipy.special import gammaincc


def non_overlapping(bits, tpl_size=9):
    """
    Non-overlapping Template Matching Test
    --------------------------------------
    Evaluates the number of occurrences of templates 
    (particular m-bit patterns) for each M-bit subsequence.

    Parameters
    ----------
    bits : 1d-ndarray uint8
        Binary sequence to be tested.
    tpl_size : int
        Bit length of each template. Can be set from 2 to 16.
    
    Returns
    -------
    p_value : 1d-ndarray
        Test results for each template.
    chi_square : 1d-ndarray
        Computed statistics for each template.
    match : 2d-ndarray
        Number of matches in each block for each template.
    """
    blk_num = 8
    blk_size = bits.size // blk_num
    mean = (blk_size - tpl_size + 1) / 2**tpl_size
    var = blk_size*(1/2**tpl_size - (2*tpl_size-1)/2**(2*tpl_size))
    if mean <= 0:
        return 0.0, None, None

    tpl = np.load("tests/templates/tpl{}.npy".format(tpl_size))
    bits = np.resize(bits, (blk_num, blk_size)).astype('uint8')
    match = np.zeros((tpl.shape[0], blk_num))
    for i in range(tpl.shape[0]):
        res = cv2.matchTemplate(bits, tpl[i].reshape((1,-1)), cv2.TM_SQDIFF)
        match[i] = np.count_nonzero(res <= 0.5, axis=1)
    chi_square = np.sum(((match - mean)/var**0.5)**2, axis=1)  # why?
    p_value = gammaincc(blk_num/2 , chi_square/2)

    return p_value, chi_square, match


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = non_overlapping(bits)
    print()
    for i, v in enumerate(results[0]):
        print("p-value of template{:3} = {}".format(i,v))
    print()
