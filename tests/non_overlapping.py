# non_overlapping.py

import numpy as np
import cv2
from scipy.special import gammaincc


def non_overlapping(bits, tpl_size=9):
    """
    Non-overlapping template matching Test

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
    matches : 2d-ndarray
        Number of matches in each block for each template.
    """
    blk_num = 8
    blk_size = bits.size // blk_num
    mean = (blk_size - tpl_size + 1) / 2**tpl_size
    var = blk_size*(1/2**tpl_size - (2*tpl_size-1)/2**(2*tpl_size))
    if var <= 0:
        return None, None, None

    tpl = np.load("tests/templates/tpl{}.npy".format(tpl_size))
    bits = np.resize(bits, (blk_num, blk_size)).astype('uint8')
    matches = np.zeros((tpl.shape[0], blk_num))
    chi_square = np.zeros(tpl.shape[0])
    p_value = np.zeros(tpl.shape[0])

    for i in range(tpl.shape[0]):
        res = cv2.matchTemplate(bits, tpl[i].reshape((1,-1)), cv2.TM_SQDIFF)
        matches[i] = np.count_nonzero(res <= 0.5, axis=1)
        chi_square[i] = np.sum(((matches[i] - mean)/var**0.5)**2)  # why?

        p_value[i] = gammaincc(blk_num/2.0 , chi_square[i]/2.0)

    return p_value, chi_square, matches


if __name__ == "__main__":

    bits = np.random.randint(0, 2, size=2**20)
    results = non_overlapping(bits)
    print()
    for i, v in enumerate(results[0]):
        print("p-value of template{} = {}".format(i,v))
    print()
