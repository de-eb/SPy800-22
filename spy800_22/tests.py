#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
"""Implementation of SP800-22 test algorithms by Python.

This module is part of the spy800_22 package and consists of 15 classes.
Each class corresponds to each test of NIST SP800-22.
These classes provide various functions (data I/O, parallel processing, etc.)
to execute each test by itself.

Details of NIST SP800-22:
https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

NIST's official implementation:
https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software

Notes
-----
    Test results may differ from those of the NIST's official implementation.
    This is due to all array operations being optimized for Numpy.

"""

from spy800_22.sts import STS, InvalidSettingError
import numpy as np
from math import sqrt, erfc
from scipy.special import gammaincc


class FrequencyTest(STS):
    """Frequency (Monobit) Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.FREQUENCY
    NAME = "Frequency (Monobit) Test"

    def __init__(self, seq_len: int, seq_num: int, proc_num: int =1,
            init: bool =True) -> None:
        """Set the test parameters.

        Parameters
        ----------
        seq_len : `int`
            Bit length of each split sequence.
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        init : `bool`
            If `True`, initialize the super class.
        
        """
        if init:
            super().__init__(seq_len, seq_num, proc_num)

    def func(self, bits: np.ndarray) -> tuple:
        """Evaluate the uniformity of 0s and 1s for the entire sequence.

        Parameters
        ----------
        bits : `1d-ndarray uint8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        zeros : `int`
            Number of occurrences of 0s for the entire sequence.
        ones : `int`
            Number of occurrences of 1s for the entire sequence.
        
        """
        ones = np.count_nonzero(bits)
        zeros = bits.size - ones
        p_value = erfc((abs(ones-zeros)/sqrt(bits.size))/sqrt(2))
        return p_value, zeros, ones

    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Called from the `save_report` method
        to support report generation for each test.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = FrequencyTest.NAME + "\n"
        msg += "\nSequenceID,p-value,zeros,ones\n"
        for i, j in enumerate(results):
            msg += "{},{},{},{}\n".format(i, j[0], j[1], j[2])
        return msg


class BlockFrequencyTest(STS):
    """Frequency Test within a Block

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.BLOCKFREQUENCY
    NAME = "Frequency Test within a Block"

    def __init__(self, seq_len: int, seq_num: int, blk_len: int =128,
            proc_num: int =1, init: bool =True) -> None:
        """Set the test parameters.

        Parameters
        ----------
        seq_len : `int`
            Bit length of each split sequence.
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        blk_len : `int`
            Bit length of each block.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        init : `bool`
            If `True`, initialize the super class.

        """
        if seq_len < blk_len:
            msg = "The sequence length must be larger than the block length."
            raise InvalidSettingError(msg)
        if init:
            super().__init__(seq_len, seq_num, proc_num)
        self.__blk_len = blk_len
        self.__blk_num = seq_len // blk_len
    
    @property
    def blk_len(self):
        """`int`: Bit length of each block."""
        return self.__blk_len
    
    @property
    def blk_num(self):
        """`int`: Number of blocks."""
        return self.__blk_num
    
    def func(self, bits: np.ndarray) -> tuple:
        """Evaluate the proportion of 1s for each M-bit subsequence.

        Parameters
        ----------
        bits : `1d-ndarray uint8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        chi_square : `float`
            Computed statistic.
        
        """
        blk = np.resize(bits, (self.__blk_num, self.__blk_len))
        sigma = np.sum((np.sum(blk, axis=1)/self.__blk_len - 0.5)**2)
        chi_square = 4 * self.__blk_len * sigma
        p_value = gammaincc(self.__blk_num/2 , chi_square/2)
        return p_value, chi_square
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Called from the `save_report` method
        to support report generation for each test.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = BlockFrequencyTest.NAME + "\n"
        msg += "\nBlock length,{}\n".format(self.__blk_len)
        msg += "Number of blocks,{}\n".format(self.__blk_num)
        msg += "\nSequenceID,p-value,chi_square\n"
        for i, j in enumerate(results):
            msg += "{},{},{}\n".format(i, j[0], j[1])
        return msg
