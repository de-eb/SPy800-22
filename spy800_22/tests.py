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

from spy800_22.sts import STS, InvalidSettingError, StatisticalError
import numpy as np
from math import sqrt, exp, erfc
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
            ig_err: bool =False, init: bool =True) -> None:
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
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.
        
        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)

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
        msg += "\nSequenceID,p-value,zeros count,ones count\n"
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1], j[2])
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
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
            proc_num: int =1, ig_err: bool =False, init: bool =True) -> None:
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
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if seq_len < blk_len:
            msg = "The sequence length must be larger than the block length."
            raise InvalidSettingError(msg)
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
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
            msg += "{},{},{}".format(i, j[0], j[1])
            if len(j) > 2:
                msg += ",{}".format(j[2])
            msg += "\n"
        return msg


class RunsTest(STS):
    """Runs Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.RUNS
    NAME = "Runs Test"

    def __init__(self, seq_len: int, seq_num: int, proc_num: int =1,
            ig_err: bool =False, init: bool =True) -> None:
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
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        
    def func(self, bits: np.ndarray) -> tuple:
        """Evaluate the total number of "Run"s for the entire sequence,
        where a "Run" is a continuation of the same bit.

        Parameters
        ----------
        bits : `1d-ndarray uint8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        pi : `float`
            Estimator criteria.
        run : `int`
            Total of "Run"s.
        
        """
        pi = np.count_nonzero(bits) / bits.size
        if abs(pi-0.5) > 2/sqrt(bits.size):
            msg = "Estimator criteria not met. (Pi = {})".format(pi)
            raise StatisticalError(msg, 0.0, pi, None)
        run = np.count_nonzero(np.diff(bits))
        p_value = erfc(
            abs(run-2*bits.size*pi*(1-pi)) / (2*pi*(1-pi)*sqrt(2*bits.size)))
        return p_value, pi, run

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
        msg = RunsTest.NAME + "\n"
        msg += "\nSequenceID,p-value,pi,Runs total\n"
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1], j[2])
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg


class LongestRunOfOnesTest(STS):
    """Test for the Longest Run of Ones in a Block

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.LONGESTRUN
    NAME = "Test for the Longest Run of Ones in a Block"

    def __init__(self, seq_len: int, seq_num: int, proc_num: int =1,
            ig_err: bool =False, init: bool =True) -> None:
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
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if seq_len < 128:
            msg = "The sequence length must be at least 128 bits."
            raise InvalidSettingError(msg)
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        if seq_len < 6272:
            self.__blk_len = 8
            self.__v = np.arange(1, 5)
            self.__pi = np.array([0.21484375, 0.3671875, 0.23046875, 0.1875])
        elif seq_len < 750000:
            self.__blk_len = 128
            self.__v = np.arange(4, 10)
            self.__pi = np.array([0.1174035788, 0.242955959, 0.249363483,
                                  0.17517706,   0.102701071, 0.112398847])
        else:
            self.__blk_len = 10000
            self.__v = np.arange(10, 17)
            self.__pi = np.array(
                [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
        self.__blk_num = seq_len // self.__blk_len
    
    @property
    def blk_len(self):
        """`int`: Bit length of each block."""
        return self.__blk_len
    
    @property
    def blk_num(self):
        """`int`: Number of blocks."""
        return self.__blk_num
        
    def func(self, bits: np.ndarray) -> tuple:
        """Evaluate the longest "Run" of 1s for each M-bit subsequence,
        where a "Run" is a continuation of the same bit.

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
        hist : `1d-ndarray`
            Histogram of the longest "Run" in each block.
        
        """
        blk = np.pad(
            np.resize(bits, (self.__blk_num, self.__blk_len)), [(0,0),(1,1)])
        longest = np.zeros(self.__blk_num, dtype=int)
        for i in range(self.__blk_num):
            longest[i] = np.max(np.diff(np.where(blk[i]==0)[0]) - 1)
        longest[longest < self.__v[0]] = self.__v[0]
        longest[longest > self.__v[-1]] = self.__v[-1]
        hist = np.histogram(longest,
            bins=self.__v.size, range=(self.__v[0], self.__v[-1]+1))[0]
        chi_square = np.sum(
            (hist-self.__blk_num*self.__pi)**2 / (self.__blk_num*self.__pi))
        p_value = gammaincc((self.__v.size-1)/2 , chi_square/2)
        return p_value, chi_square, hist

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
        msg = LongestRunOfOnesTest.NAME + "\n"
        msg += "\nBlock length,{}\n".format(self.__blk_len)
        msg += "Number of blocks,{}\n".format(self.__blk_num)
        msg += "\nSequenceID,p-value,chi_square,Histogram of longest Run\n"
        msg += (",,,{}\n".format(np.array2string(self.__v, separator=',')
            .replace('[','').replace(']','')))
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1], np.array2string(
                j[2], separator=',').replace('[','').replace(']',''))
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg


class BinaryMatrixRankTest(STS):
    """Binary Matrix Rank Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.RANK
    NAME = "Binary Matrix Rank Test"

    def __init__(self, seq_len: int, seq_num: int, proc_num: int =1,
            ig_err: bool =False, init: bool =True) -> None:
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
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        self.__m, self.__q = 32, 32
        self.__mat_num = seq_len // (self.__m * self.__q)
        if self.__mat_num == 0:
            msg = ("The sequence length must be at least {} bits."
                .format(self.__m * self.__q))
            raise InvalidSettingError(msg)
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        self.__prob = np.zeros(3)
        for i, r in enumerate([self.__m, self.__m-1]):
            j = np.arange(r, dtype=float)
            prod = np.prod(
                (1-2**(j-self.__m)) * (1-2**(j-self.__q)) / (1-2**(j-r)))
            self.__prob[i] = (
                2**(r*(self.__m + self.__q - r) - self.__m*self.__q) * prod)
        self.__prob[2] = 1 - (self.__prob[0] + self.__prob[1])
    
    @property
    def mat_shape(self):
        """`tuple`: Matrix shape used for Rank calculation."""
        return self.__m, self.__q
    
    @property
    def mat_num(self):
        """`int`: Number of matrices."""
        return self.__mat_num
        
    def func(self, bits: np.ndarray) -> tuple:
        """Evaluate the Rank of disjoint sub-matrices of the entire sequence.

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
        freq : `1d-ndarray`
            Number of occurrences of M and M-1 and other Ranks,
            where M is the length of a matrix row or column.
        
        """
        freq = np.zeros(3, dtype=int)
        rank = np.zeros(self.__mat_num, dtype=int)
        blk = np.resize(bits, (self.__mat_num, self.__m*self.__q))
        for i in range(self.__mat_num):
            rank[i] = self.__matrix_rank(blk[i].reshape((self.__m,self.__q)))
        freq[0] = np.count_nonzero(rank == self.__m)
        freq[1] = np.count_nonzero(rank == self.__m-1)
        freq[2] = self.__mat_num - (freq[0] + freq[1])
        chi_square = np.sum((freq-self.__mat_num*self.__prob)**2
                            / (self.__mat_num*self.__prob))
        p_value = exp(-chi_square/2)
        return p_value, chi_square, freq

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
        msg = BinaryMatrixRankTest.NAME + "\n"
        msg += "\nMatrix shape,{},{}\n".format(self.__m, self.__q)
        msg += "Number of matrices,{}\n".format(self.__mat_num)
        msg += "\n,,Reference probability,{}\n".format(np.array2string(
            self.__prob, separator=',').replace('[','').replace(']',''))
        msg += "SequenceID,p-value,chi_square,Histogram of Rank\n"
        msg += ",,,{},{},other\n".format(self.__m, self.__m-1)
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1], np.array2string(
                j[2], separator=',').replace('[','').replace(']',''))
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg
    
    def __matrix_rank(self, mat):
        """Calculate Rank by elementary row operations.

        Implementation to avoid using `numpy.linalg.matrix_rank`.

        Parameters
        ----------
        mat : `2d-ndarray int`
            Binary matrix to be calculated.
        
        Returns
        -------
        rank : `int`
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
    