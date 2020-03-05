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
    This is due to all array operations being optimized for Python.

"""

import os
from math import sqrt, log, log2, exp, erfc
import numpy as np
import cv2
from scipy.fftpack import fft
from scipy.special import gammaincc, loggamma
from spy800_22.sts import STS, InvalidSettingError, StatisticalError


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

    def func(self, bits) -> tuple:
        """Evaluate the uniformity of 0s and 1s for the entire sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
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
        blk_len : `int`, optional
            Bit length of each block.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        if seq_len < blk_len:
            msg = ("Sequence length must be at least {} bits."
                .format(blk_len))
            raise InvalidSettingError(msg)
        self.__blk_len = blk_len
        self.__blk_num = seq_len // blk_len
    
    def func(self, bits) -> tuple:
        """Evaluate the proportion of 1s for each M-bit subsequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        chi_square : `float`
            Computed statistic.
        
        """
        bits = np.resize(bits, (self.__blk_num, self.__blk_len))
        sigma = np.sum((np.sum(bits, axis=1)/self.__blk_len - 0.5)**2)
        chi_square = 4 * self.__blk_len * sigma
        p_value = gammaincc(self.__blk_num/2 , chi_square/2)
        return p_value, chi_square
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

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
        
    def func(self, bits) -> tuple:
        """Evaluate the total number of "Run"s for the entire sequence,
        where a "Run" is a continuation of the same bit.

        Parameters
        ----------
        bits : `1d-ndarray int8`
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
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        if seq_len < 128:
            msg = "Sequence length must be at least 128 bits."
            raise InvalidSettingError(msg)
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
        
    def func(self, bits) -> tuple:
        """Evaluate the longest "Run" of 1s for each M-bit subsequence,
        where a "Run" is a continuation of the same bit.

        Parameters
        ----------
        bits : `1d-ndarray int8`
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
        bits = np.pad(
            np.resize(bits, (self.__blk_num, self.__blk_len)), [(0,0),(1,1)])
        longest = np.zeros(self.__blk_num, dtype=int)
        for i in range(self.__blk_num):
            longest[i] = np.max(np.diff(np.where(bits[i]==0)[0]) - 1)
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
        msg += (",,,<={}<=\n".format(np.array2string(self.__v, separator=',')))
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1],
                np.array2string(j[2], separator=','))
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg.replace('[','').replace(']','')


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
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        self.__m, self.__q = 32, 32
        self.__mat_num = seq_len // (self.__m * self.__q)
        if self.__mat_num == 0:
            msg = ("Sequence length must be at least {} bits."
                .format(self.__m * self.__q))
            raise InvalidSettingError(msg)
        self.__prob = np.zeros(3)
        for i, r in enumerate([self.__m, self.__m-1]):
            j = np.arange(r, dtype=float)
            prod = np.prod(
                (1-2**(j-self.__m)) * (1-2**(j-self.__q)) / (1-2**(j-r)))
            self.__prob[i] = (
                2**(r*(self.__m + self.__q - r) - self.__m*self.__q) * prod)
        self.__prob[2] = 1 - (self.__prob[0] + self.__prob[1])
        
    def func(self, bits) -> tuple:
        """Evaluate the Rank of disjoint sub-matrices of the entire sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
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
        bits = np.resize(bits, (self.__mat_num, self.__m*self.__q))
        for i in range(self.__mat_num):
            rank[i] = self.__matrix_rank(bits[i].reshape((self.__m,self.__q)))
        freq[0] = np.count_nonzero(rank == self.__m)
        freq[1] = np.count_nonzero(rank == self.__m-1)
        freq[2] = self.__mat_num - (freq[0] + freq[1])
        chi_square = np.sum((freq-self.__mat_num*self.__prob)**2
                            / (self.__mat_num*self.__prob))
        p_value = exp(-chi_square/2)
        return p_value, chi_square, freq

    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

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
        msg += "\n,,Criteria probability,{}\n".format(
            np.array2string(self.__prob, separator=','))
        msg += "SequenceID,p-value,chi_square,Histogram of Rank\n"
        msg += ",,,{},{},other\n".format(self.__m, self.__m-1)
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1],
                np.array2string(j[2], separator=','))
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg.replace('[','').replace(']','')
    
    def __matrix_rank(self, mat):
        """Calculate Rank by elementary row operations."""
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


class DiscreteFourierTransformTest(STS):
    """Discrete Fourier Transform (Spectral) Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.DFT
    NAME = "Discrete Fourier Transform (Spectral) Test"

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
        self.__threshold = sqrt(log(1/0.05)*seq_len)
        self.__ref = 0.95 * seq_len / 2
    
    def func(self, bits) -> tuple:
        """Evaluate the peak heights in DFT of the sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        percentile : `float`
            Percentage of DFT peaks below threshold.
        count : `int`
            Number of DFT peaks below threshold.
        
        """
        bits = 2*bits - 1
        magnitude = np.abs(fft(bits)[:bits.size // 2])
        count = np.count_nonzero(magnitude < self.__threshold)
        percentile = 100*count / (bits.size/2)
        p_value = erfc(
            abs((count-self.__ref)/sqrt(bits.size*0.95*0.05/4)) / sqrt(2))
        return p_value, percentile, count
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = DiscreteFourierTransformTest.NAME + "\n"
        msg += "\nPeak threshold,{}\n".format(self.__threshold)
        msg += "Reference number of peak,{}\n".format(self.__ref)
        msg += "\nSequenceID,p-value,percentage,peaks below threshold\n"
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1], j[2])
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg


class NonOverlappingTemplateMatchingTest(STS):
    """Non-overlapping Template Matching Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.NONOVERLAPPING
    NAME = "Non-overlapping Template Matching Test"

    def __init__(self, seq_len: int, seq_num: int, tpl_len: int =9,
            proc_num: int =1, ig_err: bool =False, init: bool =True) -> None:
        """Set the test parameters.

        Parameters
        ----------
        seq_len : `int`
            Bit length of each split sequence.
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        tpl_len : `int`, optional
            Bit length of each template. Can be set from 2 to 16.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        tpl_path = "spy800_22/_templates/tpl{}.npy".format(tpl_len)
        if tpl_len < 2 or tpl_len > 16:
            msg = "Template length must be between 2 and 16."
            raise InvalidSettingError(msg)
        elif not os.path.isfile(tpl_path):
            msg = "Template file {} is not found.".format(tpl_len)
            raise InvalidSettingError(msg)
        self.__blk_num = 8
        self.__blk_len = seq_len // self.__blk_num
        if self.__blk_len <= tpl_len + 1:
            msg = ("Sequence length must be at least {} bits."
                .format((tpl_len+2)*self.__blk_num))
            raise InvalidSettingError(msg)
        self.__mean = (self.__blk_len - tpl_len + 1) / 2**tpl_len
        self.__var = self.__blk_len*(1/2**tpl_len - (2*tpl_len-1)/2**(2*tpl_len))
        self.__tpl = np.load(tpl_path)
    
    def func(self, bits) -> tuple:
        """Evaluates the number of occurrences of templates 
        (unique m-bit patterns) for each M-bit subsequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `1d-ndarray float`
            Test results for each template.
        chi_square : `1d-ndarray float`
            Computed statistics for each template.
        match : `2d-ndarray int`
            Number of matches in each block for each template.
        
        """
        bits = np.resize(
            bits, (self.__blk_num, self.__blk_len)).astype('uint8')
        match = np.zeros((self.__tpl.shape[0], self.__blk_num), dtype='uint8')
        for i in range(self.__tpl.shape[0]):
            res = cv2.matchTemplate(
                bits, self.__tpl[i].reshape((1,-1)), cv2.TM_SQDIFF)
            match[i] = np.count_nonzero(res <= 0.5, axis=1)
        chi_square = np.sum(((match - self.__mean)/self.__var**0.5)**2, axis=1)  # why?
        p_value = gammaincc(self.__blk_num/2 , chi_square/2)
        return p_value, chi_square, match
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = NonOverlappingTemplateMatchingTest.NAME + "\n"
        msg += "\nTemplate length,{}\n".format(self.__tpl.shape[1])
        msg += "Block length,{}\n".format(self.__blk_len)
        msg += "Number of blocks,{}\n".format(self.__blk_num)
        msg += "Lambda (theoretical mean of matches),{}\n".format(self.__mean)

        for i in range(self.__tpl.shape[0]):
            msg += "Template {},=\"{}\",,,,,,,,,,,".format(
                i, np.array2string(self.__tpl[i], separator=''))
        msg += "\n\n"
        for i in range(self.__tpl.shape[0]):
            msg += "SequenceID,p-value,chi_square,Number of matches,,,,,,,,,"
        msg += "\n"
        for i in range(self.__tpl.shape[0]):
            msg += ",,,B0,B1,B2,B3,B4,B5,B6,B7,,"
        msg += "\n"
        for i, j in enumerate(results):
            for k in range(self.__tpl.shape[0]):
                msg += "{},{},{},{},".format(i, j[0][k], j[1][k],
                    np.array2string(j[2][k], separator=','))
                if len(j) > 3:
                    msg += "{},".format(j[3][k])
                else:
                    msg += ","
            msg += "\n"
        return msg.replace('[','').replace(']','')


class OverlappingTemplateMatchingTest(STS):
    """Ooverlapping Template Matching Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.OVERLAPPING
    NAME = "Overlapping Template Matching Test"

    def __init__(self, seq_len: int, seq_num: int, tpl_len: int =9,
            proc_num: int =1, ig_err: bool =False, init: bool =True) -> None:
        """Set the test parameters.

        Parameters
        ----------
        seq_len : `int`
            Bit length of each split sequence.
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        tpl_len : `int`, optional
            Bit length of each template.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        if tpl_len < 2:
            msg = "Template length must be at least 2 bits."
            raise InvalidSettingError(msg)
        self.__k = 5
        self.__blk_len = 1032
        self.__blk_num = seq_len // self.__blk_len
        if self.__blk_num < 1:
            msg = ("Sequence length must be at least {} bits."
                .format(self.__blk_len))
            raise InvalidSettingError(msg)
        self.__eta = ((self.__blk_len - tpl_len + 1) / 2**tpl_len) / 2
        self.__pi = np.zeros(self.__k+1)
        self.__pi[0] = exp(-self.__eta)
        for i in range(1, self.__k):
            self.__pi[i] = 0.
            for j in range(1, i+1):
                self.__pi[i] += exp(-self.__eta - i*log(2) + j*log(self.__eta)
                    -loggamma(j+1)+loggamma(i)-loggamma(j)-loggamma(i-j+1))
        self.__pi[self.__k] = 1 - np.sum(self.__pi)
        self.__tpl = np.ones((1,tpl_len), dtype='uint8')
    
    def func(self, bits) -> tuple:
        """Evaluates the number of occurrences of a template 
        (duplicate m-bit pattern) for each M-bit subsequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        chi_square : `float`
            Computed statistic.
        hist : `1d-ndarray`
            Histogram of the number of matches in each block.
        
        """
        bits = np.resize(bits, (self.__blk_num,self.__blk_len)).astype('uint8')
        hist = np.zeros(self.__k+1, dtype='uint8')
        res = cv2.matchTemplate(bits, self.__tpl, cv2.TM_SQDIFF)
        match = np.count_nonzero(res <= 0.5, axis=1)
        for i in range(self.__k):
            hist[i] = np.count_nonzero(np.logical_and(match > i-1, match <= i))
        hist[self.__k] = np.count_nonzero(match > i)
        chi_square = np.sum(
            (hist - self.__blk_num*self.__pi)**2 / (self.__blk_num*self.__pi))
        p_value = gammaincc(self.__k/2, chi_square/2)
        return p_value, chi_square, hist
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = OverlappingTemplateMatchingTest.NAME + "\n"
        msg += "\nTemplate length,{}\n".format(self.__tpl.size)
        msg += "Template,=\"{}\"\n".format(np.array2string(
                self.__tpl, separator=''))
        msg += "Block length,{}\n".format(self.__blk_len)
        msg += "Number of blocks,{}\n".format(self.__blk_num)
        msg += (",,Pi (theoretical probabilities of matches),{}\n"
            .format(np.array2string(self.__pi, separator=',')))
        msg += "\nSequenceID,p-value,chi_square,Histogram of matches\n"
        msg += ",,,{}<=\n".format(
            np.array2string(np.arange(self.__k+1), separator=','))
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1],
                np.array2string(j[2], separator=','))
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg.replace('[','').replace(']','')


class MaurersUniversalStatisticalTest(STS):
    """Maurer's "Universal Statistical" Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.UNIVERSAL
    NAME = "Maurer's \"Universal Statistical\" Test"

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
        self.__blk_len = 5
        th = [387840, 904960, 2068480, 4654080, 10342400, 22753280,
              49643520, 107560960, 231669760, 496435200, 1059061760]
        for i in th:
            if seq_len >= i:
                self.__blk_len += 1
        if self.__blk_len < 6:
            msg = "Sequence length must be at least {} bits.".format(th[0])
            raise InvalidSettingError(msg)
        var = [0, 0, 0, 0, 0, 0, 2.954, 3.125, 3.238, 3.311,
               3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421]
        exp_val = [0, 0, 0, 0, 0, 0, 5.2177052, 6.1962507, 7.1836656,
                   8.1764248, 9.1723243, 10.170032, 11.168765,
                   12.168070, 13.167693, 14.167488, 15.167379]
        self.__var = var[self.__blk_len]
        self.__exp_val = exp_val[self.__blk_len]
        self.__q = 10 * 2**self.__blk_len
        self.__k = seq_len // self.__blk_len - self.__q
        self.__sigma = (
            (0.7 - 0.8/self.__blk_len + (4+32/self.__blk_len)/15
            * self.__k**(-3/self.__blk_len)) * sqrt(self.__var/self.__k))
    
    def func(self, bits) -> tuple:
        """Evaluate the distance between L-bit patterns repeatedly observed.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        phi : `float`
            Computed statistic.
        
        """
        t = np.zeros(2**self.__blk_len)
        bits = self.__packbits(
            np.resize(bits, (self.__k+self.__q, self.__blk_len)))
        uniq, idx = np.unique(bits[:self.__q][::-1], return_index=True)
        t[uniq] = idx
        s = 0
        for i in range(t.size):
            s += np.sum(np.log2(np.diff(
                np.append(-t[i], np.where(bits[self.__q:]==i)))))
        phi = s / self.__k
        p_value = erfc(abs(phi-self.__exp_val) / (sqrt(2)*self.__sigma))
        return p_value, phi
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = MaurersUniversalStatisticalTest.NAME + "\n"
        msg += "\nBlock length,{}\n".format(self.__blk_len)
        msg += "Theoretical variance,{}\n".format(self.__var)
        msg += "Theoretical expected value,{}\n".format(self.__exp_val)
        msg += "Theoretical standard deviation,{}\n".format(self.__sigma)
        msg += "\nSequenceID,p-value,pi\n"
        for i, j in enumerate(results):
            msg += "{},{},{}".format(i, j[0], j[1])
            if len(j) > 2:
                msg += ",{}".format(j[2])
            msg += "\n"
        return msg
    
    def __packbits(self, x, reverse=True):
        """Converts a binary matrix to a decimal value."""
        p = np.power(2, np.arange(x.shape[-1]))
        if reverse:
            p = p[::-1]
        return np.dot(x, p)


class LinearComplexityTest(STS):
    """Linear complexity Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.COMPLEXITY
    NAME = "Linear complexity Test"

    def __init__(self, seq_len: int, seq_num: int, blk_len: int =500,
            proc_num: int =1, ig_err: bool =False, init: bool =True) -> None:
        """Set the test parameters.

        Parameters
        ----------
        seq_len : `int`
            Bit length of each split sequence.
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        blk_len : `int`, optional
            Bit length of each block.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        if seq_len < blk_len:
            msg = ("Sequence length must be at least {} bits."
                .format(blk_len))
            raise InvalidSettingError(msg)
        self.__blk_len = blk_len
        self.__blk_num = seq_len // blk_len
        self.__mu = (
            blk_len/2 + ((-1)**(blk_len+1)+9)/36 - (blk_len/3+2/9)/2**blk_len)
        self.__pi = np.array(
            [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833])
    
    def func(self, bits) -> tuple:
        """Evaluate the length of the linear feedback shift register (LFSR).

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        chi_square : `float`
            Computed statistic.
        hist : `1d-ndarray int`
            Histogram of T.
        
        """
        bits = np.resize(bits, (self.__blk_num, self.__blk_len))
        l = np.empty(self.__blk_num)
        for i in range(self.__blk_num):
            l[i] = self.__bma(bits[i])
        t = (-1)**self.__blk_len * (l-self.__mu) + 2/9
        hist = np.histogram(t,
            bins=[-bits.size, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, bits.size])[0]
        chi_square = np.sum(
            (hist-self.__blk_num*self.__pi)**2 / (self.__blk_num*self.__pi))
        p_value = gammaincc(6/2.0 , chi_square/2.0)
        return p_value, chi_square, hist
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = LinearComplexityTest.NAME + "\n"
        msg += "\nBlock length,{}\n".format(self.__blk_len)
        msg += "Number of blocks,{}\n".format(self.__blk_num)
        msg += "Theoretical mean of linear complexity,{}\n".format(self.__mu)
        msg += "\nSequenceID,p-value,chi_square,Histogram of T\n"
        msg += ",,,C0,C1,C2,C3,C4,C5,C6\n"
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1],
                np.array2string(j[2], separator=','))
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg.replace('[','').replace(']','')
    
    def __bma(self, bits):
        """Berlekamp Massey Algorithm."""
        c, b = np.zeros(bits.size, dtype=int), np.zeros(bits.size, dtype=int)
        c[0], b[0] = 1, 1
        l, m, i = 0, -1, 0
        for i in range(bits.size):
            if (bits[i] + np.dot(bits[i-l:i][::-1], c[1:1+l])) % 2 == 1:
                t = c.copy()
                c[np.where(b[:l]==1)[0] + i-m] += 1
                c = c % 2
                if l <= i>>1:
                    l = i + 1 - l
                    m = i
                    b = t
        return l


class SerialTest(STS):
    """Serial Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.SERIAL
    NAME = "Serial Test"

    def __init__(self, seq_len: int, seq_num: int, blk_len: int =16,
            proc_num: int =1, ig_err: bool =False, init: bool =True) -> None:
        """Set the test parameters.

        Parameters
        ----------
        seq_len : `int`
            Bit length of each split sequence.
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        blk_len : `int`, optional
            Bit length of each block.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        if seq_len < blk_len:
            msg = ("Sequence length must be at least {} bits."
                .format(blk_len))
            raise InvalidSettingError(msg)
        self.__blk_len = blk_len
    
    def func(self, bits) -> tuple:
        """Evaluate the frequency of all possible overlapping m-bit patterns.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `1d-ndarray float`
            Test results.
        psi : `1d-ndarray float`
            Computed statistics.
        
        """
        psi = np.zeros(3)
        p_value = np.zeros(2)
        for i in range(len(psi)):
            psi[i] = self.__psi_square(bits, self.__blk_len-i)
        p_value[0] = gammaincc(2**(self.__blk_len-1)/2.0, (psi[0]-psi[1])/2.0)
        p_value[1] = gammaincc(
            2**(self.__blk_len-2)/2.0, (psi[0]-2*psi[1]+psi[2])/2.0)
        return p_value, psi
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = SerialTest.NAME + "\n"
        msg += "\nBlock length,{}\n".format(self.__blk_len)
        msg += "\nSequenceID,p-value1,p-value2,psi_m,psi_m-1,psi_m-2\n"
        for i, j in enumerate(results):
            msg += "{},{},{}".format(i, np.array2string(j[0], separator=','),
                np.array2string(j[1], separator=','))
            if len(j) > 2:
                msg += ",{}".format(j[2])
            msg += "\n"
        return msg.replace('[','').replace(']','')
    
    def __psi_square(self, x, m):
        """Compute statistics."""
        p, k = np.zeros(2**(m+1)-1), np.ones(x.size, dtype=int)
        j = np.arange(x.size)
        for i in range(m):
            ref = x[(i+j) % x.size]
            k[ref==0] *= 2
            k[ref==1] = 2*k[ref==1] + 1
        uniq, counts = np.unique(k, return_counts=True)
        p[uniq-1] = 1*counts
        s = np.sum(p[2**m-1 : 2**(m+1)-1]**2) * 2**m/x.size - x.size
        return s


class ApproximateEntropyTest(STS):
    """Approximate entropy Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    NAME : `str`
        A unique test name for the class.
    
    """

    ID = STS.TestID.ENTROPY
    NAME = "Approximate entropy Test"

    def __init__(self, seq_len: int, seq_num: int, blk_len: int =10,
            proc_num: int =1, ig_err: bool =False, init: bool =True) -> None:
        """Set the test parameters.

        Parameters
        ----------
        seq_len : `int`
            Bit length of each split sequence.
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        blk_len : `int`, optional
            Bit length of each block.
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        init : `bool`, optional
            If `True`, initialize the super class.

        """
        if init:
            super().__init__(seq_len, seq_num, proc_num, ig_err)
        ref = 2**(blk_len+5)
        if seq_len < ref:
            msg = "Sequence length must be at least {} bits.".format(ref)
            raise InvalidSettingError(msg)
        self.__blk_len = blk_len
    
    def func(self, bits) -> tuple:
        """Evaluate the frequency of all possible overlapping m-bit patterns.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : `float`
            Test result.
        chi_square : `float`
            Computed statistic.
        apen : `float`
            Approximate entropy.
        
        """
        apen = (self.__phi_m(bits, self.__blk_len)
            - self.__phi_m(bits, self.__blk_len+1))
        chi_square = 2*bits.size*(log(2) - apen)
        p_value = gammaincc(2**(self.__blk_len-1), chi_square/2.0)
        return p_value, chi_square, apen
    
    def report(self, results: list) -> str:
        """Generate a CSV string from the partial test results.

        Parameters
        ----------
        results : `list`
            List of test results (List of returns of `func` method).
        
        Returns
        -------
        msg : `str`
            Generated report.
        
        """
        msg = ApproximateEntropyTest.NAME + "\n"
        msg += "\nBlock length,{}\n".format(self.__blk_len)
        msg += "\nSequenceID,p-value,chi_square,Approximate entropy\n"
        for i, j in enumerate(results):
            msg += "{},{},{},{}".format(i, j[0], j[1], j[2])
            if len(j) > 3:
                msg += ",{}".format(j[3])
            msg += "\n"
        return msg
    
    def __phi_m(self, x, m):
        """Compute statistics."""
        p, k = np.zeros(2**(m+1)-1), np.ones(x.size, dtype=int)
        j = np.arange(x.size)
        for i in range(m):
            k *= 2
            k[x[(i+j) % x.size] == 1] += 1
        uniq, counts = np.unique(k, return_counts=True)
        p[uniq-1] = 1*counts
        ref = p[2**m-1 : 2**(m+1)-1]
        ref = ref[np.nonzero(ref)[0]]
        s = np.sum(ref*np.log(ref/x.size)) / x.size
        return s
