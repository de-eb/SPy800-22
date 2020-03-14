#!/usr/bin/env python3
# -*- Coding: utf-8 -*-


# tests.py
#
# Copyright (c) 2020 Takuya Kawashima
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php


"""Implementation of SP800-22 test algorithms by Python.

This module is part of the spy800_22 package and consists of 15 + 1 classes.
Each of the 15 classes corresponds to 15 tests of NIST SP800-22.
The last one is a wrapper class to execute them continuously.
These classes provide various functions (data I/O, parallel testing, etc.)
to execute each test.

Notes
-----
Python 3.7 or higher required.
Depends on NumPy, SciPy and OpenCV libraries.

More info.
----------
Details of NIST SP800-22:
https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

NIST's official implementation:
https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software

"""

import os
from math import exp, log, log2, sqrt

import cv2
import numpy as np
from scipy.fftpack import fft
from scipy.special import erfc, gammaincc, loggamma
from scipy.stats import norm

from spy800_22.sts import STS, InvalidSettingError, StatisticalError


class FrequencyTest(STS):
    """Frequency (Monobit) Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = FrequencyTest(
            file="file.txt",
            fmt=FrequencyTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.FREQUENCY

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            FrequencyTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.
        
        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)

    def func(self, bits) -> dict:
        """Evaluate the uniformity of 0s and 1s for the entire sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'zeros'  : int Number of occurrences of 0.
                'ones'   : int Number of occurrences of 1.
            }
        
        """
        ones = np.count_nonzero(bits)
        zeros = bits.size - ones
        p_value = erfc((abs(ones-zeros)/sqrt(bits.size))/sqrt(2))
        return {'p-value': p_value, 'zeros': zeros, 'ones': ones}

    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,zeros count,ones count\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},{}\n".format(i, res['p-value'][i],
                    res['zeros'][i], res['ones'][i])
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv


class BlockFrequencyTest(STS):
    """Frequency Test within a Block

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = BlockFrequencyTest(
            file="file.txt",
            fmt=BlockFrequencyTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.BLOCKFREQUENCY

    def __init__(self, file: str, fmt: STS.ReadAs, seq_len: int,
            seq_num: int, blk_len: int =128, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            BlockFrequencyTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        blk_len : `int`, optional
            Bit length of each block.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        if seq_len < blk_len:
            msg = "'blk_len' must be smaller than 'seq_len'."
            raise InvalidSettingError(msg, self.ID)
        self.__blk_len = blk_len
        self.__blk_num = seq_len // blk_len

    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Block length': self.__blk_len,
                'Number of blocks': self.__blk_num}
    
    def func(self, bits) -> dict:
        """Evaluate the proportion of 1s for each M-bit subsequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'chi^2'  : float Computed statistic.
            }
        
        """
        bits = np.resize(bits, (self.__blk_num, self.__blk_len))
        sigma = np.sum((np.sum(bits, axis=1)/self.__blk_len - 0.5)**2)
        chi_square = 4 * self.__blk_len * sigma
        p_value = gammaincc(self.__blk_num/2 , chi_square/2)
        return {'p-value': p_value, 'chi^2': chi_square}
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,chi^2\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{}\n".format(i,
                    res['p-value'][i], res['chi^2'][i])
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv


class RunsTest(STS):
    """Runs Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = RunsTest(
            file="file.txt",
            fmt=RunsTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.RUNS

    def __init__(self, file: str, fmt: STS.ReadAs, 
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            RunsTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        
    def func(self, bits) -> dict:
        """Evaluate the total number of "Run"s for the entire sequence.
        
        "Run" is a continuation of the same bit.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'pi'     : float Estimator criteria.
                'run'    : int Total of Runs.
            }
        
        Raise
        -----
        StatisticalError :
            When significantly biased statistics are calculated.
        
        """
        pi = np.count_nonzero(bits) / bits.size
        if abs(pi-0.5) > 2/sqrt(bits.size):
            msg = "Estimator criteria not met. (Pi = {})".format(pi)
            raise StatisticalError(msg, self.ID)
        run = np.count_nonzero(np.diff(bits)) + 1
        p_value = erfc(
            abs(run-2*bits.size*pi*(1-pi)) / (2*pi*(1-pi)*sqrt(2*bits.size)))
        return {'p-value': p_value, 'pi': pi, 'run': run}
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,pi,Total Run\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},{}\n".format(i, res['p-value'][i],
                    res['pi'][i], res['run'][i])
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv


class LongestRunOfOnesTest(STS):
    """Test for the Longest Run of Ones in a Block

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = LongestRunOfOnesTest(
            file="file.txt",
            fmt=LongestRunOfOnesTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.LONGESTRUN

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            LongestRunOfOnesTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        if seq_len < 128:
            msg = "'seq_len' must be at least 128 bits."
            raise InvalidSettingError(msg, self.ID)
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
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Block length': self.__blk_len,
                'Number of Blocks': self.__blk_num}
        
    def func(self, bits) -> dict:
        """Evaluate the longest "Run" of 1s for each M-bit subsequence.
        
        "Run" is a continuation of the same bit.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'chi^2'  : float Computed statistic.
                'hist'   : ndarray Histogram of the longest Run in each block.
            }
        
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
        return {'p-value': p_value, 'chi^2': chi_square, 'hist': hist}

    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,chi^2,Histogram of longest Run"
        csv += (",<={}<=\n".format(np.array2string(self.__v, separator=',')))
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},,{}\n".format(i,
                    res['p-value'][i], res['chi^2'][i],
                    np.array2string(res['hist'][i], separator=','))
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv.replace('[','').replace(']','')


class BinaryMatrixRankTest(STS):
    """Binary Matrix Rank Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = BinaryMatrixRankTest(
            file="file.txt",
            fmt=BinaryMatrixRankTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.RANK

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            BinaryMatrixRankTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        self.__m, self.__q = 32, 32
        self.__mat_num = seq_len // (self.__m * self.__q)
        if self.__mat_num == 0:
            msg = "'seq_len' must be at least {} bits.".format(
                self.__m * self.__q)
            raise InvalidSettingError(msg, self.ID)
        self.__prob = np.zeros(3)
        for i, r in enumerate([self.__m, self.__m-1]):
            j = np.arange(r, dtype=float)
            prod = np.prod(
                (1-2**(j-self.__m)) * (1-2**(j-self.__q)) / (1-2**(j-r)))
            self.__prob[i] = (
                2**(r*(self.__m + self.__q - r) - self.__m*self.__q) * prod)
        self.__prob[2] = 1 - (self.__prob[0] + self.__prob[1])
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Matrix shape': np.array([self.__m, self.__q]),
                'Number of matrices': self.__mat_num,
                'Criteria probability': self.__prob}
        
    def func(self, bits) -> dict:
        """Evaluate the Rank of disjoint sub-matrices of the entire sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'chi^2'  : float Computed statistic.
                'freq'   : ndarray Number of occurrences of Ranks.
            }
        
        """
        hist = np.zeros(3, dtype=int)
        rank = np.zeros(self.__mat_num, dtype=int)
        bits = np.resize(bits, (self.__mat_num, self.__m*self.__q))
        for i in range(self.__mat_num):
            rank[i] = self.__matrix_rank(bits[i].reshape((self.__m,self.__q)))
        hist[0] = np.count_nonzero(rank == self.__m)
        hist[1] = np.count_nonzero(rank == self.__m-1)
        hist[2] = self.__mat_num - (hist[0] + hist[1])
        chi_square = np.sum((hist-self.__mat_num*self.__prob)**2
                            / (self.__mat_num*self.__prob))
        p_value = exp(-chi_square/2)
        return {'p-value': p_value, 'chi^2': chi_square, 'hist': hist}
    
    def __matrix_rank(self, mat) -> int:
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
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,chi^2,Histogram of Rank"
        csv += ",{},{},other\n".format(self.__m, self.__m-1)
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},,{}\n".format(i,
                    res['p-value'][i], res['chi^2'][i],
                    np.array2string(res['hist'][i], separator=','))
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv.replace('[','').replace(']','')


class DiscreteFourierTransformTest(STS):
    """Discrete Fourier Transform (Spectral) Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = DiscreteFourierTransformTest(
            file="file.txt",
            fmt=DiscreteFourierTransformTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.DFT

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            DiscreteFourierTransformTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        self.__threshold = sqrt(log(1/0.05)*seq_len)
        self.__ref = 0.95 * seq_len / 2

    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Peak threshold': self.__threshold,
                'Reference peaks': self.__ref}
    
    def func(self, bits) -> dict:
        """Evaluate the peak heights in DFT of the sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value'   : float Test result.
                'percentile': float Percentage of DFT peaks below threshold.
                'count'     : int Number of DFT peaks below threshold.
            }
        
        """
        bits = 2*bits - 1
        magnitude = np.abs(fft(bits)[:bits.size // 2])
        count = np.count_nonzero(magnitude < self.__threshold)
        pct = 100*count / (bits.size/2)
        p_value = erfc(
            abs((count-self.__ref)/sqrt(bits.size*0.95*0.05/4)) / sqrt(2))
        return {'p-value': p_value, 'percentile': pct, 'peaks': count}
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,percentile,peaks below threshold\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},{}\n".format(i, res['p-value'][i],
                    res['percentile'][i], res['peaks'][i])
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv


class NonOverlappingTemplateMatchingTest(STS):
    """Non-overlapping Template Matching Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = NonOverlappingTemplateMatchingTest(
            file="file.txt",
            fmt=NonOverlappingTemplateMatchingTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.NONOVERLAPPING

    def __init__(self, file: str, fmt: STS.ReadAs, seq_len: int,
            seq_num: int, tpl_len: int =9, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            NonOverlappingTemplateMatchingTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        tpl_len : `int`, optional
            Bit length of each template. Can be set from 2 to 16.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        tpl_path = "spy800_22/_templates/tpl{}.npy".format(tpl_len)
        if tpl_len < 2 or tpl_len > 16:
            msg = "'tpl_len' must be between 2 and 16."
            raise InvalidSettingError(msg, self.ID)
        elif not os.path.isfile(tpl_path):
            msg = "Template file {} is not found.".format(tpl_len)
            raise InvalidSettingError(msg, self.ID)
        self.__blk_num = 8
        self.__blk_len = seq_len // self.__blk_num
        if self.__blk_len <= tpl_len + 1:
            msg = "'seq_len' must be at least "\
                "('tpl_len' + 2) x 8 bits."
            raise InvalidSettingError(msg, self.ID)
        self.__mean = (self.__blk_len - tpl_len + 1) / 2**tpl_len
        self.__var = self.__blk_len*(1/2**tpl_len-(2*tpl_len-1)/2**(2*tpl_len))
        self.__tpl = np.load(tpl_path)
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Template length': self.__tpl.shape[1],
                'Block length': self.__blk_len,
                'Number of blocks': self.__blk_num,
                'Lambda (theoretical mean of matches)': self.__mean}
    
    def func(self, bits) -> dict:
        """Evaluates the number of occurrences of templates 
        (unique m-bit patterns) for each M-bit subsequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': ndarray Test results of each template.
                'chi^2'  : ndarray Computed statistics of each template.
                'match'  : ndarray Number of template matches in each block.
            }
        
        """
        bits = np.resize(
            bits, (self.__blk_num, self.__blk_len)).astype('uint8')
        match = np.zeros((self.__tpl.shape[0], self.__blk_num), dtype=int)
        for i in range(self.__tpl.shape[0]):
            res = cv2.matchTemplate(
                bits, self.__tpl[i].reshape((1,-1)), cv2.TM_SQDIFF)
            match[i] = np.count_nonzero(res <= 0.5, axis=1)
        chi_square = np.sum(((match - self.__mean)/self.__var**0.5)**2, axis=1)
        p_value = gammaincc(self.__blk_num/2 , chi_square/2)
        return {'p-value': p_value, 'chi^2': chi_square, 'matches': match}
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Template,"
        for i in range(self.__tpl.shape[0]):
            csv += "=\"{1}\"{0}".format(","*10,
                np.array2string(self.__tpl[i], separator=''))
        csv += "\n,,Proportion,{}\n".format(
            np.array2string(res['Proportion'], separator=','*10))
        csv += ",,Uniformity,{}\n".format(
            np.array2string(res['Uniformity'], separator=','*10))
        csv += ",,SequenceID{}\n".format(
            ",p-value,chi^2,B0,B1,B2,B3,B4,B5,B6,B7"*self.__tpl.shape[0])
        for i in range(len(res['p-value'])):
            csv += "\n,,{}".format(i)
            if res['err'][i] is None:
                for j in range(self.__tpl.shape[0]):
                    csv += ",{},{},{}".format(
                        res['p-value'][i][j], res['chi^2'][i][j],
                        np.array2string(res['matches'][i][j], separator=','))
            else:
                csv += ",{}".format(res['err'][i])
        csv += "\n"
        return csv.replace('[','').replace(']','')


class OverlappingTemplateMatchingTest(STS):
    """Ooverlapping Template Matching Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = OverlappingTemplateMatchingTest(
            file="file.txt",
            fmt=OverlappingTemplateMatchingTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.OVERLAPPING

    def __init__(self, file: str, fmt: STS.ReadAs, seq_len: int,
            seq_num: int, tpl_len: int =9, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            OverlappingTemplateMatchingTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        tpl_len : `int`, optional
            Bit length of each template.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        if tpl_len < 2:
            msg = "'tpl_len' must be at least 2 bits."
            raise InvalidSettingError(msg, self.ID)
        self.__k = 5
        self.__blk_len = 1032
        self.__blk_num = seq_len // self.__blk_len
        if self.__blk_num < 1:
            msg = "'seq_len' must be at least {} bits.".format(
                self.__blk_len)
            raise InvalidSettingError(msg, self.ID)
        if tpl_len >= self.__blk_len - 1:
            msg = "'tpl_len' must be {} bits or less.".format(
                self.__blk_len - 1)
            raise InvalidSettingError(msg, self.ID)
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
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Template length': self.__tpl.size,
                'Template': "=\"{}\"".format("1"*self.__tpl.size),
                'Block length': self.__blk_len,
                'Number of blocks': self.__blk_num,
                'Pi (theoretical probabilities of matches)': self.__pi}
    
    def func(self, bits) -> dict:
        """Evaluates the number of occurrences of a template 
        (duplicate m-bit pattern) for each M-bit subsequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'chi^2'  : float Computed statistic.
                'hist'   : ndarray Histogram of the number of template matches.
            }
        
        """
        bits = np.resize(bits, (self.__blk_num,self.__blk_len)).astype('uint8')
        hist = np.zeros(self.__k+1, dtype=int)
        res = cv2.matchTemplate(bits, self.__tpl, cv2.TM_SQDIFF)
        match = np.count_nonzero(res <= 0.5, axis=1)
        for i in range(self.__k):
            hist[i] = np.count_nonzero(np.logical_and(match > i-1, match <= i))
        hist[self.__k] = np.count_nonzero(match > i)
        chi_square = np.sum(
            (hist - self.__blk_num*self.__pi)**2 / (self.__blk_num*self.__pi))
        p_value = gammaincc(self.__k/2, chi_square/2)
        return {'p-value': p_value, 'chi^2': chi_square, 'hist': hist}
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,chi^2,Histogram of matches"
        csv += ",{}<=\n".format(
            np.array2string(np.arange(self.__k+1), separator=','))
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},,{}\n".format(
                    i, res['p-value'][i], res['chi^2'][i],
                    np.array2string(res['hist'][i], separator=','))
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv.replace('[','').replace(']','')


class MaurersUniversalStatisticalTest(STS):
    """Maurer's "Universal Statistical" Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = MaurersUniversalStatisticalTest(
            file="file.txt",
            fmt=MaurersUniversalStatisticalTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.UNIVERSAL

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            MaurersUniversalStatisticalTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        self.__blk_len = 5
        th = [387840, 904960, 2068480, 4654080, 10342400, 22753280,
              49643520, 107560960, 231669760, 496435200, 1059061760]
        for i in th:
            if seq_len >= i:
                self.__blk_len += 1
        if self.__blk_len < 6:
            msg = "'seq_len' must be at least {} bits.".format(th[0])
            raise InvalidSettingError(msg, self.ID)
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
        self.__init_idx = np.arange(self.__q-1, -1, -1)
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Block length': self.__blk_len,
                'Theoretical variance': self.__var,
                'Theoretical expected value': self.__exp_val,
                'Theoretical standard deviation': self.__sigma}
    
    def func(self, bits) -> dict:
        """Evaluate the distance between L-bit patterns repeatedly observed.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'chi^2'  : float Computed statistic.
            }
        
        """
        t = np.zeros(2**self.__blk_len, dtype=int)
        bits = self.__packbits(
            np.resize(bits, (self.__k+self.__q, self.__blk_len)))
        uniq, idx = np.unique(bits[:self.__q][::-1], return_index=True)
        t[uniq] = self.__init_idx[idx]
        s = 0
        for i in range(t.size):
            s += np.sum(np.log2(np.diff(np.append(
                t[i], np.where(bits[self.__q:]==i)[0]+self.__q))))
        phi = s / self.__k
        p_value = erfc(abs(phi-self.__exp_val) / (sqrt(2)*self.__sigma))
        return {'p-value': p_value, 'phi': phi}
    
    def __packbits(self, x, reverse=True):
        """Converts a binary matrix to a decimal value."""
        p = np.power(2, np.arange(x.shape[-1]))
        if reverse:
            p = p[::-1]
        return np.dot(x, p)
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,pi\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{}\n".format(i, res['p-value'][i],
                    res['phi'][i])
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv


class LinearComplexityTest(STS):
    """Linear complexity Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = LinearComplexityTest(
            file="file.txt",
            fmt=LinearComplexityTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.COMPLEXITY

    def __init__(self, file: str, fmt: STS.ReadAs, seq_len: int,
            seq_num: int, blk_len: int =500, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            LinearComplexityTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        blk_len : `int`, optional
            Bit length of each block.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        if seq_len < blk_len:
            msg = "'blk_len' must be smaller than 'seq_len'."
            raise InvalidSettingError(msg, self.ID)
        self.__blk_len = blk_len
        self.__blk_num = seq_len // blk_len
        self.__mu = (
            blk_len/2 + ((-1)**(blk_len+1)+9)/36 - (blk_len/3+2/9)/2**blk_len)
        self.__pi = np.array(
            [0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833])

    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Block length': self.__blk_len,
                'Number of blocks': self.__blk_num,
                'Theoretical mean of linear complexity': self.__mu}
    
    def func(self, bits) -> dict:
        """Evaluate the length of the linear feedback shift register (LFSR).

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'chi^2'  : float Computed statistic.
                'hist'   : ndarray Histogram of T.
            }
        
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
        return {'p-value': p_value, 'chi^2': chi_square, 'hist': hist}
    
    def __bma(self, bits) -> int:
        """Berlekamp Massey Algorithm."""
        c, b = np.zeros(bits.size, dtype=int), np.zeros(bits.size, dtype=int)
        c[0], b[0] = 1, 1
        l, m = 0, -1
        for i in range(bits.size):
            if (bits[i] + np.dot(bits[i-l:i][::-1], c[1:1+l])) % 2 == 1:
                t = c.copy()
                c[np.where(b[:l]==1)[0] + i-m] += 1
                c = c % 2
                if l <= i>>1:
                    l = i + 1 - l
                    m, b = i, t
        return l
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,chi^2,Histogram of T"
        csv += ",C0,C1,C2,C3,C4,C5,C6\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},,{}\n".format(
                    i, res['p-value'][i], res['chi^2'][i],
                    np.array2string(res['hist'][i], separator=','))
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv.replace('[','').replace(']','')


class SerialTest(STS):
    """Serial Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.

    Examples
    --------
    >>> test = SerialTest(
            file="file.txt",
            fmt=SerialTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.SERIAL

    def __init__(self, file: str, fmt: STS.ReadAs, seq_len: int,
            seq_num: int, blk_len: int =16, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            SerialTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        blk_len : `int`, optional
            Bit length of each block.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        if seq_len < blk_len:
            msg = "'blk_len' must be smaller than 'seq_len'."
            raise InvalidSettingError(msg, self.ID)
        self.__blk_len = blk_len
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Block length': self.__blk_len}
    
    def func(self, bits) -> dict:
        """Evaluate the frequency of all possible overlapping m-bit patterns.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': ndarray Test results.
                'psi'  : ndarray Computed statistics.
            }
        
        """
        psi = np.zeros(3)
        p_value = np.zeros(2)
        for i in range(len(psi)):
            psi[i] = self.__psi_square(bits, self.__blk_len-i)
        p_value[0] = gammaincc(2**(self.__blk_len-1)/2.0, (psi[0]-psi[1])/2.0)
        p_value[1] = gammaincc(
            2**(self.__blk_len-2)/2.0, (psi[0]-2*psi[1]+psi[2])/2.0)
        return {'p-value': p_value, 'psi': psi}
    
    def __psi_square(self, x, m) -> float:
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
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{0[0]},{0[1]}\n".format(res['Proportion'])
        csv += ",,Uniformity,{0[0]},{0[1]}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value1,p-value2,psi_m,psi_m-1,psi_m-2\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{}\n".format(
                    i, np.array2string(res['p-value'][i], separator=','),
                    np.array2string(res['psi'][i], separator=','))
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv.replace('[','').replace(']','')


class ApproximateEntropyTest(STS):
    """Approximate entropy Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = ApproximateEntropyTest(
            file="file.txt",
            fmt=ApproximateEntropyTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.ENTROPY

    def __init__(self, file: str, fmt: STS.ReadAs, seq_len: int,
            seq_num: int, blk_len: int =10, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            ApproximateEntropyTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        blk_len : `int`, optional
            Bit length of each block.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.

        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        ref = 2**(blk_len+5)
        if seq_len < ref:
            msg = "'seq_len' must be at least 2^('blk_len' + 5) bits."
            raise InvalidSettingError(msg, self.ID)
        self.__blk_len = blk_len
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Block length': self.__blk_len}
    
    def func(self, bits) -> dict:
        """Evaluate the frequency of all possible overlapping m-bit patterns.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': float Test result.
                'chi^2'  : float Computed statistic.
                'apen'   : float Approximate entropy.
            }
        
        """
        apen = (self.__phi_m(bits, self.__blk_len)
            - self.__phi_m(bits, self.__blk_len+1))
        chi_square = 2*bits.size*(log(2) - apen)
        p_value = gammaincc(2**(self.__blk_len-1), chi_square/2.0)
        return {'p-value': p_value, 'chi^2': chi_square, 'entropy': apen}
    
    def __phi_m(self, x, m) -> float:
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
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{}\n".format(res['Proportion'])
        csv += ",,Uniformity,{}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value,chi^2,Approximate entropy\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},{}\n".format(i, res['p-value'][i],
                    res['chi^2'][i], res['entropy'][i])
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv


class CumulativeSumsTest(STS):
    """Cumulative Sums (Cusum) Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = CumulativeSumsTest(
            file="file.txt",
            fmt=CumulativeSumsTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.CUSUM

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            CumulativeSumsTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.
        
        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)

    def func(self, bits) -> dict:
        """Evaluate the maximal excursion (from `0`) of the random walk.
        
        Random walk is defined by the cumulative sum
        of adjusted (`-1`, `+1`) digits in the sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': ndarray Test results for forward and backward.
                'sums'   : ndarray The largest absolute values of partial sums.
            }
        
        """
        p_value = np.zeros(2)
        sums = np.zeros(2, dtype=int)
        bits = 2*bits - 1
        sums[0] = np.max(np.abs(np.cumsum(bits)))
        p_value[0] = (1 - self.__sigma(bits.size, sums[0])
                        + self.__sigma(bits.size, sums[0], term=True))
        sums[1] = np.max(np.abs(np.cumsum(bits[::-1])))
        p_value[1] = (1 - self.__sigma(bits.size, sums[1])
                        + self.__sigma(bits.size, sums[1], term=True))
        return {'p-value': p_value, 'cusums': sums}
    
    def __sigma(self, n, z, term=False) -> float:
        a, b, c = 1, 1, -1
        if term:
            a, b, c = -3, 3, 1
        k = np.arange(int((-n/z+a)//4), int((n/z-1)//4)+1)
        s = np.sum(norm.cdf((4*k+b)*z/sqrt(n)) - norm.cdf((4*k+c)*z/sqrt(n)))
        return s
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,Proportion,{0[0]},,{0[1]}\n".format(res['Proportion'])
        csv += ",,Uniformity,{0[0]},,{0[1]}\n".format(res['Uniformity'])
        csv += ",,SequenceID,p-value(Forward),maximum partial sum(Forward)"\
            ",p-value(Reverse),maximum partial sum(Reverse)\n"
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{},{},{},{}\n".format(
                    i, res['p-value'][i][0], res['cusums'][i][0],
                    res['p-value'][i][1], res['cusums'][i][1])
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv


class RandomExcursionsTest(STS):
    """Random Excursions Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = RandomExcursionsTest(
            file="file.txt",
            fmt=RandomExcursionsTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.EXCURSIONS

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            RandomExcursionsTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.
        
        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        self.__pi = np.array(
            [[0.8750000000,  0.8333333333,  0.7500000000,  0.5000000000],
            [0.01562500000, 0.02777777778, 0.06250000000, 0.25000000000],
            [0.01367187500, 0.02314814815, 0.04687500000, 0.12500000000],
            [0.01196289063, 0.01929012346, 0.03515625000, 0.06250000000],
            [0.01046752930, 0.01607510288, 0.02636718750, 0.03125000000],
            [0.0732727051,  0.0803755143,  0.0791015625,  0.0312500000]])
        self.__pi = np.hstack((self.__pi, np.fliplr(self.__pi)))
        self.__stat = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        self.__up_lim = max(1000, seq_len/100)
        self.__low_lim = max(500, 0.005*sqrt(seq_len))
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Upper limit of cycles': self.__up_lim,
                'Lower limit of cycles': self.__low_lim}

    def func(self, bits) -> dict:
        """Evaluate the number of cycles having K visits in a random walk.
        
        Random walk is defined by the cumulative sum
        of adjusted (`-1`, `+1`) digits in the sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': ndarray Test results for each state.
                'chi^2'  : ndarray Computed statistic for each state.
                'cycle'  : Number of cycles.
            }
        
        Raise
        -----
        StatisticalError :
            When significantly biased statistics are calculated.
        
        """
        bits = 2*bits - 1
        s = np.pad(np.cumsum(bits), (1,1))
        idx = np.where(s==0)[0]
        cycle = idx.size - 1
        if cycle > self.__up_lim or cycle < self.__low_lim:
            msg = "Number of cycles is out of expected range."
            raise StatisticalError(msg, self.ID)
        hist = np.zeros((cycle, self.__stat.size), dtype=int)
        freq = np.zeros((6, self.__stat.size), dtype=int)
        for i in range(cycle):
            hist[i] = np.histogram(
                s[idx[i]+1:idx[i+1]], bins=[-4, -3, -2, -1, 1, 2, 3, 4, 4.1])[0]
        for i in range(6):
            freq[i] = np.count_nonzero(hist==i, axis=0)
        freq[i] += np.count_nonzero(hist>i, axis=0)
        chi_square = np.sum((freq-cycle*self.__pi)**2/(cycle*self.__pi),axis=0)
        p_value = gammaincc(2.5, chi_square/2)
        return {'p-value': p_value, 'chi^2': chi_square, 'cycles': cycle}
    
    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,State,,{}\n".format(
            np.array2string(self.__stat, separator=',,'))
        csv += ",,Proportion,,{}\n".format(
            np.array2string(res['Proportion'], separator=',,'))
        csv += ",,Uniformity,,{}\n".format(
            np.array2string(res['Uniformity'], separator=',,'))
        csv += ",,SequenceID,Number of cycles"\
            "{}\n".format(",p-value,chi_square"*self.__stat.size)
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{}".format(i, res['cycles'][i])
                for j in range(self.__stat.size):
                    csv += ",{},{}".format(
                        res['p-value'][i][j], res['chi^2'][i][j])
                csv += "\n"
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv.replace('[','').replace(']','')


class RandomExcursionsVariantTest(STS):
    """Random Excursions Variant Test

    Attributes
    ----------
    ID : `Enum`
        A unique identifier for the class.
    
    Examples
    --------
    >>> test = RandomExcursionsVariantTest(
            file="file.txt",
            fmt=RandomExcursionsVariantTest.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    ID = STS.TestID.EXCURSIONSVAR

    def __init__(self, file: str, fmt: STS.ReadAs, seq_len: int,
            seq_num: int, init: bool =True):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            RandomExcursionsVariantTest.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each split sequence.
        
        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        init : `bool`, optional
            If `True`, initialize the super class.
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.
        
        """
        if init:
            super().__init__(file, fmt, seq_len, seq_num)
        self.__stat = np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9])
        self.__low_lim = max(500, 0.005*sqrt(seq_len))
    
    @property
    def param(self) -> dict:
        """Return the local parameters for each test as a dictionary."""
        return {'Lower limit of cycles' : self.__low_lim}

    def func(self, bits) -> dict:
        """Evaluate the total number of times that
        a particular state is visited in a random walk.
        
        Random walk is defined by the cumulative sum
        of adjusted (`-1`, `+1`) digits in the sequence.

        Parameters
        ----------
        bits : `1d-ndarray int8`
            Binary sequence to be tested.
        
        Returns
        -------
        `dict` Name and data pairs.

            {
                'p-value': ndarray Test result for each state.
                'xi'     : Number of occurrences of each state.
            }
        
        Raise
        -----
        StatisticalError :
            When significantly biased statistics are calculated.
        
        """
        bits = 2*bits - 1
        s = np.pad(np.cumsum(bits), (1,1))
        idx = np.where(s==0)[0]
        cycle = idx.size - 1
        if cycle < self.__low_lim:
            msg = "Number of cycles is out of expected range."
            raise StatisticalError(msg, self.ID)
        xi = np.zeros_like(self.__stat)
        for i in range(self.__stat.size):
            xi[i] = np.count_nonzero(s==self.__stat[i])
        p_value = erfc(
            np.abs(xi-cycle) / np.sqrt(2*cycle*(4*np.abs(self.__stat)-2)))
        return {'p-value': p_value, 'xi': xi, 'cycles': cycle}

    def to_csv(self, res: dict) -> str:
        """Generate a CSV string from the partial test results."""
        csv = ",,State,,{}\n".format(
            np.array2string(self.__stat, separator=',,'))
        csv += ",,Proportion,,{}\n".format(
            np.array2string(res['Proportion'], separator=',,'))
        csv += ",,Uniformity,,{}\n".format(
            np.array2string(res['Uniformity'], separator=',,'))
        csv += ",,SequenceID,Number of cycles"\
            "{}\n".format(",p-value,Total visits"*self.__stat.size)
        for i in range(len(res['p-value'])):
            if res['err'][i] is None:
                csv += ",,{},{}".format(i, res['cycles'][i])
                for j in range(self.__stat.size):
                    csv += ",{},{}".format(
                        res['p-value'][i][j], res['xi'][i][j])
                csv += "\n"
            else:
                csv += ",,{},{}\n".format(i, res['err'][i])
        return csv.replace('[','').replace(']','')


class Multiple(STS):
    """spy800-22 Multiple Test class

    This is a wrapper that runs multiple tests sequentially.
    Note that you cannot access the local attributes of each test.

    Examples
    --------
    >>> test = Multiple(
            file="file.txt",
            fmt=Multiple.ReadAs.ASCII,
            seq_len=1000000,
            seq_num=1000)
    >>> test.run(proc_num=4, ig_err=True)
    >>> test.report("file.csv")
    
    """

    def __init__(self, file: str, fmt: STS.ReadAs,
            seq_len: int, seq_num: int, choice: list =None,
            blk_len_blockfrequency: int =128, tpl_len_nonoverlapping: int =9,
            tpl_len_overlapping: int =9, blk_len_complexity: int =500,
            blk_len_serial: int =16, blk_len_entropy: int =10):
        """Set the test parameters.

        Parameters
        ----------
        file : `str`
            The path of the file to read.
        
        fmt : `Enum`
            A method of converting each byte read from a file into bits.
            Specify the built-in `Enum` as follows.

            Multiple.ReadAs.xxx
            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        
        seq_len : `int`
            Bit length of each sequence.

        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.

        choice : `list of Enum`, optional
            IDs of the tests to run. If None, run all tests.
            Specify the built-in `Enum` as follows.
            
            Multiple.TestID.xxx
            FREQUENCY      : "Frequency (Monobit) Test"
            BLOCKFREQUENCY : "Frequency Test within a Block"
            RUNS           : "Runs Test"
            LONGESTRUN     : "Test for the Longest Run of Ones in a Block"
            RANK           : "Binary Matrix Rank Test"
            DFT            : "Discrete Fourier Transform (Spectral) Test"
            NONOVERLAPPING : "Non-overlapping Template Matching Test"
            OVERLAPPING    : "Overlapping Template Matching Test"
            UNIVERSAL      : "Maurer's Universal Statistical Test"
            COMPLEXITY     : "Linear complexity Test"
            SERIAL         : "Serial Test"
            ENTROPY        : "Approximate entropy Test"
            CUSUM          : "Cumulative sums (cusum) Test"
            EXCURSIONS     : "Random excursions Test"
            EXCURSIONSVAR  : "Random excursions variant Test"

        blk_len_blockfrequency : `int`, optional
            Block length in "Frequency Test within a Block".
        
        tpl_len_nonoverlapping : `int`, optional
            Template length in "Non-overlapping Template Matching Test".
        
        tpl_len_overlapping : `int`, optional
            Template length in "Overlapping Template Matching Test".
        
        blk_len_complexity : `int`, optional
            Block length in "Linear complexity Test".
        
        blk_len_serial : `int`, optional
            Block length in "Serial Test".
        
        blk_len_entropy : `int`, optional
            Block length in "Approximate entropy Test".
        
        Raise
        -----
        InvalidSettingError :
            When an invalid test parameter is set.
        
        """
        super().__init__(file, fmt, seq_len, seq_num)
        if choice is not None and not isinstance(choice, list):
            msg = "'choice' must be a list of Enums (Multiple.TestID.xxx)"\
                  " or None."
            raise InvalidSettingError(msg)
        elif choice is None:
            choice = [i for i in STS.TestID]

        tests = []
        if STS.TestID.FREQUENCY in choice:
            tests.append(
                FrequencyTest(file, fmt, seq_len, seq_num, init=False))
        if STS.TestID.BLOCKFREQUENCY in choice:
            tests.append(
                BlockFrequencyTest(file, fmt, seq_len, seq_num,
                    blk_len=blk_len_blockfrequency, init=False))
        if STS.TestID.RUNS in choice:
            tests.append(RunsTest(file, fmt, seq_len, seq_num, init=False))
        if STS.TestID.LONGESTRUN in choice:
            tests.append(
                LongestRunOfOnesTest(file, fmt, seq_len, seq_num, init=False))
        if STS.TestID.RANK in choice:
            tests.append(
                BinaryMatrixRankTest(file, fmt, seq_len, seq_num, init=False))
        if STS.TestID.DFT in choice:
            tests.append(
                DiscreteFourierTransformTest(
                    file, fmt, seq_len, seq_num, init=False))
        if STS.TestID.NONOVERLAPPING in choice:
            tests.append(
                NonOverlappingTemplateMatchingTest(file, fmt, seq_len, seq_num,
                    tpl_len=tpl_len_nonoverlapping, init=False))
        if STS.TestID.OVERLAPPING in choice:
            tests.append(
                OverlappingTemplateMatchingTest(file, fmt, seq_len, seq_num,
                    tpl_len=tpl_len_overlapping, init=False))
        if STS.TestID.UNIVERSAL in choice:
            tests.append(MaurersUniversalStatisticalTest(file, fmt,
                seq_len, seq_num, init=False))
        if STS.TestID.COMPLEXITY in choice:
            tests.append(LinearComplexityTest(file, fmt,
                seq_len, seq_num, blk_len=blk_len_complexity, init=False))
        if STS.TestID.SERIAL in choice:
            tests.append(
                SerialTest(file, fmt, seq_len, seq_num,
                    blk_len=blk_len_serial, init=False))
        if STS.TestID.ENTROPY in choice:
            tests.append(
                ApproximateEntropyTest(file, fmt, seq_len, seq_num,
                    blk_len=blk_len_entropy, init=False))
        if STS.TestID.CUSUM in choice:
            tests.append(
                CumulativeSumsTest(file, fmt,seq_len, seq_num, init=False))
        if STS.TestID.EXCURSIONS in choice:
            tests.append(
                RandomExcursionsTest(file, fmt, seq_len, seq_num, init=False))
        if STS.TestID.EXCURSIONSVAR in choice:
            tests.append(
                RandomExcursionsVariantTest(
                    file, fmt, seq_len, seq_num, init=False))
        
        if len(tests) < 1:
            msg = "No tests were selected."\
                  " 'choice' must be a list of Enums (Multiple.TestID.xxx)"\
                  " or None."
            raise InvalidSettingError(msg)

        self._STS__tests = tests
