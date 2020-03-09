#!/usr/bin/env python3
# -*- Coding: utf-8 -*-


# sts.py
#
# Copyright (c) 2020 Takuya Kawashima
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php


"""Implementation of SP800-22 test algorithms by Python.

This is a core module of the spy800-22 package.
It consists of 1 base class `STS` and 6 error classes.
These error classes consist of 5 derived classes based on `STSError`.

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

import multiprocessing as mp
import os
from datetime import datetime
from enum import Enum, IntEnum, auto
from math import ceil, sqrt

import numpy as np
from scipy.special import gammaincc


class STS:
    """spy800-22 base class

    This is a class inherited by all classes in the spy800-22 package.
    Implement the functions commonly used by each class (Bits input, parallel
    testing, summary and output of test results, etc.).

    Attributes
    ----------
    ALPHA : `float`
        Significance level for p-value in all statistical tests.
    UNIF_LIM : `float`
        Significance level for uniformity (P-valueT) in final assessment.

    """

    class TestID(IntEnum):
        """Test identifier."""

        FREQUENCY = auto()
        BLOCKFREQUENCY = auto()
        RUNS = auto()
        LONGESTRUN = auto()
        RANK = auto()
        DFT = auto()
        NONOVERLAPPING = auto()
        OVERLAPPING = auto()
        UNIVERSAL = auto()
        COMPLEXITY = auto()
        SERIAL = auto()
        ENTROPY = auto()
        CUSUM = auto()
        EXCURSIONS = auto()
        EXCURSIONSVAR = auto()

        def to_name(self):
            if self is self.FREQUENCY:
                return "Frequency (Monobit) Test"
            if self is self.BLOCKFREQUENCY:
                return "Frequency Test within a Block"
            if self is self.RUNS:
                return "Runs Test"
            if self is self.LONGESTRUN:
                return "Test for the Longest Run of Ones in a Block"
            if self is self.RANK:
                return "Binary Matrix Rank Test"
            if self is self.DFT:
                return "Discrete Fourier Transform (Spectral) Test"
            if self is self.NONOVERLAPPING:
                return "Non-overlapping Template Matching Test"
            if self is self.OVERLAPPING:
                return "Overlapping Template Matching Test"
            if self is self.UNIVERSAL:
                return "Maurer's \"Universal Statistical\" Test"
            if self is self.COMPLEXITY:
                return "Linear complexity Test"
            if self is self.SERIAL:
                return "Serial Test"
            if self is self.ENTROPY:
                return "Approximate entropy Test"
            if self is self.CUSUM:
                return "Cumulative Sums (Cusum) Test"
            if self is self.EXCURSIONS:
                return "Random Excursions Test"
            if self is self.EXCURSIONS:
                return "Random Excursions Test"
            if self is self.EXCURSIONSVAR:
                return "Random Excursions Variant Test"
            

    class ReadAs(Enum):
        """Bits read mode specifier."""
        ASCII = auto()
        BYTE = auto()
        BIGENDIAN = auto()
        LITTLEENDIAN = auto()

    ALPHA = 0.01  # Significance level for p-value
    UNIF_LIM = 0.0001  # Significance level for uniformity (P-valueT)

    def __init__(self, file: str, fmt: Enum, seq_len: int, seq_num: int):
        """Set the test parameters."""
        if seq_len < 1 or seq_num < 1:
            msg = "Length and number of sequence must be at least 1."
            raise InvalidSettingError(msg)
        self.__file = file
        self.__fmt = fmt
        self.__sequence_len = int(seq_len)
        self.__sequence_num = int(seq_num)
        self.__seq_bytes = ceil(self.__sequence_len / 8)
        self.__total_sequence_size = self.__sequence_num*self.__sequence_len
        self.check_file()
        self.__tests = [self]
        self.__is_finished = False
    
    @property
    def results(self) -> dict:
        """`dict`: Test results."""
        return self.__results
    
    @property
    def param(self) -> None:
        """`dict`: Local parameters of test."""
        return None
    
    def check_file(self) -> None:
        """Check whether a file is testable."""
        print("Checking file...", end="")
        if not os.path.isfile(self.__file):
            msg = "File \"{}\" is not found.".format(self.__file)
            raise InvalidSettingError(msg)
        self.__check_bits_num()
        if self.__fmt == STS.ReadAs.ASCII or self.__fmt == STS.ReadAs.BYTE:
            self.__check_format()
        print("\r{} is testable.".format(self.__file))
    
    def __check_bits_num(self) -> None:
        """Check the number of bits in the file."""
        total_bits = os.path.getsize(self.__file)
        if (self.__fmt == STS.ReadAs.BIGENDIAN
                or self.__fmt == STS.ReadAs.LITTLEENDIAN):
            total_bits *= 8
        if total_bits < self.__total_sequence_size:
            msg = "Set value ({} x {}) exceeds the bits read ({}).".format(
                self.__sequence_num, self.__sequence_len, total_bits)
            raise BitShortageError(msg)
    
    def __check_format(self) -> None:
        """Check the format of the bits."""
        if self.__fmt == STS.ReadAs.ASCII:
            zero, one, end = "0", "1", ""
            f = open(self.__file, mode='r')
        elif self.__fmt == STS.ReadAs.BYTE:
            zero, one, end = b'\x00', b'\x01', b''
            f = open(self.__file, mode='rb')
        for n, byte in enumerate(iter(lambda:f.read(1), end)):
            if n >= self.__total_sequence_size:
                f.close()
                return
            if byte != zero and byte != one:
                msg = "Data with a different format was detected.\n"\
                    "Detected: \"{}\", Position: {}".format(byte, n)
                f.close()
                raise IllegalBitError(msg)
    
    def load_sequence(self, seq_id: int) -> np.ndarray:
        """Read data and convert it to a binary sequence."""
        if self.__fmt == STS.ReadAs.ASCII or self.__fmt == STS.ReadAs.BYTE:
            return self.__byte_to_1bit(seq_id)
        elif (self.__fmt == STS.ReadAs.BIGENDIAN
                or self.__fmt == STS.ReadAs.LITTLEENDIAN):
            return self.__byte_to_8bits(seq_id)

    def __byte_to_1bit(self, seq_id: int) -> np.ndarray:
        """Read in ASCII or BYTE format."""
        if self.__fmt == STS.ReadAs.ASCII:
            f = open(self.__file, mode='r')
        elif self.__fmt == STS.ReadAs.BYTE:
            f = open(self.__file, mode='rb')
        f.seek(seq_id*self.__sequence_len)
        seq = f.read(self.__sequence_len)
        f.close()
        return np.array(list(seq), dtype='int8')
    
    def __byte_to_8bits(self, seq_id: int) -> np.ndarray:
        """Read in BIGENDIAN or LITTLEENDIAN."""
        seek, head_ext = divmod(seq_id*self.__sequence_len, 8)
        tail_ext = 8*self.__seq_bytes - (head_ext + self.__sequence_len)
        with open(self.__file, mode='rb') as f:
            f.seek(seek)
            seq = f.read(self.__seq_bytes)
        seq = np.array(list(seq), dtype='uint8')
        if self.__fmt == STS.ReadAs.BIGENDIAN:
            seq = np.unpackbits(seq)
        elif self.__fmt == STS.ReadAs.LITTLEENDIAN:
            seq = np.unpackbits(seq, bitorder='little')
        return seq[head_ext:-tail_ext].astype('int8')
    
    def run(self, proc_num: int =1, ig_err: bool =False) -> None:
        """Run the test.

        Parameters
        ----------
        proc_num : `int`, optional
            Number of processes for running tests in parallel.
        
        ig_err : `bool`, optional
            If True, ignore any errors that occur during test execution.
        
        """
        if self.__is_finished:
            print("Testing is already over.")
            return
        if proc_num < 1:
            proc_num = 1
        if proc_num > mp.cpu_count():
            proc_num = mp.cpu_count()
        self.__ig_err = bool(ig_err)

        self.__start_time = datetime.now()
        print("Test in progress. ", end="")
        args = []
        for test in self.__tests:
            for seq_id in range(self.__sequence_num):
                args.append((test, seq_id))
        results = []
        max_progress = len(args)
        progress = 0
        with mp.Pool(processes=proc_num) as p:
            for result in p.imap_unordered(self.test_wrapper, args):
                results.append(result)
                progress += 1
                print("\rTest in progress. |{:<50}|"
                    .format("█"*int(50*progress/max_progress)), end="")
        self.__sort_results(results)
        self.__assess()
        print("\rTest completed.{}".format(" "*55))
        self.__end_time = datetime.now()
        self.__is_finished = True
    
    def test_wrapper(self, args: list) -> list:
        """Wrapper function for parallel testing.

        It also catches and re-throws exceptions.
        
        Parameters
        ----------
        args : `list`
            Contains the test instance and sequence ID.
        
        Returns
        -------
        result : `list`
            List of test results and tags to identify them.
        
        """
        test, seq_id = args
        result = [test.ID, seq_id, None, None]
        try:
            seq = self.load_sequence(seq_id)
            res = test.func(seq)
        except StatisticalError as err:
            result[2] = "StatisticalError"
            if not self.__ig_err:
                raise
        except ZeroDivisionError as err:
            result[2] = "ZeroDivisionError"
            if not self.__ig_err:
                msg = "Division by zero detected."
                raise ComputationalError(msg, test.ID)
        else:
            result[3] = res
        return result
    
    def __sort_results(self, results: list) -> None:
        """Sort the test results and generate data dictionary."""
        results.sort(key=lambda x: x[1])  # by sequence ID
        results.sort()  # by test ID
        self.__results = {}
        prev_id = None
        e = 0
        for r in results:
            if r[0] != prev_id:
                self.__results[r[0]] = {'err': [], 'p-value': []}
            self.__results[r[0]]['err'].append(r[2])
            if r[2] is not None:
                self.__results[r[0]]['p-value'].append(None)
                e += 1
            else:
                for k, v in r[3].items():
                    if k != 'p-value':
                        self.__results[r[0]].setdefault(k,[]).extend([None]*e)
                    self.__results[r[0]].setdefault(k,[]).append(v)
                e = 0
            prev_id = r[0]
    
    def __assess(self) -> None:
        """Final assessment based on NIST guidelines."""
        key = self.__results.keys()
        for k in key:
            res0 = self.__calc_proportion(self.__results[k]['p-value'])
            res1 = self.__calc_uniformity(res0[3])
            self.__results[k]['Proportion'] = res0[0]
            self.__results[k]['PropLim'] = res0[1]
            self.__results[k]['Passed'] = res0[2]
            self.__results[k]['Uniformity'] = res1[0]
            self.__results[k]['Histogram'] = res1[1]

    def __calc_proportion(self, p):
        """Calcurate the proportion of passing sequences."""
        p_values = None
        err_idx = []
        for i, j in enumerate(p):
            if j is not None:
                if p_values is None:
                    if type(j) is np.ndarray:
                        p_values = np.zeros((len(p), j.size))
                    else:
                        p_values = np.zeros((len(p), 1))
                p_values[i] = j
            else:
                err_idx.append(i)
        p_values = np.delete(p_values, obj=err_idx, axis=0)
        n = p_values.shape[0]
        p_hat = 1 - STS.ALPHA
        prop_lim = np.array([-3.,3.])*sqrt(p_hat*STS.ALPHA/n) + p_hat
        passed = np.count_nonzero(p_values >= STS.ALPHA, axis=0)
        prop = passed / n
        return prop, prop_lim, passed, p_values
    
    def __calc_uniformity(self, p_vals):
        """Calcurate the uniformity of p_values."""
        hist = np.zeros((p_vals.shape[1],10), dtype=int)
        if p_vals.shape[0] == 0:
            return 0.0, hist
        for i in range(p_vals.shape[1]):
            hist[i] = np.histogram(p_vals[:,i], range=(0,1))[0]
            hist[-1] += np.count_nonzero(p_vals[:,i] > 1)
        chi_square = np.sum(
            (hist - p_vals.shape[0]/10)**2 / (p_vals.shape[0]/10), axis=1)
        unif = gammaincc(9/2, chi_square/2)
        return unif, hist
    
    def report(self, file: str) -> None:
        """Generate and save CSV of test results.

        Note that if a file with the same path already exists,
        it will be overwritten.

        Parameters
        ----------
        file_path : `str`
            Destination directory and file name.
        
        """
        if not self.__is_finished:
            print("Test not completed. Unable to make report.")
            return
        np.set_printoptions(linewidth=100000)
        with open(file, mode='w') as f:
            csv = "SP800-22 Test Report\n\n"
            csv += ",Start,{}".format(
                self.__start_time.strftime('%Y/%m/%d,%H:%M:%S'))
            csv += "\n,End,{}\n".format(
                self.__end_time.strftime('%Y/%m/%d,%H:%M:%S'))
            csv += ",File,{}\n".format(self.__file)
            csv += ",Sequence length,{}\n".format(self.__sequence_len)
            csv += ",Number of sequence,{}\n".format(self.__sequence_num)
            csv += ",Significance level,{}\n".format(STS.ALPHA)
            f.write(csv)
            csv = "\n\nSummary\n\n"
            if self.__sequence_num < 55 or self.__sequence_len < 1000:
                csv += ",Warning:,Proportion and Uniformity values are "\
                    "unreliable due to insufficient sequence length "\
                    "or number of sequences.\n"
            csv += ",Note:,REJECTED is displayed next to the results "\
                "for which the Proportion or Uniformity is below "\
                "the reference value.\n"
            csv += ",Note:,For some tests with multiple results. "\
                "the result with the lowest Proportion and the "\
                "result with the lowest Uniformity are displayed.\n"
            csv += "\n,Test name,Proportion,,Uniformity,,Histogram of p-value"
            csv += ",0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9\n"
            for test in self.__tests:
                prop = self.__results[test.ID]['Proportion']
                prop_lim = self.__results[test.ID]['PropLim'][0]
                unif = self.__results[test.ID]['Uniformity']
                hist = self.__results[test.ID]['Histogram']
                if prop.size > 1:
                    idx = [np.argmin(prop), np.argmin(unif)]
                    if idx[0] == idx[1]:
                        idx = [idx[0]]
                else:
                    idx = [0]
                for i in idx:
                    csv += ",{},{},{},{},{},,{}\n".format(test.ID.to_name(),
                        prop[i], "REJECTED :(" if prop[i] < prop_lim else "",
                        unif[i], "REJECTED :(" if unif[i] < STS.UNIF_LIM else "",
                        np.array2string(hist[i], separator=','))
            f.write(csv.replace('[','').replace(']',''))
            f.write("\n\nComputational Information\n\n")
            for test in self.__tests:
                csv = ",{}\n".format(test.ID.to_name())
                head = test.param
                if head is not None:
                    for k, v in head.items():
                        if type(v) is np.ndarray:
                            csv += ",,{},{}\n".format(
                                k, np.array2string(v, separator=','))
                        else:
                            csv += ",,{},{}\n".format(k,v)
                csv += ",,Error count,{}\n".format(self.__sequence_num
                    - self.__results[test.ID]['err'].count(None))
                csv += (",,Confidence interval for Proportion,{0[0]},{0[1]}\n"
                    .format(self.__results[test.ID]['PropLim']))
                csv += (",,Significance level for Uniformity,{}\n"
                    .format(STS.UNIF_LIM))
                csv += test.to_csv(self.__results[test.ID])
                csv += "\n"
                f.write(csv.replace('[','').replace(']',''))


class STSError(Exception):
    """Base exception class for STS.
    
    All exceptions thrown from the package inherit this.

    Attributes
    ----------
    msg : `str`
        Human readable string describing the exception.
    
    """

    def __init__(self, msg: str, test_id: Enum =None):
        """Set the error message.

        Parameters
        ----------
        test_id : `Enum`
            Specify the built-in Enum. -> `self.TestID.xxx`
        msg : `str`
            Human readable string describing the exception.
        
        """
        self.msg = ""
        if test_id is not None:
            self.msg += test_id.to_name() + "\n"
        self.msg += msg
    
    def __str__(self):
        """Return the error message."""
        return self.msg

class InvalidSettingError(STSError):
    """Raised when an invalid test parameter is set."""

class BitShortageError(STSError):
    """Raised when bits in the file are less than user setting."""

class IllegalBitError(STSError):
    """Raised when data different from user setting format is read."""

class StatisticalError(STSError):
    """Raised when significantly biased statistics are calculated."""

class ComputationalError(STSError):
    """Raised when an incorrect calculation is detected."""
