#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
"""Implementation of SP800-22 test algorithms by Python.

This is a core module of the spy800-22 package.
It consists of 1 Base class `STS` and 7 Error classes.

Details of NIST SP800-22:
https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

NIST's official implementation:
https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software

"""

import os
from datetime import datetime
from enum import Enum, IntEnum, auto
import multiprocessing as mp
import numpy as np
from math import sqrt
from scipy.special import gammaincc

class STS:
    """spy800-22 base class

    This is a superclass inherited by all classes in the spy800-22 package.
    Implement the functions commonly used by each class (Bits input, parallel
    testing, error monitoring, summary and output of test results, etc.).

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

    def __init__(self, seq_len: int, seq_num: int, proc_num: int =1,
            ig_err: bool =False) -> None:
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
        
        """
        if seq_len < 1 or seq_num < 1 or proc_num < 1:
            msg = "Parameters must be at least 1."
            raise InvalidSettingError(msg)
        self.__sequence_len = int(seq_len)
        self.__sequence_num = int(seq_num)
        self.__sequence = None
        self.__process_num = proc_num
        if self.__process_num > mp.cpu_count():
            self.__process_num = mp.cpu_count()
        self.__ig_err = ig_err
        self.__tests = [self]
        self.__input_path = None
        self.__start_time = None
        self.__end_time = None
        self.__is_ready = False
        self.__is_tested = False
        self.__is_assessed = False
        self.__results = None
        np.set_printoptions(linewidth=100000)
    
    @property
    def sequence(self):
        """`ndarray uint`: Binary sequence."""
        return self.__sequence
    
    @property
    def results(self):
        """`list`: Test results."""
        return self.__results
    
    @property
    def param(self) -> None:
        """Return the local parameters for each test as a dictionary."""
        return None

    def load_bits(self, file_path: str, fmt: Enum) -> None:
        """Read data from a file and convert it to a binary sequence.

        Parameters
        ----------
        file_path : `str`
            The path of the file to read.
        fmt : `Enum`
            A method of converting data into bits.
            Specify the built-in `Enum`. -> `instance.ReadAs.xxx`

            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.

        """
        if not os.path.isfile(file_path):
            msg = "File \"{}\" is not found.".format(file_path)
            raise InvalidSettingError(msg)
        print("Loading bits...", end="")
        self.__sequence = np.zeros(
            self.__sequence_len*self.__sequence_num, dtype='int8')

        total_bits = os.path.getsize(file_path)
        if fmt == STS.ReadAs.BIGENDIAN or fmt == STS.ReadAs.LITTLEENDIAN:
            total_bits *= 8
        if total_bits < self.__sequence.size:
            msg = "Set value ({} x {}) exceeds the bits read ({}).".format(
                self.__sequence_num, self.__sequence_len, total_bits)
            raise BitShortageError(msg)

        if fmt == STS.ReadAs.ASCII:
            self.__read_bits_in_ascii_format(file_path)
        elif fmt == STS.ReadAs.BYTE:
            self.__read_bits_in_byte_format(file_path)
        elif fmt == STS.ReadAs.BIGENDIAN:
            self.__read_bits_in_binary_format(file_path)
        elif fmt == STS.ReadAs.LITTLEENDIAN:
            self.__read_bits_in_binary_format(file_path, reverse=True)
        else:
            msg = "File input mode must be Enum. -> instance.READ_AS.xxx"
            raise InvalidSettingError(msg)

        self.__input_path = file_path
        self.__sequence = np.resize(
            self.__sequence, (self.__sequence_num, self.__sequence_len))
        self.__is_ready = True
        print("\r{} bits loaded.".format(self.__sequence.size))
    
    def __read_bits_in_ascii_format(self, file_path: str) -> None:
        """Read data and convert it to a binary sequence."""
        with open(file_path, mode='r') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), "")):
                if n >= self.__sequence.size:
                    return
                if byte == "0" or byte == "1":
                    self.__sequence[n] = byte
                else:
                    msg = "Data with a different format was detected.\n"\
                        "Detected: \"{}\", Position: {}".format(byte, n)
                    raise IllegalBitError(msg)

    def __read_bits_in_byte_format(self, file_path: str) -> None:
        """Read data and convert it to a binary sequence."""
        with open(file_path, mode='rb') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), b'')):
                if n >= self.__sequence.size:
                    return
                if byte == b'\x00' or byte == b'\x01':
                    self.__sequence[n] = int.from_bytes(byte, 'big')
                else:
                    msg = "Data with a different format was detected.\n"\
                        "Detected: \"{}\", Position: {}".format(byte, n)
                    raise IllegalBitError(msg)
    
    def __read_bits_in_binary_format(
            self, file_path: str, reverse: bool =False) -> None:
        """Read data and convert it to a binary sequence."""
        n = 0  # Bit counter
        with open(file_path, mode='rb') as f:
            for byte in iter(lambda:f.read(1), b''):
                bits = int.from_bytes(byte, 'big')
                for i in range(8):
                    if n >= self.__sequence.size:
                        return
                    if reverse:  # Little-endian
                        self.__sequence[n] = (bits >> i) & 1
                    else:  # Big-endian
                        self.__sequence[n] = (bits >> (7-i)) & 1
                    n += 1
    
    def run(self) -> None:
        """Run the test.
        
        If the `processes_num` is set to 2 or more in the instantiation,
        the test will be parallelized.

        """
        if not self.__is_ready:
            print("No bits have been loaded. Unable to start test.")
            return
        if self.__is_tested:
            print("Test is over.")
            return

        self.__start_time = datetime.now()
        print("Test in progress. ", end="")
        args = []
        for test in self.__tests:
            for seq_id in range(self.__sequence_num):
                args.append((test, seq_id))
        results = []
        max_progress = len(args)
        progress = 0
        with mp.Pool(processes=self.__process_num) as p:
            for result in p.imap_unordered(self.test_wrapper, args):
                results.append(result)
                progress += 1
                print("\rTest in progress. |{:<50}|"
                    .format("█"*int(50*progress/max_progress)), end="")
        self.__sort_results(results)
        print("\rTest completed.{}".format(" "*55))
        self.__end_time = datetime.now()
        self.__is_tested = True
    
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
            Test result.
            [testID, sequenceID, results, Error(when it occurs)]
        
        """
        test, seq_id = args
        result = [test.ID, seq_id, None, None]
        try:
            res = test.func(self.__sequence[seq_id])
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
        """Sort the test results by test ID and sequence ID."""
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
    
    def assess(self) -> None:
        """
        """
        if not self.__is_tested:
            print("Test not completed. Unable to start assessment.")
            return
        if self.__is_assessed:
            print("Assessment is over.")
            return
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
        """Calcurate the proportion of passing sequences.

        """
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
        """Calcurate the uniformity of p_values.

        """
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
    
    def report(self, file_path: str) -> None:
        """Generate and save CSV of test results.

        Note that if a file with the same path already exists,
        it will be overwritten.

        Parameters
        ----------
        file_path : `str`
            Destination directory and file name.
        
        """
        if not self.__is_finished:
            csv = "Cannot make report because the test has not been completed."
            raise InvalidProceduralError(csv)

        with open(file_path, mode='w') as f:
            csv = "SP800-22 Test Report\n\n"
            csv += ",Start,{}".format(
                self.__start_time.strftime('%Y/%m/%d,%H:%M:%S'))
            csv += "\n,End,{}\n".format(
                self.__end_time.strftime('%Y/%m/%d,%H:%M:%S'))
            csv += ",File,{}\n".format(self.__input_path)
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
