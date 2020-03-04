#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
"""Implementation of SP800-22 test algorithms by Python.

This module is part of the spy800_22 package and inherited by other modules in
the package. It consists of 1 Base class `STS` and 7 Error classes.

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


class STS:
    """spy800-22 base class

    This is a superclass inherited by all classes in the spy800-22 package.
    Implement the functions commonly used by each class (Bits input, parallel
    testing, error monitoring, summary and output of test results, etc.).

    """

    class TestID(IntEnum):
        FREQUENCY = auto()
        BLOCKFREQUENCY = auto()
        RUNS = auto()
        LONGESTRUN = auto()
        # RANK = "Binary Matrix Rank Test"
        # DFT = "Discrete Fourier Transform (Spectral) Test"
        # NONOVERLAPPING = "Non-overlapping Template Matching Test"
        # OVERLAPPING = "Overlapping Template Matching Test"
        # UNIVERSAL = "Maurer's \"Universal Statistical\" Test"
        # COMPLEXITY = "Linear complexity Test"
        # SERIAL = "Serial Test"
        # ENTROPY = "Approximate entropy Test"
        # CUSUM = "Cumulative sums (cusum) Test"
        # EXCURSIONS = "Random excursions Test"
        # EXCURSIONSVAR = "Random excursions variant Test"

    class ReadAs(Enum):
        ASCII = auto()
        BYTE = auto()
        BIGENDIAN = auto()
        LITTLEENDIAN = auto()

    ALPHA = 0.01  # Significance level

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
        self.__is_finished = False
        self.__results = None
    
    @property
    def sequence_len(self):
        """`int`: Bit length of each split sequence."""
        return self.__sequence_len
    
    @property
    def sequence_num(self):
        """`int`: Number of sequences."""
        return self.__sequence_num
    
    @property
    def sequence(self):
        """`ndarray uint8`: Binary sequence."""
        return self.__sequence
    
    @property
    def process_num(self):
        """`int`: Number of processes for running tests in parallel."""
        return self.__process_num
    
    @property
    def is_ready(self):
        """`bool`: `True` if the test is executable, `False` otherwise."""
        return self.__is_ready
    
    @property
    def is_finished(self):
        """`bool`: `True` if test is complete, `False` otherwise."""
        return self.__is_finished
    
    @property
    def results(self):
        """`list`: Test results."""
        return self.__results

    def read_bits(self, file_path: str, fmt: Enum) -> None:
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
        self.__sequence = np.zeros(
            self.__sequence_len*self.__sequence_num, dtype='uint8')

        total_bits = os.path.getsize(file_path)
        if fmt == STS.ReadAs.BIGENDIAN or fmt == STS.ReadAs.LITTLEENDIAN:
            total_bits *= 8
        if total_bits < self.__sequence.size:
            raise BitShortageError(self.__sequence.size, total_bits)

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
    
    def run(self) -> None:
        """Run the test.
        
        If the `processes_num` is set to 2 or more in the instantiation,
        the test will be parallelized.

        """
        if not self.__is_ready:
            msg = "Cannot start test because the bits have not been read."
            raise InvalidProceduralError(msg)

        self.__start_time = datetime.now()
        print("\nTest in progress.\n")
        args = []
        for test in self.__tests:
            for seq_id in range(self.__sequence_num):
                args.append((test, seq_id))
        results = []
        with mp.Pool(processes=self.__process_num) as p:
            for result in p.imap_unordered(self.test_wrapper, args):
                results.append(result)
                print("\r |{:<50}|"
                    .format("█"*int(50*len(results)/len(args))), end="")
        self.__sort_results(results)
        self.__assess_results()
        print("\n\nTest completed.\n")
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
            Test result.
            [testID, sequenceID, results, Error(when it occurs)]
        
        """
        test, seq_id = args
        result = [test.ID, seq_id]
        try:
            res = test.func(self.__sequence[seq_id])
        except StatisticalError as err:
            res = list(err.args + ("StatisticalError",))
            if not self.__ig_err:
                msg = "{}: ".format(test.NAME) + err.msg
                raise StatisticalError(msg, None)
        except ZeroDivisionError as err:
            res = [0.0, "ZeroDivisionError"]
            if not self.__ig_err:
                msg = "{}: Zero Division detected.\n".format(test.NAME)
                raise ComputationalError(msg)
        result.extend(res)
        return result
    
    def save_report(self, file_path: str) -> None:
        """Generate and save CSV of test results.

        Note that if a file with the same path already exists,
        it will be overwritten.

        Parameters
        ----------
        file_path : `str`
            Destination directory and file name.
        
        """
        if not self.__is_finished:
            msg = "Cannot make report because the test has not been completed."
            raise InvalidProceduralError(msg)

        with open(file_path, mode='w') as f:
            f.write("NIST SP800-22 Test Report\n\n")
            f.write("Start,{}".format(
                self.__start_time.strftime('%Y/%m/%d,%H:%M:%S')))
            f.write("\nEnd,{}\n".format(
                self.__end_time.strftime('%Y/%m/%d,%H:%M:%S')))
            f.write("File,{}\n".format(self.__input_path))
            f.write("Sequence length,{}\n".format(self.__sequence_len))
            f.write("Number of sequence,{}\n".format(self.__sequence_num))

            f.write("\nSummary\n")

            f.write("\nComputational Information\n")
            for test in self.__tests:
                for res in self.__results:
                    if test.ID == res[0]:
                        f.write("\n{}\n".format(test.report(res[1])))
    
    def __read_bits_in_ascii_format(self, file_path: str) -> None:
        """Read data and convert it to a binary sequence."""
        with open(file_path, mode='r') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), "")):
                if n >= self.__sequence.size:
                    return
                if byte == "0" or byte == "1":
                    self.__sequence[n] = byte
                else:
                    raise IllegalBitError(STS.ReadAs.ASCII)

    def __read_bits_in_byte_format(self, file_path: str) -> None:
        """Read data and convert it to a binary sequence."""
        with open(file_path, mode='rb') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), b'')):
                if n >= self.__sequence.size:
                    return
                if byte == b'\x00' or byte == b'\x01':
                    self.__sequence[n] = int.from_bytes(byte, 'big')
                else:
                    raise IllegalBitError(STS.ReadAs.BYTE)
    
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
    
    def __sort_results(self, results: list) -> None:
        """Sort the test results by test ID and sequence ID."""
        results.sort(key=lambda x: x[1])  # by sequence ID
        results.sort()  # by test ID
        self.__results = []
        prev_id = None
        idx = -1
        for i in results:
            if i[0] != prev_id:
                self.__results.append([i[0],[]])
                idx += 1
            self.__results[idx][1].append(i[2:])
            prev_id = i[0]
    
    def __assess_results(self) -> None:
        return


class STSError(Exception):
    """Base exception class for STS.
    
    All exceptions thrown from the package inherit this.

    Attributes
    ----------
    msg : `str`
        Human readable string describing the exception.
    
    """

    def __init__(self, msg: str):
        """Set the error message.

        Parameters
        ----------
        msg : `str`
            Human readable string describing the exception.
        
        """
        self.msg = msg
    
    def __str__(self):
        """Return the error message."""
        return self.msg

class InvalidSettingError(STSError):
    """Raised when an invalid test parameter is set."""

class BitShortageError(STSError):
    """Raised when bits in the file are less than user setting."""

    def __init__(self, set_bits: int, read_bits: int):
        self.msg = (
            "The set value ({} bits) exceeds the bits read ({} bits)."
            .format(set_bits, read_bits))

class IllegalBitError(STSError):
    """Raised when data different from user setting format is read."""

    def __init__(self, fmt: Enum):
        if fmt == STS.ReadAs.ASCII:
            annotation = "0x30 or 0x31"
        elif fmt == STS.ReadAs.BYTE:
            annotation = "0x00 or 0x01"
        self.msg = (
            "Data in a format different from the setting ({}) was detected."
            .format(annotation))

class InvalidProceduralError(STSError):
    """Raised when methods are called in a incorrect order."""

class StatisticalError(STSError):
    """Raised when significantly biased statistics are calculated."""

    def __init__(self, msg: str, *args):
        self.msg = msg
        self.args = args

class ComputationalError(STSError):
    """Raised when an incorrect calculation is detected."""
