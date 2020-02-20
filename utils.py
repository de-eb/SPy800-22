import os
import sys
import csv
import time
import psutil
import numpy as np
from multiprocessing import Pool


class STSError(Exception):
    """Base exception class for STS.
    """
    pass

class InputError(STSError):
    """Exception raised for errors in the input.
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class InvalidSettingError(STSError):
    """ Exception raised when an invalid test parameter is set.
    """
    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return(self.message)

class IllegalBitError(STSError):
    """ Exception raised when data different from the
        user setting format is read from the input file.
    """
    def __init__(self, fmt):
        self.format = fmt
        self.message = (
            "The file contains data different from the \"{}\" format."
            .format(STS.FORMATS[fmt])
        )
    
    def __str__(self):
        return self.message

class BitShortageError(STSError):
    """ Exception raised when the number of bits 
        in the input file is less than the user setting.
    """
    def __init__(self, read_bits, set_bits):
        self.read_bits = read_bits
        self.set_bits = set_bits
        self.message = (
            "The file contains only {} bits for the set value ({} bits)."
            .format(self.read_bits, self.set_bits)
        )
    
    def __str__(self):
        return self.message

class STS:
    """
    """
    NAMES = [  # Test Specifiers
        {'file' : "monobit",         'title' : "Frequency (monobit)"},
        {'file' : "frequency",       'title' : "Frequency within a block"},
        {'file' : "runs",            'title' : "Runs"},
        {'file' : "longest_run",     'title' : "Longest run of ones in a block"},
        {'file' : "rank",            'title' : "Binary matrix rank"},
        {'file' : "dft",             'title' : "Discrete fourier transform (spectral)"},
        {'file' : "non_overlapping", 'title' : "Non-overlapping template matching"},
        {'file' : "overlapping",     'title' : "Overlapping template matching"},
        {'file' : "universal",       'title' : "Maurer\'s \"universal statistical\""},
        {'file' : "complexity",      'title' : "Linear complexity"},
        {'file' : "serial",          'title' : "Serial"},
        {'file' : "entropy",         'title' : "Approximate entropy"},
        {'file' : "cusum",           'title' : "Cumulative sums (Cusum)"},
        {'file' : "excursions",      'title' : "Random excursions"},
        {'file' : "excursions_var",  'title' : "Random excursions variant"}
    ]
    FORMATS = ['ASCII_01', 'Byte_01', 'Little-endian', 'Big-endian']  # input modes

    
    def __init__(self, path, fmt, seq_num, seq_size, proc_num=1, choice=[0],
                    bs_freq=128, bs_notm=9, bs_otm=9, bs_comp=500, bs_ser=16, bs_apen=10):
        """Constructor
        """
        if not os.path.isfile(path):
            msg = "File \"{}\" is not found.".format(path)
            raise InvalidSettingError(msg)
        if fmt not in [0, 1, 2, 3]:
            msg = "File format must be specified as either 0, 1, 2, or 3."
            raise InvalidSettingError(msg)
        if seq_num < 1:
            msg = "Number of sequences must be 1 or more."
            raise InvalidSettingError(msg)
        if seq_size < 1000:
            msg = "Sequence length must be 1000 bits or more."
            raise InvalidSettingError(msg)
        if seq_num*seq_size >= psutil.virtual_memory().available*0.8:
            msg = "Memory may be insufficient. Reduce the number or length of the sequence."
            raise InvalidSettingError(msg)
        if not (0 < proc_num <= os.cpu_count()):
            msg = "Number of processes must be between 1 and {}".format(os.cpu_count())
            raise InvalidSettingError(msg)
        if not set(choice) <= set([i for i in range(16)]):
            msg = "Test choice must be specified as a list from 0 to 15."
            raise InvalidSettingError(msg)
        if min([bs_freq, bs_notm, bs_otm, bs_comp, bs_ser, bs_apen]) < 1:
            msg = "Block sizes must be 1 or more."
            raise InvalidSettingError(msg)
        self.file_path = path
        self.file_format = fmt
        self.sequence_num = seq_num
        self.sequence_size = seq_size
        self.process_num = proc_num
        self.block_size_freq = bs_freq
        self.block_size_notm = bs_notm
        self.block_size_otm = bs_otm
        self.block_size_comp = bs_comp
        self.block_size_ser = bs_ser
        self.block_size_apen = bs_apen
        if 0 in set(choice):
            self.test_choice = [i+1 for i in range(15)]
        else:
            self.test_choice = list(set(choice))

    def show_settings(self):
        """
        """
        print()
        print("**************************************************************************")
        print("  Test settings\n")
        print("    General\n")
        print("      File path            : {}".format(self.file_path))
        print("      File format          : {}".format(STS.FORMATS[self.file_format]))
        print("      Number of sequences  : {}".format(self.sequence_num))
        print("      Sequence size (bits) : {}".format(self.sequence_size))
        print("      Number of proccesses : {}".format(self.process_num))
        print("\n    List of tests to be run\n")
        for i in self.test_choice:
            print("      {:<2}: {}".format(i, self.NAMES[i-1]['title']))
        print("\n    Block sizes\n")
        print("      {:<34}: {}".format(self.NAMES[1]['title'], self.block_size_freq))
        print("      {:<34}: {}".format(self.NAMES[6]['title'], self.block_size_notm))
        print("      {:<34}: {}".format(self.NAMES[7]['title'], self.block_size_otm))
        print("      {:<34}: {}".format(self.NAMES[9]['title'], self.block_size_comp))
        print("      {:<34}: {}".format(self.NAMES[10]['title'], self.block_size_ser))
        print("      {:<34}: {}".format(self.NAMES[11]['title'], self.block_size_apen))
        print()
        print("**************************************************************************\n")
    

    def read_bits(self):
        """
        """
        self.sequences = np.empty(  # Bit streams
            shape = (self.sequence_num, self.sequence_size),
            dtype = 'uint8'
        )
        if self.file_format == 0:  # ASCII
            self.__read_bits_in_ascii_format()
        elif self.file_format == 1:  # Byte
            self.__read_bits_in_byte_format()
        else:  # Binary
            self.__read_bits_in_binary_format()
    
    def __read_bits_in_ascii_format(self):
        with open(self.file_path, mode='r') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), "")):
                if n > self.sequences.size -1:
                    return
                if byte == "0" or byte == "1":  # 0x30 or 0x31
                    row, col = divmod(n, self.sequences.shape[1])
                    self.sequences[row, col] = byte
                else:
                    raise IllegalBitError(self.file_format)
        raise BitShortageError(n+1, self.sequences.size)

    def __read_bits_in_byte_format(self):
        with open(self.file_path, mode='rb') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), b'')):
                if n > self.sequences.size -1:
                    return
                if byte == 0x00 or byte == 0x01:
                    row, col = divmod(n, self.sequences.shape[1])
                    self.sequences[row, col] = int.from_bytes(byte, 'big')
                else:
                    raise IllegalBitError(self.file_format)
        raise BitShortageError(n+1, self.sequences.size)
    
    def __read_bits_in_binary_format(self):
        n = 0  # Bit counter
        with open(self.file_path, mode='rb') as f:
            for byte in iter(lambda:f.read(1), b''):
                bits = int.from_bytes(byte, 'big')
                for i in range(8):
                    if n > self.sequences.size -1:
                        return
                    row, col = divmod(n, self.sequences.shape[1])
                    if self.file_format == 2:  # Little-endian
                        self.sequences[row, col] = (bits >> i) & 1
                    elif self.file_format == 3:  # Big-endian
                        self.sequences[row, col] = (bits >> (7 - i)) & 1
                    n += 1
        raise BitShortageError(n+1, self.sequences.size)


if __name__ == "__main__":

    sts = STS(
        path = ".debug/bits.txt",
        fmt = 0,
        seq_num = 10,
        seq_size = 1000
    )

    sts.read_bits()