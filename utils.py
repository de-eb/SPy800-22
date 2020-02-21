import os
import sys
import csv
import time
import psutil
import numpy as np
from importlib import import_module
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
    NIST SP800-22 rev.1a (STS2.1.2) manager class
    
    Parameters
    ----------
    path : str
        Path of the file to be tested.
    fmt : int
        Specifies how each byte in the file is converted to bits. If 0, it
        converts 0x30 and 0x31 to 0 and 1 (that is, ASCII characters 0 or 1).
        If 1, convert 0x00 and 0x01 to 0 and 1, respectively. If 2 or 3, each
        bit is used as it is. If 2, each byte is read from LSB (little
        endian). If 3, each byte is read from MSB (big endian).
    seq_num : int
        Number of sequences to be tested individually.
    seq_size : int
        Bit length of each sequence.
    proc_num : int
        Number of processes for running tests in parallel.
    choice : tuple of int
        A tuple of IDs of the tests to run. If None, run all tests.(0-monobit,
        1-block_frequency, 2-runs, 3-longest_run, 4-rank, 5-dft,
        6-non_overlapping, 7-overlapping, 8-universal, 9-complexity, 10-serial,
        11-entropy, 12-cusums, 13-excursions, 14-excursions_variant)
    bs_freq : int
        Block size in "block frequency" test.
    bs_notm : int
        Block size in "non-overlapping template matching" test.
    bs_otm : int
        Block size in "overlapping template matching" test.
    bs_comp : int
        Block size in "liner complexity" test.
    bs_ser : int
        Block size in "serial" test.
    bs_apen : int
        Block size in "approximate entropy" test.
    """
    # Test sources
    SRC = [ 'monobit', 'frequency', 'runs', 'longest_run', 'rank', 'dft',
            'non_overlapping', 'overlapping', 'universal', 'complexity',
            'serial', 'entropy', 'cusum', 'excursions', 'excursions_var']
    NAMES = [  # Test Specifiers
        {'func' : "monobit",         'title' : "Frequency (monobit)"},
        {'func' : "frequency",       'title' : "Frequency within a block"},
        {'func' : "runs",            'title' : "Runs"},
        {'func' : "longest_run",     'title' : "Longest run of ones in a block"},
        {'func' : "rank",            'title' : "Binary matrix rank"},
        {'func' : "dft",             'title' : "Discrete fourier transform (spectral)"},
        {'func' : "non_overlapping", 'title' : "Non-overlapping template matching"},
        {'func' : "overlapping",     'title' : "Overlapping template matching"},
        {'func' : "universal",       'title' : "Maurer\'s \"universal statistical\""},
        {'func' : "complexity",      'title' : "Linear complexity"},
        {'func' : "serial",          'title' : "Serial"},
        {'func' : "entropy",         'title' : "Approximate entropy"},
        {'func' : "cusum",           'title' : "Cumulative sums (Cusum)"},
        {'func' : "excursions",      'title' : "Random excursions"},
        {'func' : "excursions_var",  'title' : "Random excursions variant"}
    ]
    # Input file formats
    FORMATS = ['ASCII_01', 'Byte_01', 'Little-endian', 'Big-endian']

    
    def __init__(self, path, fmt, seq_num, seq_size, proc_num=1, choice=None,
        bs_freq=128, bs_notm=9, bs_otm=9, bs_comp=500, bs_ser=16, bs_apen=10):
        """Constructor
        """
        if not os.path.isfile(path):
            msg = "File \"{}\" is not found.".format(path)
            raise InvalidSettingError(msg)
        self.file_path = path

        if fmt not in range(len(STS.FORMATS)):
            msg = "File format must be between 0 and {}.".format(len(STS.FORMATS))
            raise InvalidSettingError(msg)
        self.file_format = fmt

        if seq_num < 1:
            msg = "Number of sequences must be 1 or more."
            raise InvalidSettingError(msg)
        if seq_size < 1000:
            msg = "Sequence length must be 1000 bits or more."
            raise InvalidSettingError(msg)
        if seq_num*seq_size >= psutil.virtual_memory().available*0.8:
            msg = "Memory may be insufficient. Reduce the number or length of the sequence."
            raise InvalidSettingError(msg)
        self.sequence_num = seq_num
        self.sequence_size = seq_size

        if not (0 < proc_num <= os.cpu_count()):
            msg = "Number of processes must be between 1 and {}".format(os.cpu_count())
            raise InvalidSettingError(msg)
        self.process_num = proc_num

        if choice is None:
            self.test_choice = {i for i in range(len(STS.NAMES))}
        elif type(choice) == 'tuple':
            if not set(choice) <= {i for i in range(len(STS.NAMES))}:
                msg = "Test choice must be between 0 and 14."
                raise InvalidSettingError(msg)
            else:
                self.test_choice = set(choice)
        else:
            msg = "Test choice must be specified as None or tuple."
            raise InvalidSettingError(msg)

        if min([bs_freq, bs_notm, bs_otm, bs_comp, bs_ser, bs_apen]) < 1:
            msg = "Block sizes must be 1 or more."
            raise InvalidSettingError(msg)
        self.block_sizes = {1 : bs_freq, 6 : bs_notm, 7 : bs_otm,
                            9 : bs_comp, 10 : bs_ser, 11 : bs_apen}
    
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
            print("      {:<2}: {}".format(i, STS.NAMES[i-1]['title']))
        print("\n    Block sizes\n")
        print("      {:<34}: {}".format(STS.NAMES[1]['title'], self.block_size_freq))
        print("      {:<34}: {}".format(STS.NAMES[6]['title'], self.block_size_notm))
        print("      {:<34}: {}".format(STS.NAMES[7]['title'], self.block_size_otm))
        print("      {:<34}: {}".format(STS.NAMES[9]['title'], self.block_size_comp))
        print("      {:<34}: {}".format(STS.NAMES[10]['title'], self.block_size_ser))
        print("      {:<34}: {}".format(STS.NAMES[11]['title'], self.block_size_apen))
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

    def invoke_test(self, test_id, seq_id):
        """
        """
        test = getattr(
            import_module("tests." + STS.NAMES[test_id-1]['func']),
            STS.NAMES[test_id]['func'])
        if test_id -1 == 1:
            results = test(self.sequences[seq_id], )

        try:
            success, p_val, p_val_list, msg = test(bits)
        except ZeroDivisionError as err:
            p_val = 0.0
            judge = "ERROR"
            message += "  ZeroDivisionError: {}\n".format(err)
            message += "  Time  : {} sec\n".format(time.time()-start_time)
            return [test_name, file_path, p_val, judge, message]
        else:
            message += msg
            if success:
                message += "  Result: Pass"
                judge = "PASS"
            else:
                message += "  Result: Fail"
                judge = "FAIL"
            if p_val is not None:
                message += "   P = {}\n".format(p_val)
            if p_val_list is not None:
                p_val = min(p_val_list)
                message += "   P = {}\n".format(p_val)
            message += "  Time  : {} sec\n".format(time.time()-start_time)
            return [test_name, file_path, p_val, judge, message]


    def wrapper_for_run(self, args):
        """A wrapper for parallel processing.
        """
        return self.invoke_test(*args)


    def run(self):
        """Run multiple tests for multiple binary sequences in parallel.
        """
        tasks = []
        for path in self.path_list:
            for test_name in self.run_list:
                tasks.append((path, test_name))

        print("\n Test start.\n")
        start_time = time.time()
        results = []
        with Pool(processes=self.process_num) as p:
            for result in p.imap_unordered(self.wrapper_for_run, tasks):
                results.append(result[:4])
                if self.show_details:
                    print(" Progress :  {} / {}".format(len(results), len(tasks)))
                    print(result[4])
                else:
                    print("\r |{:<50}|  {} / {}".format("█"*int(50*len(results)/len(tasks)),
                                                    len(results),
                                                    len(tasks)),
                                                    end="")
        print("\n\n Test completed.")
        print(" Time: {} sec".format(time.time()-start_time))
        results = self.__summarize(results)
        self.__save_result(results)


if __name__ == "__main__":

    # sts = STS(
    #     path = ".debug/bits.txt",
    #     fmt = 0,
    #     seq_num = 10,
    #     seq_size = 1000
    # )

    # sts.read_bits()

    a = None
    set()