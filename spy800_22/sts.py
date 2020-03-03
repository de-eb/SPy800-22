import os
from datetime import datetime
from enum import Enum, IntEnum, auto
import multiprocessing as mp
import numpy as np


class STS:
    """
    STS base class
    ==============
    Define some methods to be used for all tests.
    
    Attributes
    ----------
    sequence_len : int
        Bit length of each sequence.
    
    sequence_num : int
        Number of sequences.
    
    sequence : ndarray uint8
        Sequence. None if not yet loaded.
    
    process_num : int
        Number of processes for running tests in parallel.
    
    is_ready : bool
        Whether the test can be performed.
    
    is_finished : bool
        Whether the test has been completed.
    
    results : list
        Test results. None if the test has not been completed.
    """
    class TestID(IntEnum):
        FREQUENCY = auto()
        BLOCKFREQUENCY = auto()
        # RUNS = "Runs Test"
        # LONGESTRUN = "Test for the Longest Run of Ones in a Block"
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

    def __init__(self, seq_len, seq_num, proc_num=1) -> None:
        """
        Parameters
        ----------
        seq_len : int
            Bit length of each sequence.

        seq_num : int
            Number of sequences.
            If more than 1, the loaded sequence is split and tested separately.
        
        proc_num : int
            Number of processes for running tests in parallel.
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
        self.__tests = [self]
        self.__input_path = None
        self.__start_time = None
        self.__end_time = None
        self.__is_ready = False
        self.__is_finished = False
        self.__results = None
    
    @property
    def sequence_len(self):
        return self.__sequence_len
    
    @property
    def sequence_num(self):
        return self.__sequence_num
    
    @property
    def sequence(self):
        return self.__sequence
    
    @property
    def process_num(self):
        return self.__process_num
    
    @property
    def is_ready(self):
        return self.__is_ready
    
    @property
    def is_finished(self):
        return self.__is_finished
    
    @property
    def results(self):
        return self.__results

    def read_bits(self, file_path: str, fmt)  -> None:
        """
        Read data from a file and convert it to a binary sequence.

        Parameters
        ----------
        file_path : str
            The path of the file to read.
        
        fmt : Enum
            A method of converting data into bits.
            Specify the built-in Enum. -> instance.ReadAs.xxx

            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        """
        if not os.path.isfile(file_path):
            msg = "File \"{}\" is not found.".format(file_path)
            raise InvalidSettingError(msg)

        self.__sequence = np.empty(
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
    
    def run(self):
        """
        Run the test. If the processes_num is set to 2 or more
        in the instantiation, the test will be parallelized.
        """
        if not self.__is_ready:
            msg = "Cannot start test because the bits have not been read."
            raise InvalidProceduralError(msg)

        self.__start_time = datetime.now()
        print("\nTest in progress.")
        args = []
        for test_num in range(len(self.__tests)):
            for seq_num in range(self.__sequence_num):
                args.append((test_num, seq_num))
        results = []
        with mp.Pool(processes=self.__process_num) as p:
            for result in p.imap_unordered(self.run_wrapper, args):
                results.append(result)
                print("\r |{:<50}|"
                    .format("█"*int(50*len(results)/len(args))), end="")
        self.__sort_results(results)
        self.__assess_results()
        print("\nTest completed.")
        self.__end_time = datetime.now()
        self.__is_finished = True
    
    def save_report(self, file_path):
        """
        Generate and save CSV of test results.
        Note that if a file with the same path already exists,
        it will be overwritten.

        Parameters
        ----------
        file_path : str
            File name including the path to the save directory.
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
            f.write("\nFile,{}\n".format(self.__input_path))
            f.write("Sequence length,{}\n".format(self.__sequence_len))
            f.write("Number of sequence,{}\n".format(self.__sequence_num))

            for test in self.__tests:
                for res in self.__results:
                    if test.ID == res[0]:
                        f.write(test.report(res[1]))
    
    def __read_bits_in_ascii_format(self, file_path):
        with open(file_path, mode='r') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), "")):
                if n >= self.__sequence.size:
                    return
                if byte == "0" or byte == "1":
                    self.__sequence[n] = byte
                else:
                    raise IllegalBitError(STS.READ_AS.ASCII)

    def __read_bits_in_byte_format(self, file_path):
        with open(file_path, mode='rb') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), b'')):
                if n >= self.__sequence.size:
                    return
                if byte == b'\x00' or byte == b'\x01':
                    self.__sequence[n] = int.from_bytes(byte, 'big')
                else:
                    raise IllegalBitError(STS.READ_AS.BYTE)
    
    def __read_bits_in_binary_format(self, file_path, reverse=False):
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
    
    def run_wrapper(self, args):
        """
        Wrapper function for parallel processing.
        Do not access this method from outside.
        """
        result = [self.__tests[args[0]].ID, args[1]]
        result.extend(self.__tests[args[0]].func(self.__sequence[args[1]]))
        return result
    
    def __sort_results(self, results):
        results.sort(key=lambda x: x[1])
        results.sort()
        self.__results = []
        prev_id = None
        idx = -1
        for i in results:
            if i[0] != prev_id:
                self.__results.append([i[0],[]])
                idx += 1
            self.__results[idx][1].append(i[2:])
            prev_id = i[0]
    
    def __assess_results(self): 
        return


class STSError(Exception):
    """
    Base exception class for STS.
    """
    pass

class InvalidSettingError(STSError):
    """
    Raised when an invalid test parameter is set.
    """
    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return(self.message)

class IllegalBitError(STSError):
    """
    Raised when data different from user setting format is read.
    """
    def __init__(self, fmt):
        if fmt == STS.READ_AS.ASCII:
            annotation = "0x30 or 0x31"
        elif fmt == STS.READ_AS.BYTE:
            annotation = "0x00 or 0x01"
        self.message = (
            "Data in a format different from the setting ({}) was detected."
            .format(annotation))
    
    def __str__(self):
        return self.message

class BitShortageError(STSError):
    """
    Raised when the number of bits in the input file is less than user setting.
    """
    def __init__(self, set_bits, read_bits):
        self.message = (
            "The set value ({} bits) exceeds the bits read ({} bits)."
            .format(set_bits, read_bits)
        )
    
    def __str__(self):
        return self.message

class InvalidProceduralError(STSError):
    """
    Raised when methods are called in a different order than expected.
    """
    def __init__(self, message):
        self.message = message
    
    def __str__(self):
        return(self.message)
