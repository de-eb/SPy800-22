import os
from enum import Enum
import numpy as np
import multiprocessing as mp


class Base:
    """
    STS base class
    ==============
    Define some methods to be used for all tests.
    
    Attributes
    ----------
    """
    ALPHA = 0.01  # Significance level
    READ_AS = Enum('READ_AS', 'ASCII BYTE BIGENDIAN LITTLEENDIAN')

    def __init__(self, seq_len, seq_num, proc_num=1) -> None:
        """
        Parameters
        ----------
        seq_len : int
            Bit length of each sequence.

        seq_num : int
            Number of sequences.
            If more than 1, the loaded sequence is split and tested separately.
        """
        if seq_len < 1 or seq_num < 1 or proc_num < 1:
            msg = "All parameters must be at least 1."
            raise InvalidSettingError(msg)
        self.input_path = None
        self.output_path = None
        self.sequence_len = int(seq_len)
        self.sequence_num = int(seq_num)
        self.sequence = np.empty(
            self.sequence_len*self.sequence_num, dtype='uint8')
        self.process_num = proc_num
        if self.process_num > mp.cpu_count():
            self.process_num = mp.cpu_count()

    def read_bits(self, file_path: str, fmt)  -> None:
        """
        Read data from a file and convert it to a binary sequence.

        Parameters
        ----------
        file_path : str
            The path of the file to read.
        
        fmt : Enum
            A method of converting data into bits.
            Specify the built-in Enum. -> instance.READ_AS.xxx

            ASCII        : 0x30,0x31 ("0","1") are converted to 0,1.
            BYTE         : 0x00,0x01 are converted to 0,1.
            BIGENDIAN    : 0x00-0xFF are converted to 8 bits in big endian.
            LITTLEENDIAN : 0x00-0xFF are converted to 8 bits in little endian.
        """
        if not os.path.isfile(file_path):
            msg = "File \"{}\" is not found.".format(file_path)
            raise InvalidSettingError(msg)

        total_bits = os.path.getsize(file_path)
        if fmt == Base.READ_AS.BIGENDIAN or fmt == Base.READ_AS.LITTLEENDIAN:
            total_bits *= 8
        if total_bits < self.sequence.size:
            raise BitShortageError(self.sequence.size, total_bits)

        if fmt == Base.READ_AS.ASCII:
            self.__read_bits_in_ascii_format(file_path)
        elif fmt == Base.READ_AS.BYTE:
            self.__read_bits_in_byte_format(file_path)
        elif fmt == Base.READ_AS.BIGENDIAN:
            self.__read_bits_in_binary_format(file_path)
        elif fmt == Base.READ_AS.LITTLEENDIAN:
            self.__read_bits_in_binary_format(file_path, reverse=True)
        else:
            msg = "File input mode must be Enum. -> instance.READ_AS.xxx"
            raise InvalidSettingError(msg)

        self.input_path = file_path
        self.sequence = np.resize(
            self.sequence, (self.sequence_num, self.sequence_len))
    
    def __read_bits_in_ascii_format(self, file_path):
        with open(file_path, mode='r') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), "")):
                if n >= self.sequence.size:
                    return
                if byte == "0" or byte == "1":
                    self.sequence[n] = byte
                else:
                    raise IllegalBitError(Base.READ_AS.ASCII)

    def __read_bits_in_byte_format(self, file_path):
        with open(file_path, mode='rb') as f:
            for n, byte in enumerate(iter(lambda:f.read(1), b'')):
                if n >= self.sequence.size:
                    return
                if byte == b'\x00' or byte == b'\x01':
                    self.sequence[n] = int.from_bytes(byte, 'big')
                else:
                    raise IllegalBitError(Base.READ_AS.BYTE)
    
    def __read_bits_in_binary_format(self, file_path, reverse=False):
        n = 0  # Bit counter
        with open(file_path, mode='rb') as f:
            for byte in iter(lambda:f.read(1), b''):
                bits = int.from_bytes(byte, 'big')
                for i in range(8):
                    if n >= self.sequence.size:
                        return
                    if reverse:  # Little-endian
                        self.sequence[n] = (bits >> i) & 1
                    else:  # Big-endian
                        self.sequence[n] = (bits >> (7-i)) & 1
                    n += 1


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
        if fmt == Base.READ_AS.ASCII:
            annotation = "0x30 or 0x31"
        elif fmt == Base.READ_AS.BYTE:
            annotation = "0x00 or 0x01"
        self.message = (
            "Data in a format different from the setting ({}) was detected."
            .format(annotation)
        )
    
    def __str__(self):
        return self.message

class BitShortageError(STSError):
    """ Exception raised when the number of bits 
        in the input file is less than the user setting.
    """
    def __init__(self, set_bits, read_bits):
        self.message = (
            "The set value ({} bits) exceeds the bits read ({} bits)."
            .format(set_bits, read_bits)
        )
    
    def __str__(self):
        return self.message
