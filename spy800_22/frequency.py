# frequency.py

from spy800_22.sts import STS, InvalidSettingError
import numpy as np
from scipy.special import gammaincc


class BlockFrequencyTest(STS):
    """
    Frequency Test within a Block
    =============================
    Evaluate the proportion of 1s for each M-bit subsequence.

    Attributes
    ----------
    blk_len : int
        Bit length of each block.
    
    blk_num : int
        Number of blocks.
    
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

    ID = STS.TestID.BLOCKFREQUENCY
    NAME = "Frequency Test within a Block"

    def __init__(self, seq_len: int, seq_num: int, blk_len: int =128,
            proc_num: int =1, init: bool =True) -> None:
        """
        Parameters
        ----------
        seq_len : int
            Bit length of each sequence.

        seq_num : int
            Number of sequences.
            If more than 1, the loaded sequence is split and tested separately.

        blk_len : int
            Bit length of each block.
        
        proc_num : int
            Number of processes for running tests in parallel.
        
        init : bool
            Whether to initialize super class.
            To avoid redefinition of super class in multiple tests.
        
        """
        if seq_len < blk_len:
            msg = "The sequence length must be larger than the block length."
            raise InvalidSettingError(msg)
        if init:
            super().__init__(seq_len, seq_num, proc_num)
        self.__blk_len = blk_len
        self.__blk_num = seq_len // blk_len
    
    @property
    def blk_len(self):
        return self.__blk_len
    
    @property
    def blk_num(self):
        return self.__blk_num
    
    def func(self, bits: np.ndarray) -> tuple:
        """
        Main function.

        Parameters
        ----------
        bits : 1d-ndarray uint8
            Binary sequence to be tested.
        
        Returns
        -------
        p_value : float
            Test result.
        chi_square : float
            Computed statistic.
        
        """
        blk = np.resize(bits, (self.__blk_num, self.__blk_len))
        sigma = np.sum((np.sum(blk, axis=1)/self.__blk_len - 0.5)**2)
        chi_square = 4 * self.__blk_len * sigma
        p_value = gammaincc(self.__blk_num/2 , chi_square/2)
        return p_value, chi_square
    
    def report(self, results: list) -> str:
        """
        Generate a CSV string from the partial test results.
        Use save_report() to generate a CSV for the entire test.

        """
        msg = BlockFrequencyTest.NAME + "\n"
        msg += "\nBlock length,{}\n".format(self.__blk_len)
        msg += "Number of blocks,{}\n".format(self.__blk_num)
        msg += "\nSequenceID,p-value,chi_square\n"
        for i, j in enumerate(results):
            msg += "{},{},{}\n".format(i, j[0], j[1])
        return msg
