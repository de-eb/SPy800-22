# monobit.py

from spy800_22.sts import STS
import numpy as np
from math import sqrt, erfc


class FrequencyTest(STS):
    """
    Frequency (Monobit) Test
    ========================
    Evaluate the uniformity of 0s and 1s for the entire sequence.

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
    ID = STS.TestID.FREQUENCY
    NAME = "Frequency (Monobit) Test"

    def __init__(self, seq_len: int, seq_num: int, proc_num: int =1,
            init: bool =True) -> None:
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
        
        init : bool
            Whether to initialize super class.
            To avoid redefinition of super class in multiple tests.
        """
        if init:
            super().__init__(seq_len, seq_num, proc_num)

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
        
        zeros : int
            Number of occurrences of 0s for the entire sequence.
        
        ones : int
            Number of occurrences of 1s for the entire sequence.
        """
        ones = np.count_nonzero(bits)
        zeros = bits.size - ones
        p_value = erfc((abs(ones-zeros)/sqrt(bits.size))/sqrt(2))
        return p_value, zeros, ones

    def report(self, results: list) -> str:
        """
        Generate a CSV string from the partial test results.
        Use save_report() to generate a CSV for the entire test.
        """
        msg = FrequencyTest.NAME + "\n"
        msg += "Partial results\n"
        msg += "SequenceID,p-value,zeros,ones\n"
        for i, j in enumerate(results):
            msg += "{},{},{},{}\n".format(i, j[0], j[1], j[2])
        return msg
