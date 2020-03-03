# frequency.py

from tests.base import Base, InvalidSettingError
import numpy as np
from scipy.special import gammaincc


class BlockFrequencyTest(Base):
    """
    Frequency Test within a Block
    =============================

    Attributes
    ----------
    """
    ID = Base.TestID.BLOCKFREQUENCY
    NAME = "Frequency Test within a Block"

    def __init__(self, seq_len, seq_num, blk_len=128, proc_num=1,
                init=True) -> None:
        """
        Parameters
        ----------
        seq_len : int
            Bit length of each sequence.

        seq_num : int
            Number of sequences.
            If more than 1, the loaded sequence is split and tested separately.

        blk_size : int
            Bit length of each block.
        """
        if seq_len < blk_len:
            msg = "The sequence length must be larger than the block length."
            raise InvalidSettingError(msg)
        if init:
            super().__init__(seq_len, seq_num, proc_num)
            # self.__tests.append(self)
        self.__blk_len = blk_len
        self.__blk_num = seq_len // blk_len
    
    def info(self) -> str:
        """
        Test information.

        Returns
        -------
        msg : str
            A string containing test parameters and results annotations.
        """
        msg = BlockFrequencyTest.ID.value + "\n"
        msg += "Block length,{}\n".format(self.__blk_len)
        msg += "Number of blocks,{}\n".format(self.__blk_num)
        msg += "Partial results\n"
        msg += "SequenceID,p-value,chi_square\n"
        return msg
    
    def func(self, bits):
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
