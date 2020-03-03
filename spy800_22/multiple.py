from spy800_22.sts import STS, InvalidSettingError
# from spy800_22.monobit import FrequencyTest
# from spy800_22.frequency import BlockFrequencyTest
from spy800_22.tests import FrequencyTest
from spy800_22.tests import BlockFrequencyTest

class Multiple(STS):
    """
    STS Multiple Test class
    =======================
    Interface for multiple tests.
    
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
    ID = None
    NAME = "Multiple"

    def __init__(self, seq_len, seq_num, proc_num=1, choice=None,
            blk_len_blockfrequency=128, blk_len_nonoverlapping=9,
            blk_len_overlapping=9, blk_len_complexity=500, blk_len_serial=16,
            blk_len_entropy=10) -> None:
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

        choice : tuple of Enum
            IDs of the tests to run. If None, run all tests.
            Specify the built-in Enum. -> Multiple.TestID.xxx
            
            FREQUENCY      : "Frequency (Monobit) Test"
            BLOCKFREQUENCY : "Frequency Test within a Block"
            RUNS           : "Runs Test"
            LONGESTRUN     : "Test for the Longest Run of Ones in a Block"
            RANK           : "Binary Matrix Rank Test"
            DFT            : "Discrete Fourier Transform (Spectral) Test"
            NONOVERLAPPING : "Non-overlapping Template Matching Test"
            OVERLAPPING    : "Overlapping Template Matching Test"
            UNIVERSAL      : "Maurer's Universal Statistical Test"
            COMPLEXITY     : "Linear complexity Test"
            SERIAL         : "Serial Test"
            ENTROPY        : "Approximate entropy Test"
            CUSUM          : "Cumulative sums (cusum) Test"
            EXCURSIONS     : "Random excursions Test"
            EXCURSIONSVAR  : "Random excursions variant Test"

        blk_len_blockfrequency : int
            Block length in "Frequency Test within a Block".
        
        blk_len_nonoverlapping : int
            Block length in "Non-overlapping Template Matching Test".
        
        blk_len_overlapping : int
            Block length in "Overlapping Template Matching Test".
        
        blk_len_complexity : int
            Block length in "Linear complexity Test".
        
        blk_len_serial : int
            Block length in "Serial Test".
        
        blk_len_entropy : int
            Block length in "Approximate entropy Test".
        """
        STS.__init__(self, seq_len, seq_num, proc_num)
        if choice is not None and not isinstance(choice, list):
            msg = "Test choice must be a list of Enums (Multiple.TestID.xxx)"\
                  " or None."
            raise InvalidSettingError(msg)
        elif choice is None:
            choice = [i for i in STS.TestID]

        tests = []
        if STS.TestID.FREQUENCY in choice:
            tests.append(FrequencyTest(seq_len, seq_num, init=False))
        if STS.TestID.BLOCKFREQUENCY in choice:
            tests.append(
                BlockFrequencyTest(
                    seq_len, seq_num, blk_len_blockfrequency, init=False))
        
        if len(tests) < 1:
            msg = "No tests were selected."\
                  " Test choice must be a list of Enums (Multiple.TestID.xxx)"\
                  " or None."
            raise InvalidSettingError(msg)
        self._STS__tests = tests
