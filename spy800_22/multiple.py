#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
"""Implementation of SP800-22 test algorithms by Python.

This module is part of the spy800_22 package.
Each class corresponds to each test of NIST SP800-22.
These classes provide various functions (data I/O, parallel processing, etc.)
to execute each test by itself.

Details of NIST SP800-22:
https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

NIST's official implementation:
https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software

"""

from spy800_22.sts import STS, InvalidSettingError
from spy800_22.tests import FrequencyTest
from spy800_22.tests import BlockFrequencyTest
from spy800_22.tests import RunsTest
from spy800_22.tests import LongestRunOfOnesTest
from spy800_22.tests import BinaryMatrixRankTest
from spy800_22.tests import DiscreteFourierTransformTest
from spy800_22.tests import NonOverlappingTemplateMatchingTest
from spy800_22.tests import OverlappingTemplateMatchingTest
from spy800_22.tests import MaurersUniversalStatisticalTest
from spy800_22.tests import LinearComplexityTest
from spy800_22.tests import SerialTest
from spy800_22.tests import ApproximateEntropyTest
from spy800_22.tests import CumulativeSumsTest
from spy800_22.tests import RandomExcursionsTest
from spy800_22.tests import RandomExcursionsVariantTest

class Multiple(STS):
    """STS Multiple Test class"""

    ID = None
    NAME = "Multiple"

    def __init__(self, seq_len: int, seq_num: int, proc_num: int =1,
            choice: list =None, blk_len_blockfrequency: int =128,
            tpl_len_nonoverlapping: int =9, tpl_len_overlapping: int =9,
            blk_len_complexity: int =500, blk_len_serial: int =16,
            blk_len_entropy: int =10, ig_err: bool =False) -> None:
        """
        Parameters
        ----------
        seq_len : `int`
            Bit length of each sequence.

        seq_num : `int`
            Number of sequences.
            If `1` or more, the sequence is split and tested separately.
        
        proc_num : `int`, optional
            Number of processes for running tests in parallel.

        choice : `list of Enum`, optional
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

        blk_len_blockfrequency : `int`, optional
            Block length in "Frequency Test within a Block".
        
        tpl_len_nonoverlapping : `int`, optional
            Template length in "Non-overlapping Template Matching Test".
        
        tpl_len_overlapping : `int`, optional
            Template length in "Overlapping Template Matching Test".
        
        blk_len_complexity : `int`, optional
            Block length in "Linear complexity Test".
        
        blk_len_serial : `int`, optional
            Block length in "Serial Test".
        
        blk_len_entropy : `int`, optional
            Block length in "Approximate entropy Test".
        
        """
        super().__init__(seq_len, seq_num, proc_num, ig_err)
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
            tests.append(BlockFrequencyTest(
                seq_len, seq_num, blk_len=blk_len_blockfrequency, init=False))
        if STS.TestID.RUNS in choice:
            tests.append(RunsTest(seq_len, seq_num, init=False))
        if STS.TestID.LONGESTRUN in choice:
            tests.append(LongestRunOfOnesTest(seq_len, seq_num, init=False))
        if STS.TestID.RANK in choice:
            tests.append(BinaryMatrixRankTest(seq_len, seq_num, init=False))
        if STS.TestID.DFT in choice:
            tests.append(DiscreteFourierTransformTest(
                seq_len, seq_num, init=False))
        if STS.TestID.NONOVERLAPPING in choice:
            tests.append(NonOverlappingTemplateMatchingTest(
                seq_len, seq_num, tpl_len=tpl_len_nonoverlapping, init=False))
        if STS.TestID.OVERLAPPING in choice:
            tests.append(OverlappingTemplateMatchingTest(
                seq_len, seq_num, tpl_len=tpl_len_overlapping, init=False))
        if STS.TestID.UNIVERSAL in choice:
            tests.append(MaurersUniversalStatisticalTest(
                seq_len, seq_num, init=False))
        if STS.TestID.COMPLEXITY in choice:
            tests.append(LinearComplexityTest(
                seq_len, seq_num, blk_len=blk_len_complexity, init=False))
        if STS.TestID.SERIAL in choice:
            tests.append(SerialTest(
                seq_len, seq_num, blk_len=blk_len_serial, init=False))
        if STS.TestID.ENTROPY in choice:
            tests.append(ApproximateEntropyTest(
                seq_len, seq_num, blk_len=blk_len_entropy, init=False))
        if STS.TestID.CUSUM in choice:
            tests.append(CumulativeSumsTest(seq_len, seq_num, init=False))
        if STS.TestID.EXCURSIONS in choice:
            tests.append(RandomExcursionsTest(seq_len, seq_num, init=False))
        if STS.TestID.EXCURSIONSVAR in choice:
            tests.append(RandomExcursionsVariantTest(
                seq_len, seq_num, init=False))
        
        if len(tests) < 1:
            msg = "No tests were selected."\
                  " Test choice must be a list of Enums (Multiple.TestID.xxx)"\
                  " or None."
            raise InvalidSettingError(msg)

        self._STS__tests = tests
