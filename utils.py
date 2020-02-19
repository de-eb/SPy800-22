import os
import sys
import csv
import time
import argparse
from multiprocessing import Pool


class STS212:
    """
    """
    # Test Specifiers
    TESTS = [
        {'file' : "_monobit",         'name' : "Frequency (monobit)"},
        {'file' : "_frequency",       'name' : "Frequency within a block"},
        {'file' : "_runs",            'name' : "Runs"},
        {'file' : "_longest_run",     'name' : "Longest run of ones in a block"},
        {'file' : "_rank",            'name' : "Binary matrix rank"},
        {'file' : "_dft",             'name' : "Discrete fourier transform (spectral)"},
        {'file' : "_non_overlapping", 'name' : "Non-overlapping template matching"},
        {'file' : "_overlapping",     'name' : "Overlapping template matching"},
        {'file' : "_universal",       'name' : "Maurer\'s \"universal statistical\""},
        {'file' : "_complexity",      'name' : "Linear complexity"},
        {'file' : "_serial",          'name' : "Serial"},
        {'file' : "_entropy",         'name' : "Approximate entropy"},
        {'file' : "_cusum",           'name' : "Cumulative sums (Cusum)"},
        {'file' : "_excursions",      'name' : "Random excursions"},
        {'file' : "_excursions_var",  'name' : "Random excursions variant"}
    ]
    
    # input modes
    MODES = {'ASCII' : 0, 'BYTE' : 1, 'BIN_LE' : 2, 'BIN_BE' : 3}
    
    def __init__(self, f_path, f_fmt, t_sel, seq_size, seq_num, proc_num=1,
                    bs_freq=128, bs_notm=9, bs_otm=9, bs_lcomp=500, bs_ser=16, bs_apen=10):
        """Constructor
        """
        print()
        print("**************************************************************************")
        print("                      NIST SP800-22 rev.1a by Python3\n")
        print("  A Statistical Test Suite for Random and Pseudorandom Number Generators")
        print("                      for Cryptographic Applications.\n")
        print("   This is an unofficial implementation by volunteers unrelated to NIST.")
        print("**************************************************************************\n")

        self.settings = {   'File path'     : f_path,
                            'File format'   : f_fmt,
                            'Test select'   : t_sel,
                            'Sequence size' : seq_size,
                            'Sequence num'  : seq_num,
                            'Process num'   : proc_num  }

        self.block_sizes = {    'Frequency within a block'          : bs_freq,
                                'Non-overlapping template matching' : bs_notm,
                                'Overlapping template matching'     : bs_otm,
                                'Linear complexity'                 : bs_lcomp,
                                'Serial'                            : bs_ser,
                                'Approximate entropy'               : bs_apen   }

    def show_settings(self):
        """
        """
        print()
        print("**************************************************************************")
        print("  Test settings\n")
        print("    General\n")
        print("      File path            : {}".format(self.settings['File path']))
        print("      File format          : ", end="")
        if self.settings['File format'] == 0:
            print("ASCII characters of \"0\" or \"1\"")
        elif self.settings['File format'] == 1:
            print("Byte values of 0 or 1")
        elif self.settings['File format'] == 2:
            print("Binary (little-endian) from 0 to 255")
        elif self.settings['File format'] == 3:
            print("Binary (big-endian) from 0 to 255")
        print("      Number of sequences  : {}".format(self.settings['Sequence num']))
        print("      Sequence size (bits) : {}".format(self.settings['Sequence size']))
        print("      Number of proccesses : {}".format(self.settings['Process num']))
        print("\n    List of tests to be run\n")
        for i in self.settings['Test select']:
            print("      {:<2}: {:<34}".format(i, self.TESTS[i-1]['name']))
        if not set([2,7,8,10,11,12]).isdisjoint(self.settings['Test select']):
            print("\n    Block sizes\n")
            for i in [2,7,8,10,11,12]:
                if i in self.settings['Test select']:
                    print("      {:<34}: {}"
                        .format(self.TESTS[i-1]['name'], self.block_sizes[self.TESTS[i-1]['name']]))
        print()
        print("**************************************************************************\n")
    