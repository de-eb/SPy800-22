#!/usr/bin/env python3
# -*- Coding: utf-8 -*-
"""Implementation of SP800-22 test algorithms by Python.

This module is part of the spy800_22 package and consists of 15 classes.
Each class corresponds to each test of NIST SP800-22.
These classes provide various functions (data I/O, parallel processing, etc.)
to execute each test by itself.

Details of NIST SP800-22:
https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf

NIST's official implementation:
https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software

Notes
-----
    Test results may differ from those of the NIST's official implementation.
    This is due to all array operations being optimized for Numpy.

"""

