# SPy800-22 :game_die:
Random number testing with [NIST SP800-22](https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf) in Python.

## Description
With this module you can:
- Easy test setup with Python script.
- Testing under the same conditions as [sts-2.1.2](https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software) (NIST source code).
- Faster testing with multi-processing.
- Detailed test report output in CSV format.

## VS. sts-2.1.2
**Comparison of features.**
Features| SPy800-22 | sts-2.1.2
:-:|:-:|:-:
Compile | Unnecessary | Necessary
Interface | Python script | Command line
Bit format | 4 types | 2 types
Output | csv | txt
Speed | Slow | Fast

**<details><summary>Comparison of test results.</summary><div>**

Testing environment.
```
OS  : Windows 10 Home
CPU : Intel Corei5-8250U @ 1.60 GHz, 1800 Mhz, 4 cores, 8 logical processors
RAM : 16.0 GB
Python : 3.8.0 64bit
C compiler : gcc 9.2.0 (MinGW.org GCC Build-20200227-1)
```
Random number generation method.
```python
import numpy as np
bits = np.random.randint(0,2, 1000000000, dtype='uint8')
with open("nprandom.txt", mode='w') as f:
    np.savetxt(f, bits, fmt='%d', delimiter='', newline='')
```
SPy800-22 settings.
```python
from spy800_22.tests import Multiple
test = Multiple(file="nprandom.txt", fmt=Multiple.ReadAs.ASCII, seq_len=1000000, seq_num=1000)
test.run(proc_num=8, ig_err=True)
test.report("results.csv")
```
sts-2.1.2 settings.
```
$> assess.exe 1000000
           G E N E R A T O R    S E L E C T I O N
           ______________________________________

    [0] Input File                 [1] Linear Congruential
    [2] Quadratic Congruential I   [3] Quadratic Congruential II
    [4] Cubic Congruential         [5] XOR
    [6] Modular Exponentiation     [7] Blum-Blum-Shub
    [8] Micali-Schnorr             [9] G Using SHA-1

   Enter Choice: 0


                User Prescribed Input File: nprandom.txt

                S T A T I S T I C A L   T E S T S
                _________________________________

    [01] Frequency                       [02] Block Frequency
    [03] Cumulative Sums                 [04] Runs
    [05] Longest Run of Ones             [06] Rank
    [07] Discrete Fourier Transform      [08] Nonperiodic Template Matchings
    [09] Overlapping Template Matchings  [10] Universal Statistical
    [11] Approximate Entropy             [12] Random Excursions
    [13] Random Excursions Variant       [14] Serial
    [15] Linear Complexity

         INSTRUCTIONS
            Enter 0 if you DO NOT want to apply all of the
            statistical tests to each sequence and 1 if you DO.

   Enter Choice: 1

        P a r a m e t e r   A d j u s t m e n t s
        -----------------------------------------
    [1] Block Frequency Test - block length(M):         128
    [2] NonOverlapping Template Test - block length(m): 9
    [3] Overlapping Template Test - block length(m):    9
    [4] Approximate Entropy Test - block length(m):     10
    [5] Serial Test - block length(m):                  16
    [6] Linear Complexity Test - block length(M):       500

   Select Test (0 to continue): 0

   How many bitstreams? 1000

   Input File Format:
    [0] ASCII - A sequence of ASCII 0's and 1's
    [1] Binary - Each byte in data file contains 8 bits of data

   Select input mode:  0
```


Test results.
Test name| SPy800-22<br>Proportion / Uniformity | sts-2.1.2<br>Proportion / Uniformity
--:|:-:|:-:
Frequency (Monobit) Test | 0.988 / 0.868 | 0.988 / 0.868
Frequency Test within a Block | 0.990 / 0.639 | 0.990 / 0.639
Runs Test | 0.980 / 0.0753 | 0.980 / 0.0753
Test for the Longest Run of Ones in a Block | 0.993 / 0.432 | 0.993 / 0.432
Binary Matrix Rank Test | 0.989 / 0.362 | 0.989 / 0.362
Discrete Fourier Transform (Spectral) Test | 0.986 / 0.284 | 0.986 / 0.284
Non-overlapping Template Matching Test (Lowest Prop. / 148) | 0.981 / 0.579 | 0.981 / 0.579
Non-overlapping Template Matching Test (Lowest Unif. / 148) | 0.989 / 0.000116 | 0.989 / 0.000116
Overlapping Template Matching Test | 0.991 / 0.699 | 0.991 / 0.699
Maurer's "Universal Statistical" Test | 0.991 / 0.979 | 0.991 / 0.979
Linear complexity Test | 0.998 / 0.569 | 0.998 / 0.569
Serial Test (Lowest Prop. & Unif. / 2) | 0.989 / 0.0861 | 0.989 / 0.0861
Approximate entropy Test | 0.988 / 0.608 | 0.988 / 0.608
Cumulative Sums (Cusum) Test (Lowest Prop. / 2) | 0.986 / 0.963 | 0.986 / 0.963
Cumulative Sums (Cusum) Test (Lowest Unif. / 2) | 0.988 / 0.565 | 0.988 / 0.565
Random Excursions Test (Lowest Prop. / 8) | 0.985 / 0.595 | 0.985 / 0.581
Random Excursions Test (Lowest Unif. / 8) | 0.987 / 0.0300 | 0.987 / 0.0277
Random Excursions Variant Test (Lowest Prop. / 18) | 0.984 / 0.483 | 0.984 / 0.470
Random Excursions Variant Test (Lowest Unif. / 18) | 0.995 / 0.100 | 0.995 / 0.0941
**Total test time** | **2 hours 30 minutes** | **1 hour 50 minutes**

</div></details>

## Requirement
- Python 3.7 or higher
- NumPy
- SciPy
- OpenCV

## Usage
```python
from spy800_22.tests import Multiple

test = Multiple(file="binarysequence.txt", fmt=Multiple.ReadAs.ASCII, seq_len=1000000, seq_num=1000)
test.check_file()
test.run(proc_num=4, ig_err=True)
test.report("testresults.csv")
```

## Install
```
$ git clone https://github.com/de-eb/SPy800-22
```

```
$ pip install -r requirements.txt
```

## Licence
This software is released under the [MIT License](https://opensource.org/licenses/MIT), see LICENSE.txt.

## Author

[de-eb](https://github.com/de-eb)