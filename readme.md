# SPy800-22 :game_die:
Random number testing with [SP800-22](https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-22r1a.pdf) in Python.

## Description
With this module you can:
- Easy test setup with Python script.
- Testing under the same conditions as the [NIST source code](https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software).
- Faster testing with multi-processing.
- Detailed test report output in CSV format.

## Requirement
- Python 3.7 or higher
- NumPy
- SciPy
- OpenCV

## Usage
```python
from spy800_22.tests import Multiple

test = Multiple(seq_len=1000000, seq_num=1000)
test.load_bits("binarysequence.txt", fmt=test.ReadAs.ASCII)
test.run()
test.assess()
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