# assess.py
# Copyright (C) 2017 Takuya Kawashima (NIT Tokyo College, Mito Lab.)


import os
import utils


if __name__ == "__main__":

    sts = utils.STS212(f_path=None, f_fmt=None, t_sel=[], seq_size=None, seq_num=None)

    # Test settings
    print("  Enter the path to the file to be tested.\n")
    while True:
        user_inputs = input("    >> ")
        try:
            f = open(user_inputs, 'r')
        except:
            print("\n    File not found. Type it again.\n")
        else:
            sts.settings['File path'] = user_inputs
            f.close()
            break
    
    print("\n  Select the file format from the following.\n")
    print("    0: ASCII  - Each byte represents a character 0 (0x30) or 1 (0x31).")
    print("    1: Byte   - Each byte represents a binary 0 (0x00) or 1 (0x01).")
    print("    2: Bin LE - Each byte represents a binary from 0 (0x00) to 255 (0xFF).")
    print("                Each bit within the byte is read from MSB to LSB.")
    print("    3: Bin BE - Each byte represents a binary from 0 (0x00) to 255 (0xFF).")
    print("                Each bit within the byte is read from LSB to MSB.\n")
    while True:
        user_inputs = input("    >> ")
        if user_inputs in ['0', '1', '2', '3']:
            sts.settings['File format'] = int(user_inputs)
            break
        else:
            print("\n    Invalid input. Type it again.\n")
    
    print("\n  Set the number of sequences to be tested individually.\n")
    while True:
        user_inputs = input("    >> ")
        try:
            user_inputs = int(user_inputs)
        except:
            print("\n    Invalid input. Type it again.\n")
        else:
            if user_inputs < 1:
                print("\n    Too small. Type it again.\n")
            else:
                sts.settings['Sequence num'] = user_inputs
                break
    
    print("\n  Set the bit length of each sequence.\n")
    while True:
        user_inputs = input("    >> ")
        try:
            user_inputs = int(user_inputs)
        except:
            print("\n    Invalid input. Type it again.\n")
        else:
            if user_inputs < 1000:
                print("\n    Too small. Type it again.\n")
            else:
                sts.settings['Sequence size'] = user_inputs
                break
    
    print("\n  Select the test to be run from the following.\n")
    for idx, val in enumerate(sts.TESTS):
        print("    {:<2}: {:<34}".format(idx+1, val['name']), end="")
        if idx % 2 != 0:
            print()
    print("\n\n    To run multiple tests, enter them as \"1,2,3,...\".")
    print("    To run all tests, enter \"0\".\n")
    while True:
        user_inputs = input("    >> ").split(",")
        if set(user_inputs) <= set([str(i) for i in range(len(sts.TESTS)+1)]):
            sts.settings['Test select'] = list(set([int(i) for i in user_inputs]))
            break
        else:
            print("\n    Invalid input. Type it again.\n")
    if 0 in sts.settings['Test select']:
        sts.settings['Test select'] = [i+1 for i in range(len(sts.TESTS))]
    
    for i in [2,7,8,10,11,12]:
        if i in sts.settings['Test select']:
            print("\n  Set the block length in \"{}\" test."
                .format(sts.TESTS[i-1]['name']))
            print("\n    Press Enter to set the default value \"{}\".\n"
                .format(sts.block_sizes[sts.TESTS[i-1]['name']]))
            while True:
                user_inputs = input("    >> ")
                if user_inputs == "":
                    break
                try:
                    user_inputs = int(user_inputs)
                except:
                    print("\n    Invalid input. Type it again.\n")
                else:
                    if user_inputs < 1:
                        print("\n    Too small. Type it again.\n")
                    else:
                        sts.block_sizes[sts.TESTS[i-1]['name']] = user_inputs
                        break

    if (os.cpu_count() > 1
            and (0 in sts.settings['Test select'] or len(sts.settings['Test select'])> 1)):
        print("\n  To run tests in parallel, set the number of processes to {} or less.\n"
            .format(os.cpu_count()))
        print("    Press Enter to run the test in a single process.\n")
        while True:
            user_inputs = input("    >> ")
            if user_inputs == "":
                    break
            try:
                user_inputs = int(user_inputs)
            except:
                print("\n    Invalid input. Type it again.\n")
            else:
                if user_inputs < 1:
                    print("\n    Too small. Type it again.\n")
                elif user_inputs > os.cpu_count():
                    print("\n    Too big. Type it again.\n")
                else:
                    sts.settings['Process num'] = user_inputs
                    break
    
    sts.show_settings()
