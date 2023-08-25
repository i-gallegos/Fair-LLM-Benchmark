#script to remove empty lines in parallel data
#noticed issues with pre-processing script => sometimes empty line on one side, but not on the other
#make sure to remove empty lines when they occur (and parallel sentence) on other side.
# IN THE END: not necessary because automatically removed by scripts
import re
import argparse
import codecs

def remove_parallel(inputfile1, inputfile2):
    file1 = read_file(inputfile1)
    file2 = read_file(inputfile2)
    remove_empty(file1,file2)

def read_file(input_file):
    with codecs.open(input_file,'r') as inF:
        input_sents=inF.readlines()
    return input_sents

def remove_empty(file1, file2):
    counter = 0
    for line1,line2 in zip(file1,file2):
        if re.match(r'^\s*$', line1) or re.match(r'^\s*$', line2):
            file1.pop(counter)
            file2.pop(counter)
        counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse sentences using stanzaNLP')
    parser.add_argument("-i", "--input_file", required=True)                            #inputfile format: one sentence per line
    parser.add_argument("-o", "--output_file", required=True)                           #name of outputfile: one sentence per line
    args = parser.parse_args()
    remove_parallel(args.input_file, args.output_file)