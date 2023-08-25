import codecs
import argparse
from sentence_splitter import SentenceSplitter, split_text_into_sentences

def sentence_split(inputF, outputF):
    input = read_file(inputF)
    splitter = SentenceSplitter(language="en")
    split_sents = splitter.split(text=input[0])
    write_output(outputF,split_sents)

def read_file(input):
    with codecs.open(input,'r') as inF:
        input_sents=inF.readlines()
    return input_sents

def write_output(outputfile,output):
     with codecs.open(outputfile, 'w') as outF:
         outF.write(''.join(output))

if __name__ == '__main__':
     # USAGE: python splitsentences.py -i inputF -o outputF
     parser = argparse.ArgumentParser(description='split textfile into one sentence per line')
     parser.add_argument("-i", "--input_file", required=True)
     parser.add_argument("-o", "--output_file", required=True)
     args = parser.parse_args()
     sentence_split(args.input_file, args.output_file)