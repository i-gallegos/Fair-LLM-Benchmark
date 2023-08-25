import codecs
import argparse
from jiwer import wer

#compute WORD ERROR RATE between two files
def read_file(input):
    with codecs.open(input,'r') as inF:
        input_sents=inF.readlines()
    return input_sents

def write_output(outputfile, output):
     with codecs.open(outputfile, 'w') as outF:
         print(str(output * 100) + "%")
         outF.write(str(output * 100) + "%")

def compute_wer(ref_file, hyp_file, out_file):
    references = read_file(ref_file)
    hypotheses = read_file(hyp_file)
    word_error_rate = wer(references, hypotheses)
    write_output(out_file,word_error_rate)

if __name__ == '__main__':
     # USAGE: python filesWER.py -r referenceFile -i rewriterFile -o outputFile
     parser = argparse.ArgumentParser(description='compute WER reference file vs hypotheses file')
     parser.add_argument("-r", "--ref_file", required=True)
     parser.add_argument("-i", "--hyp_file", required=True)
     parser.add_argument("-o", "--output_file", required=False)
     args = parser.parse_args()
     compute_wer(args.ref_file, args.hyp_file, args.output_file)
