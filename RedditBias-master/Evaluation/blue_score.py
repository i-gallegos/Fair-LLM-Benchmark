"""
This script calculates corpus BLEU score for generated responses w.r.t DSTC7 reference set
"""
import nltk
from nltk.translate.bleu_score import SmoothingFunction
import argparse
import json


# parser = argparse.ArgumentParser()
# parser.add_argument('--hyp_file', help='Path containing hypothesis file')
# parser.add_argument('--ref_file', help='Path containing reference file')
#
# args = parser.parse_args()
#
# hyp_file_path = args.hyp_file
# ref_file_path = args.ref_file

hyp_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/response_hyp.txt'
ref_file_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/dstc7/test.refs.txt'

# with open(hyp_file_path, encoding="utf-8") as f:
#     hyp_lines = [line for line in f.read()]

with open(hyp_file_path) as f:
    hyp_lines = json.load(f)

print(hyp_lines[0])

with open(ref_file_path, encoding="utf-8") as f:
    ref_lines = [line for line in f.read()]

print(ref_lines[0])

hyp_sentences = []
ref_sentences = []
dict_ref = {}

for ref in ref_lines:
    print('ref is {}'.format(ref))
    key = ref.split('\t')[0]
    print('key {}'.format(key))
    dict_ref[key] = ref.split('\t')[-1]
    # response = ref['response'].replace(ref['input'], '')
    # key = ref['key']
    # dict_ref[key] = response

for line in hyp_lines[:100]:
    # key = line.split(': ')[0]
    # key = key.strip('{"')
    # hyp_sent = line.split(': ')[1]
    hyp_sent = line['response'].replace(line['input'], '')
    key = line['key']
    ref_sent = dict_ref[key]
    print('hyp_sent {}'.format(hyp_sent))
    print('ref_sent {}'.format(ref_sent))
    ref_sentences.append(ref_sent)
    hyp_sentences.append(hyp_sent)

print('number of hyp {}'.format(len(hyp_sentences)))
print('number of ref {}'.format(len(ref_sentences)))

smooth = SmoothingFunction().method4
bleu_cor = nltk.translate.bleu_score.corpus_bleu(hyp_sentences, ref_sentences, smoothing_function=smooth)
print('bleu for corpus: {}'.format(bleu_cor))
