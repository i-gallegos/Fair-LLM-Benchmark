import sys
import argparse
import h5py
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
import json
from utils.holder import *
from utils.extract import get_tokenizer, tokenize_underspecified_input
from transformers import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--topk', help="The the topk to consider", type=int, default=8)
# bert specs
parser.add_argument('--transformer_type', help="The type of transformer encoder",default = "roberta-base")
#
parser.add_argument('--output', help="The path to json format of output", default='')

def load_input(path):
	rs = []
	so = []
	ans_cand = []
	with open(path, 'r') as f:
		for l in f:
			if l.strip() != '':
				head = l.rstrip().split('[Q]')[0]
				sopair = head.split('[CONTEXT]')[0].strip()
				tid = sopair.split('[OPAIR]')[0].split('[T]')[1].strip()
				spair = sopair.split('[OPAIR]')[0].split('[T]')[0].split('[SPAIR]')[1].strip().split('|')
				opair = sopair.split('[OPAIR]')[1].strip().split('|')
				context_choices = head.split('[CONTEXT]')[1].strip()
				context = context_choices.split('[CHOICE]')[0].strip()
				choices = [p.strip() for p in context_choices.split('[CHOICE]')[1].split('|')]
				questions = [p.strip() for p in l.rstrip().split('[Q]')[1:]]
				rs.append((spair, tid, opair, context, choices, questions))
	return rs


def load_mask_filler(opt):
	return pipeline('fill-mask', 
		model=AutoModelWithLMHead.from_pretrained(opt.transformer_type), 
		tokenizer=AutoTokenizer.from_pretrained(opt.transformer_type),
		device=opt.gpuid)

# modified from FillMaskPipeline.__call__ function
def topk_tokens(opt, mask_filler, seq):
	inputs = mask_filler._parse_and_tokenize(seq)
	outputs = mask_filler._forward(inputs, return_tensors=True)

	results = []
	batch_size = outputs.size(0)

	for i in range(batch_size):
		input_ids = inputs["input_ids"][i]
		result = []

		masked_index = (input_ids == mask_filler.tokenizer.mask_token_id).nonzero().item()
		logits = outputs[i, masked_index, :]
		probs = logits.softmax(dim=0)
		values, predictions = probs.topk(opt.topk)

		for idx, p in zip(predictions.tolist(), values.tolist()):
			tok = mask_filler.tokenizer.decode(idx).strip()
			# this is a buggy behavior of bert tokenizer's decoder
			#	Note this also applies to distilbert
			if 'bert-base-uncased' in opt.transformer_type or 'bert-large-uncased' in opt.transformer_type:
				tok = tok.replace(' ', '')
			result.append((tok, p))

		# Append
		results += [result]

	if len(results) == 1:
		return results[0]
	return results


def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# build model
	mask_filler = load_mask_filler(opt)

	ongoing = True
	cnt = 0
	while ongoing:
		if cnt == 0:
			print('running an example')
			seq = 'John went to a market. Mary also went to a market. {0} bought banana.'.format(mask_filler.tokenizer.mask_token)
			print('seq: ', seq)
		else:
			seq = input("seq: ")
		
		topk = topk_tokens(opt, mask_filler, seq)

		for tok, p in topk:
			print('{0}\t{1:.4f}'.format(tok, p))
		cnt += 1


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
