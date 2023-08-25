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
from templates.lists import Lists

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="./data/")
parser.add_argument('--input', help="Path to input file.", default="")
## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--topk', help="The the topk to consider", type=int, default=10)
# bert specs
parser.add_argument('--transformer_type', help="The type of transformer encoder",default = "roberta-base")
#
parser.add_argument('--batch_size', help="The batch size used in batch mode, has to be >=2", type=int, default=100)
parser.add_argument('--output', help="The path to json format of output", default='')
parser.add_argument('--use_he_she', help="Whether to account for lm predictions on he/she", type=int, default=0)


def load_gender_names(lists):
	female = []
	for k, ls in lists.subjects.items():
		if k.startswith('female'):
			female.extend([p['[subj]'] for p in ls])
	female = list(set(female))
	female = [p.lower() for p in female]

	male = []
	for k, ls in lists.subjects.items():
		if k.startswith('male'):
			male.extend([p['[subj]'] for p in ls])
	male = list(set(male))
	male = [p.lower() for p in male]
	return female, male



def load_input(path):
	rs = []
	with open(path, 'r') as f:
		json_data = json.load(f)
		for key, ex in json_data.items():
			context = ex['context'].strip()
			choices = [ex['q0']['ans0']['text'].strip(), ex['q0']['ans1']['text'].strip()]
			questions = [ex['q0']['question'].strip(), ex['q1']['question'].strip()]
			subj0_cluster, subj1_cluster, subj0, subj1, tid, a_cluster, obj0, obj1 = key.strip().split('|')
			rs.append(((subj0_cluster, subj1_cluster), (subj0, subj1), tid, a_cluster, (obj0, obj1), context, choices, questions))
	return rs


def preprocess(source):
	rs = []
	for i, (scluster, spair, tid, acluster, opair, context, choices, questions) in enumerate(source):
		for j, q in enumerate(questions):
			rs.append(((i,j), scluster, spair, tid, acluster, opair, context + ' ' + q, choices))
	return rs


def load_mask_filler(opt):
	return pipeline('fill-mask', 
		model=AutoModelForMaskedLM.from_pretrained(opt.transformer_type), 
		tokenizer=AutoTokenizer.from_pretrained(opt.transformer_type),
		device=opt.gpuid)

# modified from FillMaskPipeline.__call__ function
def topk_tokens(opt, mask_filler, batch_seq):
	inputs = mask_filler._parse_and_tokenize(batch_seq, padding=True)
	outputs = mask_filler._forward(inputs, return_tensors=True)

	results = []
	batch_size = outputs.size(0)

	for i in range(batch_size):
		input_ids = inputs["input_ids"][i]
		result = []

		if torch.nonzero(input_ids == mask_filler.tokenizer.mask_token_id).numel() != 1:
			print(batch_seq[i])
			assert(False)

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


def predict(opt, mask_filler, batch_seq, batch_choices):
	batch_topk = topk_tokens(opt, mask_filler, batch_seq)

	rs = []
	for topk, choices in zip(batch_topk, batch_choices):
		topk_choice = [p[0].strip().lower() for p in topk]
		topk_p = [p[1] for p in topk]
		choices = [p.lower() for p in choices]
		leftover_p = 0.0

		p_he = topk_p[topk_choice.index('he')] if 'he' in topk_choice else 0.0	# TODO, should we aggregate the p(pronoun) with p(name) or just take max?
		p_she = topk_p[topk_choice.index('she')] if 'she' in topk_choice else 0.0
	
		rs.append([])
		for c in choices:
			p_c = topk_p[topk_choice.index(c)] if c in topk_choice else leftover_p

			if opt.use_he_she == 1:
				if c in opt.female:
					p_c = max(p_she, p_c)
				elif c in opt.male:
					p_c = max(p_he, p_c)
				else:
					raise Exception('unknown gender of {0}'.format(c))

			rs[-1].append(p_c)
	return rs


def main(args):
	opt = parser.parse_args(args)

	lists = Lists("word_lists", None)

	opt.input = opt.dir + opt.input
	opt.female, opt.male = load_gender_names(lists)

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# build model
	mask_filler = load_mask_filler(opt)

	# load source
	source = load_input(opt.input)
	source = preprocess(source)

	#
	print('start prediction...')
	cnt = 0
	batch_cnt = 0
	num_ex = len(source)
	rs_map = {}
	while cnt < num_ex:
		step_size = opt.batch_size if cnt + opt.batch_size < num_ex else num_ex - cnt
		batch_source = source[cnt:cnt+step_size]
		batch_seq = [row[6] for row in batch_source]
		batch_choices = [row[7] for row in batch_source]
		batch_idx = [row[0] for row in batch_source]
		batch_scluster = [row[1] for row in batch_source]
		batch_spair = [row[2] for row in batch_source]
		batch_tid = [row[3] for row in batch_source]
		batch_acluster = [row[4] for row in batch_source]
		batch_opair = [row[5] for row in batch_source]

		with torch.no_grad():
			batch_output = predict(opt, mask_filler, batch_seq, batch_choices)

		assert(len(batch_output) == step_size)

		for k in range(len(batch_output)):
			row_id, q_id = batch_idx[k]
			
			keys = '|'.join([batch_scluster[k][0], batch_scluster[k][1], batch_spair[k][0], batch_spair[k][1], batch_tid[k], batch_acluster[k], batch_opair[k][0], batch_opair[k][1]])
			if keys not in rs_map:
				rs_map[keys] = {}
				#rs_map[keys]['line'] = row_id
				rs_map[keys]['context'] = 'NA'

			q_row = {}
			q_row['question'] = batch_seq[k]
			q_row['pred'] = 'NA'

			for z, p in enumerate(batch_output[k]):
				key = 'ans{0}'.format(z)
				q_row[key] = {'text': batch_choices[k][z], 'start': p, 'end': p}
	
			rs_map[keys]['q{0}'.format(q_id)] = q_row

		cnt += step_size
		batch_cnt += 1

		if batch_cnt % 1000 == 0:
			print("predicted {} examples".format(batch_cnt * opt.batch_size))

	print('predicted {0} examples'.format(cnt))

	# organize a bit
	ls = []
	for keys, ex in rs_map.items():
		toks = keys.split('|') 
		sort_keys = sorted(toks[0:3])
		sort_keys.extend(toks[3:])
		sort_keys = '|'.join(sort_keys)
		ls.append((sort_keys, keys, ex))
	ls = sorted(ls, key=lambda x: x[0])
	rs_map = {keys:ex for sort_keys, keys, ex in ls}

	json.dump(rs_map, open(opt.output, 'w'), indent=4)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
