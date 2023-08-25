import sys
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from utils.holder import *
from .data import *
from .pipeline import *
from utils.extract import get_tokenizer, tokenize_underspecified_input, get_special_tokens

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="./data/")
parser.add_argument('--load_file', help="Path to where model to be loaded.", default="")
parser.add_argument('--input', help="Path to input file.", default="")
## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
parser.add_argument('--span_l', help="The maximal span length allowed for prediction", type=int, default=6)
# bert specs
parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
#
parser.add_argument('--output', help="The path to json format of output", default='')
parser.add_argument('--batch_size', help="The batch size used in batch mode, has to be >=2", type=int, default=100)
parser.add_argument('--max_seq_l', help="The maximum concatenated sequence length, larger than this will be dropped", type=int, default=350)
parser.add_argument('--num_q_per_ex', help="The number of questions in each ex, 1 or 2", type=int, default=2)
#
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)


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


def preprocess(opt, tokenizer, source):
	bos_tok, eos_tok = get_special_tokens(tokenizer)
	tokenized = tokenize_underspecified_input(tokenizer, source, opt.verbose==1)
	print(tokenized[:2])
	batch_tok_idx = torch.zeros(len(source)*opt.num_q_per_ex, opt.max_seq_l).long()
	batch_att_mask = torch.zeros(len(source)*opt.num_q_per_ex, opt.max_seq_l)
	batch_ex_idx = []
	batch_seq_l = torch.zeros(len(source)*opt.num_q_per_ex).int()
	batch_context_l = torch.zeros(len(source)*opt.num_q_per_ex).int()
	batch_query_l = torch.zeros(len(source)*opt.num_q_per_ex).int()
	batch_context_start = torch.zeros(len(source)*opt.num_q_per_ex).int()
	batch_concated_toks = []
	batch_concated_choices = []
	for i, (scluster, spair, tid, acluster, opair, context, choices, questions) in enumerate(tokenized):

		#if opt.num_q_per_ex == 1:
			#assert(questions[0] == questions[-1])

		for j in range(opt.num_q_per_ex):
			question = questions[j]
			concated_toks = [bos_tok] + question + [eos_tok] + context + [eos_tok]
			concated_choices = [(span[0]+2+len(question), span[1]+2+len(question)) for span in choices]

			concated = tokenizer.convert_tokens_to_ids(concated_toks)
			concated = torch.from_numpy(np.asarray(concated, dtype=int))
			batch_tok_idx[i*opt.num_q_per_ex+j, :len(concated_toks)] = concated
			batch_att_mask[i*opt.num_q_per_ex+j, :len(concated_toks)] = 1
			batch_ex_idx.append((i, j))
			batch_seq_l[i*opt.num_q_per_ex+j] = len(concated_toks)
			batch_context_l[i*opt.num_q_per_ex+j] = len(context)
			batch_query_l[i*opt.num_q_per_ex+j] = len(question)
			batch_concated_toks.append(concated_toks)
			batch_concated_choices.append(concated_choices)

	return tokenized, batch_concated_toks, batch_concated_choices, batch_ex_idx, batch_tok_idx, batch_att_mask, batch_seq_l, batch_context_l, batch_query_l, batch_context_start


def improve_prediction_roberta(concated_toks, p_start, p_end, idx0, idx1):
	ps = p_start[idx0].data.item()
	pe = p_end[idx1].data.item()

	new_idx0 = idx0
	new_idx1 = idx1
	#print("*********************")
	#print('improving answers on ')
	#print(concated_toks)
	quantifiers = ['Ġan', 'Ġa', 'Ġthe', 'Ġsome', 'Ġfew', 'Ġseveral', 'ĠAn', 'ĠA', 'ĠThe', 'ĠSome', 'ĠFew', 'ĠSeveral']	# all starts with Ġ since question is ahead of context in concated
	prev_1tok = concated_toks[idx0-1]
	if idx0 != 0 and prev_1tok in quantifiers:
		ps = max(ps, p_start[idx0-1].data.item())
		new_idx0 = idx0-1

	prev_3tok = concated_toks[idx0-3:idx0]
	if prev_3tok == ['ĠA', 'Ġgroup', 'Ġof'] or prev_3tok == ['ĠA', 'Ġteam', 'Ġof'] or prev_3tok == ['ĠA', 'Ġcouple', 'Ġof'] or \
		prev_3tok == ['Ġa', 'Ġgroup', 'Ġof'] or prev_3tok == ['Ġa', 'Ġteam', 'Ġof'] or prev_3tok == ['Ġa', 'Ġcouple', 'Ġof']:
		ps = max(ps, p_start[idx0-3:idx0+1].data.max().item())
		new_idx0 = idx0-3

	post_1tok = concated_toks[idx1+1] if idx1 != len(concated_toks)-1 else None
	suffices = ['Ġman', 'Ġwoman', 'Ġboy', 'Ġgirl', 'Ġchild', 'Ġkid', 'Ġperson', 'Ġfolk', 'Ġpeople', 'Ġcouple', 
				'Ġmen', 'Ġwomen', 'Ġboys', 'Ġgirls', 'Ġchildren', 'Ġkids', 'Ġpersons', 'Ġfolks', 
				'Ġcity', 'Ġcountry', 'Ġcities', 'Ġcountries', '.']
	if post_1tok in suffices:
		pe = max(pe, p_end[idx1+1].data.item())
		new_idx1 = idx1+1

	#print('orig answer: {0} improved answer: {1}'.format(' '.join(concated_toks[idx0:idx1+1]), ' '.join(concated_toks[new_idx0:new_idx1+1])))

	return ps, pe

def improve_prediction_bert(concated_toks, p_start, p_end, idx0, idx1):
	ps = p_start[idx0].data.item()
	pe = p_end[idx1].data.item()

	new_idx0 = idx0
	new_idx1 = idx1
	#print("*********************")
	#print('improving answers on ')
	#print(concated_toks)
	quantifiers = ['an', 'a', 'the', 'some', 'few', 'several']	# all starts with Ġ since question is ahead of context in concated
	prev_1tok = concated_toks[idx0-1]
	if idx0 != 0 and prev_1tok in quantifiers:
		ps = max(ps, p_start[idx0-1].data.item())
		new_idx0 = idx0-1

	prev_3tok = concated_toks[idx0-3:idx0]
	if prev_3tok == ['a', 'group', 'of'] or prev_3tok == ['a', 'team', 'of'] or prev_3tok == ['A', 'couple', 'of']:
		ps = max(ps, p_start[idx0-3:idx0+1].data.max().item())
		new_idx0 = idx0-3

	post_1tok = concated_toks[idx1+1] if idx1 != len(concated_toks)-1 else None
	suffices = ['man', 'woman', 'boy', 'girl', 'child', 'kid', 'person', 'folk', 'people', 'couple', 
				'men', 'women', 'boys', 'girls', 'children', 'kids', 'persons', 'folks', 
				'city', 'country', 'cities', 'countries', '.']
	if post_1tok in suffices:
		pe = max(pe, p_end[idx1+1].data.item())
		new_idx1 = idx1+1

	#print('orig answer: {0} improved answer: {1}'.format(' '.join(concated_toks[idx0:idx1+1]), ' '.join(concated_toks[new_idx0:new_idx1+1])))

	return ps, pe

def improve_prediction(transformer_type, concated_toks, p_start, p_end, idx0, idx1):
	if 'roberta' in transformer_type:
		return improve_prediction_roberta(concated_toks, p_start, p_end, idx0, idx1)
	elif 'bert' in transformer_type:
		return improve_prediction_bert(concated_toks, p_start, p_end, idx0, idx1)
	else:
		raise Exception('unrecognized transformer_type', transformer_type)


def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	opt.input = opt.dir + opt.input

	# fix some hyperparameters automatically
	if 'base' in opt.transformer_type:
		opt.hidden_size = 768
	elif 'large' in opt.transformer_type:
		opt.hidden_size = 1024

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# build model
	m = Pipeline(opt, shared)

	# initialization
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	if opt.fp16 == 1:
		m.half()

	if opt.gpuid != -1:
		m = m.cuda()

	# load tokenizer
	print('preprocessing source...')
	tokenizer = get_tokenizer(opt.transformer_type)
	source = load_input(opt.input)
	tokenized, batch_concated_toks, batch_concated_choices, \
		batch_ex_idx, batch_tok_idx, batch_att_mask, batch_seq_l, batch_context_l, batch_query_l, batch_context_start = preprocess(opt, tokenizer, source)

	# sanity check
	assert(opt.batch_size % 2 == 0)

	#
	print('start prediction...')
	m.train(False)
	m.begin_pass()
	shared.is_train = False

	cnt = 0
	batch_cnt = 0
	num_ex = batch_tok_idx.shape[0]
	rs_map = {}
	while cnt < num_ex:
		step_size = opt.batch_size if cnt + opt.batch_size < num_ex else num_ex - cnt

		concated_l = batch_seq_l[cnt:cnt+step_size]
		ex_idx = batch_ex_idx[cnt:cnt+step_size]
		tok_idx = batch_tok_idx[cnt:cnt+step_size, :concated_l.max()]
		att_mask = batch_att_mask[cnt:cnt+step_size, :concated_l.max()]
		context_l = batch_context_l[cnt:cnt+step_size]
		query_l = batch_query_l[cnt:cnt+step_size]
		context_start = batch_context_start[cnt:cnt+step_size]

		tok_idx = to_device(Variable(tok_idx, requires_grad=False), opt.gpuid)
		att_mask = to_device(Variable(att_mask, requires_grad=False), opt.gpuid)
		m.update_context([1]*step_size, step_size, concated_l.max(), context_l, query_l, context_start, {})

		# forward pass
		with torch.no_grad():
			log_start, log_end = m.forward(tok_idx, att_mask)

		pred_span = pick_best_span_bounded(log_start.data.cpu().float(), log_end.data.cpu().float(), opt.span_l)
		p_start, p_end = log_start.exp(), log_end.exp()

		for k in range(pred_span.shape[0]):
			row_id, q_id = ex_idx[k]
			scluster, spair, tid, acluster, opair, context, choices, questions = tokenized[row_id]
			
			keys = '|'.join([scluster[0], scluster[1], spair[0], spair[1], tid, acluster, opair[0], opair[1]])
			if keys not in rs_map:
				rs_map[keys] = {}
				#rs_map[keys]['line'] = row_id
				rs_map[keys]['context'] = source[row_id][5]

			q_row = {}
			q_row['question'] = source[row_id][7][q_id]
			pred_ans = ' '.join(batch_concated_toks[cnt+k][pred_span[k][0]:pred_span[k][1]+1])
			q_row['pred'] = cleanup_G(pred_ans)

			for z, span in enumerate(batch_concated_choices[cnt+k]):
				key = 'ans{0}'.format(z)

				# better coverage on article, quantifiers, and etc.
				s, e = improve_prediction(opt.transformer_type, batch_concated_toks[cnt+k], p_start[k], p_end[k], span[0], span[1])

				q_row[key] = {'text': source[row_id][6][z], 'start': s, 'end': e}

			rs_map[keys]['q{0}'.format(q_id)] = q_row

		cnt += step_size
		batch_cnt += 1

		if batch_cnt % 1000 == 0:
			print("predicted {} examples".format(batch_cnt * opt.batch_size))

	print('predicted {0} examples'.format(cnt))
	m.end_pass()

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
