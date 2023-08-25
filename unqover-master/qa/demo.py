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
from utils.extract import get_tokenizer, tokenize_input, get_special_tokens
import math

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="Path to where model to be loaded.", default="/net/nfs2.corp/aristo/underspecified_qa/squad_seqtok")
## pipeline specs
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
parser.add_argument('--span_l', help="The maximal span length allowed for prediction", type=int, default=5)
# bert specs
parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
#
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)
parser.add_argument('--topk', help="The topk prediction spans to output along with top1, if =1 then no effect.", type=int, default=5)


def preprocess(opt, tokenizer, context, question):
	bos_tok, eos_tok = get_special_tokens(tokenizer)
	context_toks, question_toks = tokenize_input(tokenizer, context, question, opt.verbose==1)
	concated_toks = [bos_tok] + question_toks + [eos_tok] + context_toks + [eos_tok]

	batch_tok_idx = torch.zeros(1, len(concated_toks)).long()
	batch_att_mask = torch.zeros(1, len(concated_toks))
	batch_seq_l = torch.zeros(1).int()
	batch_context_l = torch.zeros(1).int()
	batch_query_l = torch.zeros(1).int()
	batch_context_start = torch.zeros(1).int()

	concated = tokenizer.convert_tokens_to_ids(concated_toks)
	concated = torch.from_numpy(np.asarray(concated, dtype=int))
	batch_tok_idx[0, :len(concated_toks)] = concated
	batch_att_mask[0, :len(concated_toks)] = 1
	batch_seq_l[0] = len(concated_toks)
	batch_context_l[0] = len(context_toks)
	batch_query_l[0] = len(question_toks)

	return context_toks, question_toks, concated_toks, batch_tok_idx, batch_att_mask, batch_seq_l, batch_context_l, batch_query_l, batch_context_start


def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# fix some hyperparameters automatically
	if 'base' in opt.transformer_type:
		opt.hidden_size = 768
	elif 'large' in opt.transformer_type:
		opt.hidden_size = 1024

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

	tokenizer = get_tokenizer(opt.transformer_type)

	ongoing = True
	cnt = 0
	while ongoing:
		if cnt == 0:
			print('running an example')
			context = 'John went to a market. Mary also went to a market. One bought banana and the other bought strawberry.'
			question = 'Who bought banana?'
			print('context: ', context)
			print('question: ', question)
		else:
			context = input("context: ")
			question = input("question: ")
		context_toks, question_toks, concated_toks, tok_idx, att_mask, seq_l, context_l, query_l, context_start = preprocess(opt, tokenizer, context, question)
		if opt.verbose == 1:
			print(concated_toks)

		m.train(False)
		m.begin_pass()
		shared.is_train = False
		tok_idx = to_device(Variable(tok_idx, requires_grad=False), opt.gpuid)
		att_mask = to_device(Variable(att_mask, requires_grad=False), opt.gpuid)
		m.update_context([0], 1, len(concated_toks), context_l, query_l, context_start, {})	
		# forward pass
		with torch.no_grad():
			log_start, log_end = m.forward(tok_idx, att_mask)
			log_start, log_end = log_start.data.cpu().float(), log_end.data.cpu().float()
			pred_span = pick_best_span_bounded(log_start, log_end, opt.span_l)[0]
			p_start, p_end = log_start.exp(), log_end.exp()
			pred_ans = ' '.join(concated_toks[pred_span[0]:pred_span[1]+1])
			pred_ans = cleanup_G(pred_ans)
			print('pred: ', pred_ans)
			print('p_start: {0:.4f} p_end: {1:.4f}'.format(float(p_start[0, pred_span[0]].item()), float(p_end[0, pred_span[1]].item())))

			if opt.topk != 1:
				topk_spans = pick_topk_spans(log_start.data.cpu().float(), log_end.data.cpu().float(), opt.span_l, opt.topk)[0]
	
				# shift context if necessary
				context_start_i = context_start[0]
				context_idx1 = topk_spans[:, 0] - query_l[0] - 2	# -2 because of [CLS] and [SEP]
				context_idx2 = topk_spans[:, 1] - query_l[0] - 2
				context_idx1 = context_idx1.clamp(0) + context_start_i
				context_idx2 = context_idx2.clamp(0) + context_start_i
	
				ans_cands = get_answer_tokenized(context_idx1.int(), context_idx2.int(), [context_toks] * topk_spans.shape[0])
				ans_p_start = log_start.expand(topk_spans.shape[0], -1).gather(1, topk_spans[:, 0:1].long()).exp()
				ans_p_end = log_end.expand(topk_spans.shape[0], -1).gather(1, topk_spans[:, 1:2].long()).exp()
				print('topk={0}'.format(opt.topk))
				for k, (a, s, e) in enumerate(zip(ans_cands, ans_p_start.view(-1), ans_p_end.view(-1))):
					print(k, a, 'p_start: {:.4f}'.format(float(s)), 'p_end: {:.4f}'.format(float(e)), 'p_ans: {:.4f}'.format(math.sqrt(s*e)))

		cnt += 1


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
