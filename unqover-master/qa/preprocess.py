import sys
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import torch
from torch import cuda
from transformers import *
from utils.extract import *


class Indexer:
	def __init__(self, symbols = ["<blank>"]):
		self.PAD = symbols[0]
		self.num_oov = 1
		self.d = {self.PAD: 0}
		self.cnt = {self.PAD: 0}
		for i in range(self.num_oov): #hash oov words to one of 100 random embeddings
			oov_word = '<oov'+ str(i) + '>'
			self.d[oov_word] = len(self.d)
			self.cnt[oov_word] = 0
			
	def convert(self, w):		
		return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(self.num_oov)) + '>']

	def convert_sequence(self, ls):
		return [self.convert(l) for l in ls]

	def write(self, output):
		print(len(self.d), len(self.cnt))
		assert(len(self.d) == len(self.cnt))
		with open(output, 'w+') as f:
			items = [(v, k) for k, v in self.d.items()]
			items.sort()
			for v, k in items:
				f.write('{0} {1} {2}\n'.format(k, v, self.cnt[k]))

	# register tokens only appear in wv
	#   NOTE, only do counting on training set
	def register_words(self, wv, seq, count):
		for w in seq:
			if w in wv and w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

	#   NOTE, only do counting on training set
	def register_all_words(self, seq, count):
		for w in seq:
			if w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]


def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls  


def load_span(opt, path):
	print('loading from {0}...'.format(path))
	all_spans = []
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			toks = l.rstrip().split()
			all_spans.append((int(toks[0]), int(toks[1])))
	return all_spans


def make_vocab(opt, word_indexer, concated_toks, min_seq_l, max_seq_l, count):
	num_ex = 0
	for _, qc_toks in enumerate(concated_toks):

		min_seq_l = min(len(qc_toks), min_seq_l)
		max_seq_l = max(len(qc_toks), max_seq_l)

		num_ex += 1
		word_indexer.register_all_words(qc_toks, count)
	return num_ex, min_seq_l, max_seq_l


def convert(opt, tokenizer, word_indexer, concated_toks, spans, impossibles, output, num_ex, seed=0):
	np.random.seed(seed)

	# record indices to only those appear in word_indexer
	concated = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	concated_lengths = np.zeros((num_ex,), dtype=int)
	query_lengths = np.zeros((num_ex,), dtype=int)
	context_lengths = np.zeros((num_ex,), dtype=int)
	concated_spans = np.zeros((num_ex, 2), dtype=int)	# the answer span in the concatenated list
	context_starts = np.zeros((num_ex,), dtype=int)	# the actual context starting position
	impossible_flags = np.zeros((num_ex,), dtype=int)
	batch_keys = np.array([None for _ in range(num_ex)])
	ex_idx = np.zeros(num_ex, dtype=int)

	# get the CLS and SEP tokens
	CLS, SEP = get_special_tokens(tokenizer)

	sent_id = 0
	for _, (qc_toks, span, impossible) in enumerate(zip(concated_toks, spans, impossibles)):
		assert(len(span) == 2)
		
		mid_boundary = qc_toks.index(SEP)
		start_position = 0
		# if the actual seq is larger than the max seq len, find a chunk that contains the ground truth answer
		if len(qc_toks) > opt.max_seq_l:
			c_toks = qc_toks[mid_boundary+1:-1]
			q_toks = qc_toks[1:mid_boundary]
			ans_start, ans_end = span

			ans_before_chunk = c_toks[ans_start:ans_end+1]

			max_context_l = opt.max_seq_l - 3 - len(q_toks)
			if max_context_l < 0:
				print(q_toks)
				print('example with super long q_toks {0} skipped'.format(len(q_toks)))
				assert(False)

			while ans_end >= start_position + max_context_l:
				start_position += opt.chunk_stride

			if ans_start < start_position:
				print('answer out of chunk, answer len {0}'.format(ans_end - ans_start + 1))

			c_toks = c_toks[start_position: start_position+max_context_l]
			ans_start = ans_start - start_position
			ans_end = ans_end - start_position
			ans_after_chunk = c_toks[ans_start:ans_end+1]

			# reconstruct qc_toks and span
			qc_toks = [CLS] + q_toks + [SEP] + c_toks + [SEP]
			span = [ans_start, ans_end]
			assert(len(qc_toks) <= opt.max_seq_l)

			#logging
			if opt.verbose == 1:
				print('chunked context exceeding max_context_l: {0}'.format(max_context_l))
				print(c_toks)
				print(ans_start, ans_end)
				print(ans_before_chunk)
				print(ans_after_chunk)
			assert(ans_before_chunk == ans_after_chunk)

		c_toks = qc_toks[mid_boundary+1:-1]
		q_toks = qc_toks[1:mid_boundary]

		if impossible == 'false':
			span2 = np.array([span[0]+len(q_toks)+2, span[1]+len(q_toks)+2], dtype=int)
			# sanity check
			assert(c_toks[span[0]:span[1]+1] == qc_toks[span2[0]:span2[1]+1])
		else:
			span2 = np.array([-1, -1], dtype=int)
		
		start_position = np.array([start_position], dtype=int)

		concated[sent_id, :len(qc_toks)] = np.asarray(tokenizer.convert_tokens_to_ids(qc_toks))
		concated_lengths[sent_id] = len(qc_toks)	# including [cls], [sep] and [sep]
		query_lengths[sent_id] = len(q_toks)
		context_lengths[sent_id] = len(c_toks)
		concated_spans[sent_id] = span2
		context_starts[sent_id] = start_position
		impossible_flags[sent_id] = int(bool(impossible))
		batch_keys[sent_id] = len(qc_toks)

		sent_id += 1
		if sent_id % 10000 == 0:
			print("{}/{} examples processed".format(sent_id, num_ex))

	assert(sent_id == num_ex)
	print("{}/{} examples processed".format(sent_id, num_ex))

	# shuffle
	rand_idx = np.random.permutation(num_ex)
	concated_lengths = concated_lengths[rand_idx]
	query_lengths = query_lengths[rand_idx]
	context_lengths = context_lengths[rand_idx]
	concated = concated[rand_idx]
	concated_spans = concated_spans[rand_idx]
	context_starts = context_starts[rand_idx]
	batch_keys = batch_keys[rand_idx]
	impossible_flags = impossible_flags[rand_idx]
	ex_idx = rand_idx
	
	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples
	concated_lengths = concated_lengths[sorted_idx]
	query_lengths = query_lengths[sorted_idx]
	context_lengths = context_lengths[sorted_idx]
	concated = concated[sorted_idx]
	concated_spans = concated_spans[sorted_idx]
	context_starts = context_starts[sorted_idx]
	impossible_flags = impossible_flags[sorted_idx]
	ex_idx = rand_idx[sorted_idx]

	cur_l = []
	batch_location = []
	for j,i in enumerate(sorted_idx):
		if batch_keys[i] != cur_l:
			cur_l = batch_keys[i]
			batch_location.append(j)
	if batch_location[-1] != len(concated): 
		batch_location.append(len(concated)-1)

	# get batch strides
	cur_idx = 0
	batch_idx = [0]
	batch_l = []
	concated_l_new = []
	for i in range(len(batch_location)-1):
		end_location = batch_location[i+1]
		while cur_idx < end_location:
			cur_idx = min(cur_idx + opt.batch_size, end_location)
			batch_idx.append(cur_idx)

	# rearrange examples according to batch strides
	for i in range(len(batch_idx)):
		end = batch_idx[i+1] if i < len(batch_idx)-1 else len(concated)

		batch_l.append(end - batch_idx[i])
		concated_l_new.append(concated_lengths[batch_idx[i]])

		assert(batch_l[-1]<=opt.batch_size)

		# sanity check
		#print('*****', concated_l_new[-1])
		#print(concated_lengths[batch_idx[i]: end])
		for k in range(batch_idx[i], end):
			assert(concated[k, concated_lengths[k]:].sum() == 0)
			assert(concated_lengths[k] == concated_l_new[-1])


	# Write output
	f = h5py.File(output, "w")		
	f["concated"] = concated
	f["concated_l"] = concated_l_new	# (batch_l,)
	f["context_l"] = context_lengths
	f["query_l"] = query_lengths
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f["concated_span"] = concated_spans
	f['context_start'] = context_starts
	f['impossible'] = impossible_flags
	f['ex_idx'] = ex_idx
	print("Saved {} examples.".format(len(f["concated"])))
	print('Number of batches: {0}'.format(len(batch_idx)))
	f.close()				
	

def process(opt):
	#tokenizer = get_tokenizer(opt.transformer_type)
	tokenizer = AutoTokenizer.from_pretrained(opt.transformer_type, add_special_tokens=False, use_fast=True)
	CLS, SEP = get_special_tokens(tokenizer)

	# first extract all
	context, query, span, qid, impossible = tokenize_squad_and_write(tokenizer, opt.train_data, opt.output+'.train', opt.verbose==1)
	context_val, query_val, span_val, qid_val, impossible_val = tokenize_squad_and_write(tokenizer, opt.dev_data, opt.output+'.dev', opt.verbose==1)

	word_indexer = Indexer(symbols = ["<blank>", CLS, SEP])

	# concat to form [CLS] query [SEP] context [SEP]
	concated = [[CLS] + q + [SEP] + c + [SEP] for c, q in zip(context, query)]
	concated_val = [[CLS] + q + [SEP] + c + [SEP] for c, q in zip(context_val, query_val)]

	min_seq_l = 100000
	max_seq_l = 0
	print("First pass through data to get vocab...")
	num_ex_train, min_seq_l, max_seq_l = make_vocab(opt, word_indexer, concated, min_seq_l, max_seq_l, count=True)
	print("Number of examples in training: {0}, number of tokens: {1}".format(num_ex_train, len(word_indexer.d)))
	num_ex_valid, min_seq_l, max_seq_l = make_vocab(opt, word_indexer, concated_val, min_seq_l, max_seq_l, count=False)
	print("Number of examples in valid: {0}, number of tokens: {1}".format(num_ex_valid, len(word_indexer.d)))   
	
	print('Number of tokens collected: {0}'.format(len(word_indexer.d)))
	word_indexer.write(opt.output + ".word.dict")

	print("Min seq length: {}".format(min_seq_l))
	print("Max seq length: {}".format(max_seq_l))	

	min_seq_l = 1000000
	max_seq_l = 0
	convert(opt, tokenizer, word_indexer, concated_val, span_val, impossible_val, opt.output + ".val.hdf5", num_ex_valid, opt.seed)
	convert(opt, tokenizer, word_indexer, concated, span, impossible, opt.output + ".train.hdf5", num_ex_train, opt.seed)
	
def main(arguments):
	parser = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
	parser.add_argument('--dir', help="Path to the data dir", default = "data/squad/")
	parser.add_argument('--train_data', help="Path to SQUAD json train set file", default="train-v1.1.json")
	parser.add_argument('--dev_data', help="Path to SQUAD json dev set file", default="dev-v1.1.json")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=16)
	parser.add_argument('--max_seq_l', help="Maximum sequence length, should leq the pretrained BERT max len.", type=int, default=384)
	parser.add_argument('--chunk_stride', help=".", type=int, default=128)
	parser.add_argument('--output', help="Prefix of the output file names.", type=str, default = "squad")
	parser.add_argument('--seed', help="seed of shuffling sentences.", type = int, default = 1)
	parser.add_argument('--verbose', type=int, default = 1)
	opt = parser.parse_args(arguments)

	#
	opt.train_data = opt.dir + opt.train_data
	opt.dev_data = opt.dir + opt.dev_data
	opt.output = opt.dir + opt.output

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
