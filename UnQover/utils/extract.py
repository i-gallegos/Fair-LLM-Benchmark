import sys
import os
import ujson
import argparse
import re
import torch
from torch import cuda
from transformers import *

def get_tokenizer(key):
	print('loading tokenizer: {0}'.format(key))
	return AutoTokenizer.from_pretrained(key, add_special_tokens=False, use_fast=True)

# there is some un-unified interfaces in current huggingface transformer
#	so need to do customized vocab fetching
def get_vocab(tokenizer, key):
	if key == 'bert-base-uncased':
		return tokenizer.vocab
	elif key == 'roberta-base' or key == 'gpt2':
		return tokenizer.encoder
	elif key == 'transfo-xl-wt103':
		return tokenizer.sym2idx
	else:
		raise Exception('unsupported bert type {0}'.format(key))

def get_special_tokens(tokenizer):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	if CLS is None or SEP is None:
		CLS, SEP = tokenizer.bos_token, tokenizer.eos_token
	if CLS is None:
		CLS = SEP
	return CLS, SEP

def get_gold(answer_spans):
	cnt = {}
	for span in answer_spans:
		if span in cnt:
			cnt[span] = cnt[span] + 1
		else:
			cnt[span] = 1
	sorted_keys = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
	maj_span = sorted_keys[0][0]
	return (maj_span, answer_spans.index(maj_span))


def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))


def tokenize_and_map(tokenizer, seq):
	rs = tokenizer.encode_plus(seq, add_special_tokens=False, return_offsets_mapping=True)
	# None in offset_mapping represents bos/eos.
	#	We will deal with that later, but not here.
	if rs['offset_mapping'][0] is None:
		rs['offset_mapping'] = rs['offset_mapping'][1:]
		rs['input_ids'] = rs['input_ids'][1:]
		# sometimes there are multiple None's at the tail
		while rs['offset_mapping'][-1] is None:
			rs['offset_mapping'] = rs['offset_mapping'][:-1]
			rs['input_ids'] = rs['input_ids'][:-1]

	assert(None not in rs['offset_mapping'])
	assert(len(rs['input_ids']) == len(rs['offset_mapping']))

	# sanity check on offset mapping
	for span in rs['offset_mapping']:
		assert(' ' not in seq[span[0]:span[1]].strip())

	token_idx = rs['input_ids']
	token_offsets = rs['offset_mapping']

	# improve token offsets to remove spaces
	improved_offsets = []
	for span in token_offsets:
		if seq[span[0]] == ' ' and span[1] != span[0]+1:
			improved_offsets.append((span[0]+1, span[1]))
		else:
			improved_offsets.append(span)

	assert(len(token_offsets) == len(token_idx))

	# smooth spans
	#	HF changed the interface to skip spaces....
	#	but we need the space count as part of the token follows it
	new_offsets = []
	for i, span in enumerate(token_offsets):
		if i == 0:
			new_offsets.append((0, span[1]))	# force to start form 0
		elif span[0] > token_offsets[i-1][1]:
			new_offsets.append((token_offsets[i-1][1], span[1]))
		else:
			new_offsets.append((span))
	token_offsets = new_offsets

	# get char to token map
	char_to_orig_tok = []
	for i, c in enumerate(seq):
		for k, span in enumerate(token_offsets):
			if i >= span[0] and i < span[1]:
				char_to_orig_tok.append(k)
				break

	if len(char_to_orig_tok) != len(seq):
		print(seq)
		print(token_offsets)
		print(len(seq), len(char_to_orig_tok))
		k = 0
		start = 0
		while True:
			if k+1 not in char_to_orig_tok:
				print(seq[start:], (start, len(char_to_orig_tok)))
				break
			end = char_to_orig_tok.index(k+1)
			print(seq[start:end], (start, end))
			k = k+1
			start = end
	assert(len(char_to_orig_tok) == len(seq))

	return token_idx, token_offsets, improved_offsets, char_to_orig_tok

def get_proper_spans(context, choices):
	context = context.lower()
	rs = [None for _ in choices]
	for i, c in sorted([(i, c) for i, c in enumerate(choices)], key=lambda x:len(x[1]), reverse=True):
		c = c.lower()
		if c not in context:
			raise Exception(c + ' not in ' + context)

		start = context.index(c)
		end = start + len(c)

		# if there is conflict, then search in reverse order again
		if len([1 for span in rs if span is not None and start >= span[0] and end <= span[1]]) != 0:
			start = context.rindex(c)
			end = start + len(c)

		# if still has conflict, then screw it
		if len([1 for span in rs if span is not None and start >= span[0] and end <= span[1]]) != 0:
			print(context)
			print(choices)
			raise Exception('cant find proper span')

		rs[i] = (start, end)

	return rs


# customized for for predict.py
def tokenize_underspecified_input(tokenizer, source, verbose=False):
	rs = []
	cnt = 0
	for scluster, spair, tid, acluster, opair, context, choices, questions in source:
		tok_idx, token_offsets, improved_offsets, char_to_orig_tok = tokenize_and_map(tokenizer, context)
		context_toks = tokenizer.convert_ids_to_tokens(tok_idx)

		choice_char_spans = get_proper_spans(context, choices)
		choice_spans = []
		for choice, char_span in zip(choices, choice_char_spans):
			choice_span_idx1, choice_span_idx2 = char_span
			choice_span = (choice_span_idx1, choice_span_idx2)

			assert(context.lower()[char_span[0]:char_span[1]] == choice.lower())

			orig_tok_ans_idx = (char_to_orig_tok[choice_span[0]], char_to_orig_tok[choice_span[1]])
			orig_tok_ans_idx = (orig_tok_ans_idx[0], orig_tok_ans_idx[1]-1)	# make the ending index inclusive
			tokenized_answer = context_toks[orig_tok_ans_idx[0]:orig_tok_ans_idx[1]+1]

			if verbose:
				print(choice, tokenized_answer)
			choice_spans.append(orig_tok_ans_idx)
		question_toks = [tokenizer.tokenize(q) for q in questions]
		cnt += 1

		if cnt % 100000 == 0:
			print('tokenized {0} lines'.format(cnt))

		rs.append((scluster, spair, tid, acluster, opair, context_toks, choice_spans, question_toks))

	print('tokenized {0} lines'.format(cnt))
	return rs


def tokenize_input(tokenizer, context, question, verbose=False):
	tok_idx, token_offsets, improved_offsets, char_to_orig_tok = tokenize_and_map(tokenizer, context)
	context_toks = tokenizer.convert_ids_to_tokens(tok_idx)

	question_toks = tokenizer.tokenize(question)

	return context_toks, question_toks


def tokenize_squad(tokenizer, json_file, verbose=False):
	all_raw_context = []
	all_context_tokenized = []
	all_context_orig_tok = []
	all_query = []
	all_raw_query = []
	all_query_id = []
	all_span = []
	all_raw_ans = []
	all_tok_ans = []
	all_tok_to_orig_tok = []
	all_impossible = []
	context_max_sent_num = 0
	max_sent_l = 0
	ex_cnt = 0

	with open(json_file, 'r') as f:
		f_str = f.read()
	j_obj = ujson.loads(f_str)

	data = j_obj['data']

	for article in data:
		title = article['title']
		pars = article['paragraphs']
		for p in pars:
			context = p['context'].rstrip()
			qas = p['qas']

			# new tokenization
			tok_idx, token_offsets, improved_offsets, char_to_orig_tok = tokenize_and_map(tokenizer, context)
			context_toks = tokenizer.convert_ids_to_tokens(tok_idx)

			for qa in qas:
				query = qa['question']
				query_id = qa['id']
				if 'is_impossible' in qa:
					is_impossible = qa['is_impossible']
				else:
					is_impossible = False
				ans = qa['answers']

				query = query.strip()
				query_toks = tokenizer.tokenize(query)
				max_sent_l = max(max_sent_l, len(query_toks))

				answer_orig_spans = []
				for a in ans:
					a_txt = a['text']
					if a_txt.strip() == '':
						continue
					#idx1 = a['answer_start']	# answer_start is not to be trusted in MultiQA
					if a_txt in context:
						idx1 = context.index(a_txt)
						idx2 = idx1 + len(a_txt) - 1	# end idx is inclusive

						answer_orig_spans.append((idx1, idx2))

				if len(answer_orig_spans) == 0 and is_impossible is False:
					print('skipping {0} since no answer is given'.format(query_id))
					continue

				if not is_impossible:
					orig_maj_span = get_gold(answer_orig_spans)[0]
					orig_answer = context[orig_maj_span[0]:orig_maj_span[1]+1]
					assert(list(orig_maj_span) != [0, -1])
	
					tokenized_ans_idx = (char_to_orig_tok[orig_maj_span[0]], char_to_orig_tok[orig_maj_span[1]])
					tokenized_answer = context_toks[tokenized_ans_idx[0]:tokenized_ans_idx[1]+1]
	
					if verbose:
						print(orig_answer, tokenized_answer)
	
					all_orig_answers = [context[orig_span[0]:orig_span[1]+1] for orig_span in answer_orig_spans]
					all_orig_answers = [p.replace('\n', ' ') for p in all_orig_answers]

					all_tokenized_answers = []
					for orig_span in answer_orig_spans:
						idx = (char_to_orig_tok[orig_span[0]], char_to_orig_tok[orig_span[1]])
						all_tokenized_answers.append(' '.join(context_toks[idx[0]:idx[1]+1]).replace('\n', ' '))
						
				else:
					tokenized_ans_idx = [0,0]
					all_orig_answers = ['IMPOSSIBLE']


				# add to final list
				all_raw_query.append(query.replace('\n', ' ').rstrip())
				all_raw_context.append(context.replace('\n', ' ').rstrip())
				all_context_tokenized.append(' '.join(context_toks))
				all_query.append(' '.join(query_toks))
				all_query_id.append(query_id)
				all_span.append((tokenized_ans_idx[0], tokenized_ans_idx[1]))
				all_raw_ans.append('|||'.join(all_orig_answers))
				all_tok_ans.append('|||'.join(all_tokenized_answers))
				all_impossible.append(str(is_impossible).lower())

				ex_cnt += 1
				if ex_cnt % 10000 == 0:
					print('extracted {0} examples'.format(ex_cnt))


		if verbose:
			print('max seq len: {0}'.format(max_sent_l))

	return (all_raw_context, all_context_tokenized, all_raw_query, all_query, all_span, all_raw_ans, all_tok_ans, all_query_id, all_impossible)


# the interface that can be called from external
def tokenize_squad_and_write(tokenizer, json_path, output, verbose):
	print('extracting examples from {0}...'.format(json_path))
	raw_context, context, raw_query, query, span, raw_ans, tok_ans, query_id, impossible = tokenize_squad(tokenizer, json_path, verbose)
	print('{0} examples extracted.'.format(len(context)))

	write_to(raw_context, output + '.raw_context.txt')
	write_to(context, output + '.context.txt')
	write_to(raw_query, output + '.raw_query.txt')
	write_to(query, output + '.query.txt')
	write_to(raw_ans, output + '.raw_answer.txt')
	write_to(tok_ans, output + '.tok_answer.txt')
	write_to(query_id, output + '.query_id.txt')
	write_to(impossible, output + '.impossible.txt')
	write_to(['{0} {1}'.format(p[0], p[1]) for p in span], output + '.span.txt')

	context = [p.split() for p in context]
	query = [p.split() for p in query]
	return context, query, span, query_id, impossible



def main(args):
	parser = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
	parser.add_argument('--dir', help="Path to the data dir", default="data/squad/")
	parser.add_argument('--data', help="Path to SQUAD json file", default="dev-v1.1.json")
	parser.add_argument('--output', help="Prefix to the path of output", default="dev")
	parser.add_argument('--verbose', type=int, default = 1)

	opt = parser.parse_args(args)

	# append path
	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output

	tokenizer = get_tokenizer(opt.transformer_type)

	context, query, span, query_id, impossible = tokenize_squad_and_write(tokenizer, opt.data, opt.output, verbose=opt.verbose==1)
	print('{0} examples processed.'.format(len(context)))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


