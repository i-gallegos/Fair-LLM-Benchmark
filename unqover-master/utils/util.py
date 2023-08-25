import sys
import h5py
import torch
from torch import nn
from torch import cuda
import string
import re
from collections import Counter
import numpy as np
import json
import math


def to_device(x, gpuid):
	if gpuid == -1:
		return x.cpu()
	if x.device != gpuid:
		return x.cuda(gpuid)
	return x

def has_nan(t):
	return torch.isnan(t).sum() == 1

def tensor_on_dev(t, is_cuda):
	if is_cuda:
		return t.cuda()
	else:
		return t

def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()


def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))


def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f


def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.items():
		file.create_dataset(name, data=p)

	file.close()


def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			w, idx, cnt = l.strip().split()
			rs[int(idx)] = w
	return rs


def rand_tensor(shape, r1, r2):
	return (r1 - r2) * torch.rand(shape) + r2


def build_rnn(type, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional):
	if type == 'lstm':
		return nn.LSTM(input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bias=bias,
			batch_first=batch_first,
			dropout=dropout,
			bidirectional=bidirectional)
	elif type == 'gru':
		return nn.GRU(input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bias=bias,
			batch_first=batch_first,
			dropout=dropout,
			bidirectional=bidirectional)
	else:
		assert(False)


###### official evaluation
# TODO, for unicode, there are versions of punctuations (esp. brackets)
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


###### official evaluation
def f1_bow(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

###### offcial evaluation
def em_bow(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

###### official evaluation
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Given a prediction and multiple valid answers, return the score of the best
    prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# tok_idx is scalar integer
# sent_ls is a list of integer for sentence lengths
def get_sent_idx(tok_idx, sent_ls):
	sent_idx = -1
	acc_l = 0
	for i, l in enumerate(sent_ls):
		acc_l += l
		if tok_idx < acc_l:
			sent_idx = i
			break
	assert(sent_idx != -1)
	return sent_idx

# the gold is a single span pf gold token_idx
#	could be the start or the end
def get_em_sent(pred_tok_idx, gold_tok_idx, context_sent_l):
	pred_sent_idx = torch.Tensor([get_sent_idx(int(idx), sent_l) for idx, sent_l in zip(pred_tok_idx, context_sent_l)])
	gold_sent_idx = torch.Tensor([get_sent_idx(int(idx), sent_l) for idx, sent_l in zip(gold_tok_idx, context_sent_l)])
	return (pred_sent_idx == gold_sent_idx).float()


def get_sent(tok_idx, context_sent_l, batch_token_span, batch_raw):
	sent_idx = [get_sent_idx(int(idx), sent_l) for idx, sent_l in zip(tok_idx, context_sent_l)]
	raw_sent = []
	for i, idx in enumerate(sent_idx):
		start = sum(context_sent_l[i][:idx])
		end = start + context_sent_l[i][idx]-1
		start = batch_token_span[i][start][0]
		end = batch_token_span[i][end][1]
		assert(start != -1)
		assert(end != -1)
		raw_sent.append(batch_raw[i][start:end+1])
	return raw_sent
	

# pick the best span given a maximal length
def pick_best_span_bounded(log_p1, log_p2, bound):
	log_p1, log_p2 = log_p1.cpu(), log_p2.cpu()
	assert(len(log_p1.shape) == 2)	# (batch_l, context_l)
	assert(len(log_p2.shape) == 2)
	batch_l, context_l = log_p1.shape
	cross = log_p1.unsqueeze(-1) + log_p2.unsqueeze(1)
	# build mask to search within bound steps
	mask = torch.ones(context_l, context_l).triu().tril(bound-1).unsqueeze(0)
	valid = cross * mask + (1.0 - mask) * -1e8

	spans = torch.zeros(batch_l, 2).long()
	for i in range(batch_l):
		max_idx = np.argmax(valid[i])
		max_idx = np.unravel_index(max_idx, valid[i].shape)
		spans[i] = torch.LongTensor(max_idx)
	return spans


def pick_spans_bounded_above(log_p1, log_p2, bound, threshold):
	log_p1, log_p2 = log_p1.cpu(), log_p2.cpu()
	assert(len(log_p1.shape) == 2)	# (batch_l, context_l)
	assert(len(log_p2.shape) == 2)
	batch_l, context_l = log_p1.shape
	cross = log_p1.unsqueeze(-1) + log_p2.unsqueeze(1)
	# build mask to search within bound steps
	mask = torch.ones(context_l, context_l).triu().tril(bound-1).unsqueeze(0)
	valid = cross * mask + (1.0 - mask) * -1e8

	rs = []
	for i in range(batch_l):
		idx = (valid[i] > threshold).nonzero()
		rs.append(idx)
	return rs

# log_p1 and log_p2 are the log probabilities of the start and end positions
# bound is the maximal answer span length
# topk is the k value for answer candidate selection
# different from pick_topk_spans_nonoverlap, here we allow overlap
def pick_topk_spans(log_p1, log_p2, bound, topk):
	log_p1, log_p2 = log_p1.cpu(), log_p2.cpu()
	assert(len(log_p1.shape) == 2)	# (batch_l, context_l)
	assert(len(log_p2.shape) == 2)
	batch_l, context_l = log_p1.shape
	cross = log_p1.unsqueeze(-1) + log_p2.unsqueeze(1)
	# build mask to search within bound steps
	mask = torch.ones(context_l, context_l).triu().tril(bound-1).unsqueeze(0)
	valid = cross * mask + (1.0 - mask) * -1e8

	rs = []
	for i in range(batch_l):
		valid_i = valid[i]
		idx_ls = valid_i.view(-1).topk(topk, dim=-1)[1]
		rs.append([])
		for idx in idx_ls:
			rs[-1].append((int(idx / valid_i.shape[-1]), int(idx % valid_i.shape[-1])))
		rs[-1] = torch.Tensor(rs[-1]).int()
	return rs


# log_p1 and log_p2 are the log probabilities of the start and end positions
# bound is the maximal answer span length
# topk is the k value for answer candidate selection
def pick_topk_spans_nonoverlap(log_p1, log_p2, bound, topk):
	log_p1, log_p2 = log_p1.cpu(), log_p2.cpu()
	assert(len(log_p1.shape) == 2)	# (batch_l, context_l)
	assert(len(log_p2.shape) == 2)
	batch_l, context_l = log_p1.shape
	cross = log_p1.unsqueeze(-1) + log_p2.unsqueeze(1)
	# build mask to search within bound steps
	mask = torch.ones(context_l, context_l).triu().tril(bound-1).unsqueeze(0)
	valid = cross * mask + (1.0 - mask) * -1e8

	rs = []
	for i in range(batch_l):
		valid_i = valid[i]
		rs.append([])
		for k in range(topk):
			if valid_i.sum() == -1e8 * valid_i.numel():
				break
			idx = valid_i.view(-1).topk(1, dim=-1)[1]
			start = idx / valid_i.shape[-1]
			end = idx % valid_i.shape[-1]
			valid_i[:end+1, start:] = -1e8	# non-overlap
			rs[-1].append((start, end))

		rs[-1] = torch.Tensor(rs[-1]).int()
	return rs



def pick_idx(p):
	p = p.cpu().numpy()
	return np.argmax(p, axis=1)

def count_correct_idx(pred, gold):
	return np.equal(pred, gold).sum()

def cleanup_G(seq):
	seq = seq.replace(' ##', '')	# for bert
	seq = seq.replace('##', '')
	seq = seq.replace('Ġ', '')	# for roberta
	seq = seq.replace(' Ġ', '')
	return seq

# get answer from tokenized indices in batch mode
def get_answer_tokenized(token_idx1, token_idx2, tokenized):
	#assert(len(token_idx1.shape) == 1)
	#assert(token_idx1.shape[0] == len(tokenized))

	batch_ans = []
	for i, (idx1, idx2) in enumerate(zip(token_idx1, token_idx2)):
		toks = tokenized[i][idx1:idx2+1]

		# compact answer
		ans_str = ' '.join(toks)
		ans_str = ans_str.strip()
		ans_str = cleanup_G(ans_str)
		#
		batch_ans.append(ans_str)
	return batch_ans


def get_em_bow(pred_ans, gold_ans):
	assert(len(pred_ans) == len(gold_ans))
	ems = []
	for pred, gold in zip(pred_ans, gold_ans):
		ems.append(metric_max_over_ground_truths(em_bow, pred, gold))
	return ems

def get_f1_bow(pred_ans, gold_ans):
	assert(len(pred_ans) == len(gold_ans))
	f1s = []
	for pred, gold in zip(pred_ans, gold_ans):
		f1s.append(metric_max_over_ground_truths(f1_bow, pred, gold))
	return f1s

def get_contain_bow(pred_ans, gold_ans):
	assert(len(pred_ans) == len(gold_ans))
	ems = []
	for pred, gold in zip(pred_ans, gold_ans):
		ems.append(metric_max_over_ground_truths(contain_bow, pred, gold))
	return ems


def contain_bow(prediction, ground_truth):
	p = normalize_answer(prediction)
	g = normalize_answer(ground_truth)
	return float((p in g or g in p))


def get_norm2(t):
	return (t * t).sum()



def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

def warmup_linear_flat(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is fixed. """
    if x < warmup:
        return x/warmup
    return 1.0


def linear_incr_decr(scale, cur_update, total_update, pivot):
	return scale * warmup_linear(float(cur_update) / total_update, pivot)

def linear_decr(scale, cur_update, total_update):
	return scale * (1.0 - min(1.0, float(cur_update) / total_update))

def rapid_decr(scale, cur_update, total_update):
	x = min(1.0, float(cur_update) / total_update)
	return scale * (2.0 / (1.0 + math.exp(8.0 * x)))

def rapid_decr(scale, cur_update, total_update):
	x = min(1.0, float(cur_update) / total_update)
	return scale * (2.0 / (1.0 + math.exp(8.0 * x)))


def save_json(data, output):
	print('saving to {0}'.format(output))
	json.dump(data, open(output, 'w'), indent=4)


if __name__ == '__main__':
	s1 = 'something in common (NAC)(PAG)'
	s2 = 'something weird'
	print(s1)
	print(s2)
	print(f1_bow(s1, s2))