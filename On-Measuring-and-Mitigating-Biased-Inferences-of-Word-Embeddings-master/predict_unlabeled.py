import sys
from pipeline import *
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
from holder import *
from data import *
from multiclass_loss import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/nli_bias/")
parser.add_argument('--data', help="Path to validation data hdf5 file.", default="unlabeled.hdf5")
parser.add_argument('--res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "unlabeled_glove.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "unlabeled.word.dict")
parser.add_argument('--load_file', help="Path to where model to be loaded.", default="")
# for bias
parser.add_argument('--debias', help="Whether to debias embeddings", type=int, default=0)
parser.add_argument('--bias_type', help="What type of bias to remove", default='')
parser.add_argument('--bias_glove', help="The glove bias vector", default='')
parser.add_argument('--bias_elmo', help="The elmo bias vector", default='')
parser.add_argument('--num_bias', help="The number o felmo bias vectors", type=int, default=1)
parser.add_argument('--contract_v1', help="The glove contraction vector1", default='')
parser.add_argument('--contract_v2', help="The glove contraction vector2", default='')
# generic parameter
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_uniform')
parser.add_argument('--param_init', help="The scale of the normal distribution from which weights are initialized", type=float, default=0.01)
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
## pipeline specs
parser.add_argument('--encoder', help="The type of encoder", default="proj")
parser.add_argument('--attention', help="The type of attention", default="local")
parser.add_argument('--classifier', help="The type of classifier", default="local")
parser.add_argument('--rnn_layer', help="The number of layers of rnn encoder", type=int, default=1)
parser.add_argument('--rnn_type', help="What type of rnn to use, default lstm", default='lstm')
parser.add_argument('--birnn', help="Whether to use bidirectional rnn", type=int, default=1)
parser.add_argument('--num_label', help="The number of prediction labels", type=int, default=3)
# dimensionality
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=200)
parser.add_argument('--cls_hidden_size', help="The hidden size of the classifier", type=int, default=200)
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
# elmo specs
parser.add_argument('--elmo_in_size', help="The input elmo dim", type=int, default=1024)
parser.add_argument('--elmo_size', help="The hidden elmo dim", type=int, default=1024)
parser.add_argument('--elmo_layer', help="The number of elmo layers", type=int, default=3)
parser.add_argument('--use_elmo_post', help="Whether to use elmo after encoder", type=int, default=1)
parser.add_argument('--dynamic_elmo', help="Whether to use elmo model to parse text dynamically, or use cached ELMo", type=int, default=0)
parser.add_argument('--elmo_dropout', help="The dropout probability on ELMO", type=float, default=0.0)
parser.add_argument('--elmo_blend', help="The type of blending function for elmo, e.g. interpolate/concat", default="interpolate")
parser.add_argument('--use_elmo_only', help="Whether to use elmo only, i.e. ignore glove.", type=int, default="0")
# specs for unlabeled
parser.add_argument('--pred_output', help="Prediction output file", default='./models/unlabeled_pred.txt')
#parser.add_argument('--sent1', help="The path to tokenized premise file", default='unlabeled.sent1.txt')
#parser.add_argument('--sent2', help="The path to tokenized hypothesis file", default='unlabeled.sent2.txt')
#parser.add_argument('--x_pair', help="The path to sentence template x1 and x2 words", default='unlabeled.x_pair.txt')

def load_sent(path):
	print('loading tokenized sentences from {0}'.format(path))
	sents = []
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			sents.append(l.rstrip().split(' '))
	return sents

def load_x_pairs(path):
	print('laoding x pairs from {0}'.format(path))
	x1, x2 = [], []
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			toks = l.rstrip().split()
			x1.append(toks[0])
			x2.append(toks[1])
	return x1, x2


def write_log(path, lines):
	print('writing log to {0}'.format(path))
	with open(path, 'w') as f:
		for l in lines:
			f.write(l + '\n')


def evaluate(opt, shared, m, data):
	m.train(False)

	batch_cnt = 0
	val_loss = 0.0
	num_ex = 0

	loss = MulticlassLoss(opt, shared)

	val_idx, val_num_ex = data.subsample(1.0)
	data_size = val_idx.size()[0]
	print('evaluating on {0} batches {1} examples'.format(data_size, val_num_ex))

	log = ['x1,x2,premise,hypothesis,entail_probability,neutral_probability,contradiction_probability']

	m.begin_pass()
	for i in range(data_size):
		(data_name, source, target,
			batch_ex_idx, batch_l, source_l, target_l, label, res_map) = data[val_idx[i]]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		y_gold = Variable(label, requires_grad=False)

		# update network parameters
		m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)

		# forward pass
		pred = m.forward(wv_idx1, wv_idx2)

		# loss
		batch_loss = loss(pred, y_gold)

		# stats
		val_loss += float(batch_loss.data)
		num_ex += batch_l

		# logging
		dist = pred.data.exp()
		for k, ex_idx in enumerate(batch_ex_idx):
			# output format is: premise, hypothesis, p(E), p(N), p(C)
			log.append('{0},{1},{2},{3},{4:.4f},{5:.4f},{6:.4f}'.format(res_map['x_pair'][k][0], res_map['x_pair'][k][1], ' '.join(res_map['sent1'][k]), ' '.join(res_map['sent2'][k]), float(dist[k][0]), float(dist[k][1]), float(dist[k][2])))

		if (batch_cnt + 1) % 1000 == 0:
			print('predicted {0} batches'.format(batch_cnt + 1))
		batch_cnt += 1

	perf, extra_perf = loss.get_epoch_metric()
	m.end_pass()
	print('finished evaluation on {0} examples'.format(num_ex))

	# printing
	write_log(opt.pred_output, log)

	return (perf, extra_perf, val_loss / num_ex, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.data = opt.dir + opt.data
	opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])
	opt.word_vecs = opt.dir + opt.word_vecs
	opt.dict = opt.dir + opt.dict
	opt.bias_glove = opt.dir + opt.bias_glove
	opt.bias_elmo = opt.dir + opt.bias_elmo
	opt.contract_v1 = opt.dir + opt.contract_v1
	opt.contract_v2 = opt.dir + opt.contract_v2
	#opt.sent1 = opt.dir + opt.sent1 
	#opt.sent2 = opt.dir + opt.sent2
	#opt.x_pair = opt.dir + opt.x_pair

	#shared.sent1 = load_sent(opt.sent1)
	#shared.sent2 = load_sent(opt.sent2)
	#shared.x1, shared.x2 = load_x_pairs(opt.x_pair)

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# build model
	m = Pipeline(opt, shared)

	# initialization
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	if opt.gpuid != -1:
		m = m.cuda()

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data, res_files)

	#
	perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
	#print('Val {0:.4f} Extra {1} Loss: {2:.4f}'.format(
	#	perf, extra_perf_str, avg_loss))

	#print('saving model to {0}'.format('tmp'))
	#param_dict = m.get_param_dict()
	#save_param_dict(param_dict, '{0}.hdf5'.format('tmp'))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
