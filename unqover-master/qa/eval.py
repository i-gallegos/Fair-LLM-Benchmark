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
from utils.util import *
from .data import *
from .boundary_loss import *
from .pipeline import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/squad/")
parser.add_argument('--data', help="Path to training data hdf5 file.", default="")
parser.add_argument('--load_file', help="Path to where model to be loaded.", default="")
parser.add_argument('--is_hf', help="Whether the load_file points to a hf model (either online or local)", type=int, default=0)
# resource specs
parser.add_argument('--res', help="Path to validation resource files, seperated by comma.", default="")
## dim specs
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.1)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=30)
# bert specs
parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
## pipeline stages
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--span_l', help="The maximal span length allowed for prediction", type=int, default=30)
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of loss, boundary", default='boundary')
#
parser.add_argument('--output_official', help="The path to official format of output", default='')
#
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)



def load_hf_model(m, m_hf):
	if isinstance(m_hf, RobertaForQuestionAnswering):
		m.encoder.transformer = m_hf.roberta
		m.classifier.linear[1] = m_hf.qa_outputs
	elif isinstance(m_hf, DistilBertForQuestionAnswering):
		m.encoder.transformer = m_hf.distilbert
		m.classifier.linear[1] = m_hf.qa_outputs
	elif isinstance(m_hf, BertForQuestionAnswering):
		m.encoder.transformer = m_hf.bert
		m.classifier.linear[1] = m_hf.qa_outputs
	else:
		raise Exception('unrecognized HF model type {0}'.format(type(hf_m)))


def evaluate(opt, shared, m, data):

	val_loss = 0.0
	num_ex = 0

	loss = None
	if opt.loss == 'boundary':
		loss = BoundaryLoss(opt, shared)
	elif opt.loss == 'boundary_reg':
		loss = BoundaryRegLoss(opt, shared)
	else:
		assert(False)

	loss.verbose = opt.verbose==1

	shared.has_gold = False
	m.train(False)
	data.begin_pass()
	m.begin_pass()
	loss.begin_pass()
	for i in range(data.size()):
		(data_name, batch_ex_idx, concated, batch_l, concated_l, context_l, query_l,
				concated_span, context_start, res_map) = data[i]

		wv_idx = Variable(concated, requires_grad=False)
		y_gold = Variable(concated_span, requires_grad=False)

		m.update_context(batch_ex_idx, batch_l, concated_l, context_l, query_l, context_start, res_map)

		# forward pass
		with torch.no_grad():
			output = m.forward(wv_idx)

		# loss
		batch_loss = loss(output, y_gold, to_device(torch.ones(1), opt.gpuid))

		# stats
		val_loss += float(batch_loss.data)
		num_ex += batch_l

	perf, extra_perf = loss.get_epoch_metric()
	loss.end_pass()
	m.end_pass()
	data.end_pass()
	print('finished evaluation on {0} examples'.format(num_ex))

	return (perf, extra_perf, val_loss / num_ex, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# fix some hyperparameters automatically
	if 'base' in opt.transformer_type:
		opt.hidden_size = 768
	elif 'large'in opt.transformer_type:
		opt.hidden_size = 1024

	# 
	opt.data = opt.dir + opt.data
	opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(1)

	# build model
	m = Pipeline(opt, shared)

	if not opt.is_hf:
		# initialization
		print('loading pretrained model from {0}...'.format(opt.load_file))
		param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
		m.set_param_dict(param_dict)
	else:
		m_hf = AutoModelForQuestionAnswering.from_pretrained(opt.load_file)
		load_hf_model(m, m_hf)

	if opt.fp16 == 1:
		m.half()

	if opt.gpuid != -1:
		m = m.cuda()

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data, res_files)

	#
	perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
	print('Val {0:.4f} Extra {1} Loss: {2:.4f}'.format(
		perf, extra_perf_str, avg_loss))

	#print('saving model to {0}'.format('tmp'))
	#param_dict = m.get_param_dict()
	#save_param_dict(param_dict, '{0}.hdf5'.format('tmp'))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
