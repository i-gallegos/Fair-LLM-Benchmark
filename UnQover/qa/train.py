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
from .pipeline import *
from .optimizer import *
from .data import *
from utils.util import *
from .boundary_loss import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/squad/")
parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="squad.train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
# resource specs
parser.add_argument('--train_res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--val_res', help="Path to validation resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline, will try to set it automatically, but no guarantee.", type=int, default=768)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.1)
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--div_percent', help="The percent of training data to divide as train/val", type=float, default=0.8)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=30)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adamw_fp16')
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.00001)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=1.0)
parser.add_argument('--adam_betas', help="The betas used in adam", default='0.9,0.999')
parser.add_argument('--weight_decay', help="The factor of weight decay", type=float, default=0.01)
# bert specs
parser.add_argument('--transformer_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
parser.add_argument('--warmup_perc', help="The percentages of total expectec updates to warmup", type=float, default=0.1)
parser.add_argument('--warmup_type', help="The type of warmup", default="linear")
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of loss, default boundary", default='boundary')
#
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=200)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--span_l', help="The maximal span length allowed for prediction", type=int, default=10000)
#
parser.add_argument('--skip_eval', help="Whether to skip epoch evaluation", type=int, default=0)
parser.add_argument('--output_gold', help="The path to training gold output, in official format", default='')
parser.add_argument('--output_official', help="The path to prediction output, in official format", default='')


def train_epoch(opt, shared, m, optim, data, epoch_id, sub_idx):
	train_loss = 0.0
	num_ex = 0
	start_time = time.time()
	train_idx1_correct = 0
	train_idx2_correct = 0
	min_grad_norm2 = 1000000000000.0
	max_grad_norm2 = 0.0
	total_em_bow = 0.0
	total_f1_bow = 0.0

	loss = None
	if opt.loss == 'boundary':
		loss = BoundaryLoss(opt, shared)
	else:
		assert(False)

	data_size = 0
	all_data = []
	if data is not None:
		data_size = len(sub_idx)
		batch_order = torch.randperm(data_size)
		all_data = []
		for i in range(data_size):
			all_data.append((data, sub_idx[i]))

	shared.is_train = True
	m.train(True)
	loss.begin_pass()
	if data is not None:
		data.begin_pass()
	m.begin_pass()

	for i in range(data_size):
		shared.epoch = epoch_id
		shared.skip_backward = False

		cur_data, cur_idx = all_data[batch_order[i]]

		# do a normal pass
		(data_name, batch_ex_idx, concated, batch_l, concated_l, context_l, query_l,
				concated_span, context_start, res_map) = cur_data[cur_idx]
	
		wv_idx = Variable(concated, requires_grad=False)
		y_gold = Variable(concated_span, requires_grad=False)
		m.update_context(batch_ex_idx, batch_l, concated_l, context_l, query_l, context_start, res_map)
	
		output = m.forward(wv_idx)

		batch_loss = loss(output, y_gold, to_device(torch.ones(1), opt.gpuid))

		# stats
		train_loss += float(batch_loss.data)
		num_ex += batch_l
		time_taken = time.time() - start_time

		# average batch loss
		avg_batch_loss = batch_loss / max(float(batch_l), 1)

		# accumulate grads
		if not shared.skip_backward: 
			optim.backward(m, avg_batch_loss)
			grad_norm2 = optim.step(m)
			# clear up grad
			m.zero_grad()
			# stats
			shared.num_update += 1

			min_grad_norm2 = min(min_grad_norm2, grad_norm2)
			max_grad_norm2 = max(max_grad_norm2, grad_norm2)


		# printing
		if i == data_size-1 or (i+1) % opt.print_every == 0:
			time_taken = time.time() - start_time

			if (i+1) % opt.print_every == 0 or i == data_size-1:
				stats = '{0}, Batch {1:.1f}k '.format(epoch_id+1, float(i+1)/1000)
				stats += 'Grad {0:.1f}/{1:.1f} '.format(min_grad_norm2, max_grad_norm2)
				stats += 'Loss {0:.4f} '.format(train_loss / num_ex)
				if data is not None:
					stats += loss.print_cur_stats()
				stats += 'Time {0:.1f}'.format(time_taken)
				print(stats)


	perf = -1
	extra_perf = []
	if data is not None:
		perf, extra_perf = loss.get_epoch_metric()

	shared.is_train = False
	m.end_pass()
	loss.end_pass()
	if data is not None:
		data.end_pass()

	return perf, extra_perf, train_loss / num_ex, num_ex


def train(opt, shared, m, optim, train_data, val_data):
	best_val_perf = 0.0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []
	extra_perfs = []


	if train_data is not None:
		print('{0} batches in train set'.format(train_data.size()))
		if val_data is not None:
			print('{0} batches in dev set'.format(val_data.size()))
		else:
			print('no dev set specified, will split train set into train/dev folds')

	train_idx = None
	train_num_ex = 0
	val_idx = None
	val_num_ex = 0
	if train_data is not None:
		print('subsampling train set by {0}'.format(opt.percent))
		train_idx, train_num_ex = train_data.subsample(opt.percent, random=True)
		print('for the record, first 10 batches: {0}'.format(train_idx[:10]))

		if val_data is None:
			val_data = train_data
			print('splitting train set into train/dev folds by {0}'.format(opt.div_percent))
			train_idx, val_idx, train_num_ex, val_num_ex = train_data.split(train_idx, opt.div_percent)
		else:
			val_idx, val_num_ex = val_data.subsample(1.0, random=False)	# use all val data as dev set

	num_train_batch = 0
	shared.num_train_ex = 0
	if train_data is not None:
		num_train_batch += len(train_idx)
		shared.num_train_ex += train_num_ex
	
	if train_idx is not None:
		print('final train set has {0} batches {1} examples'.format(num_train_batch, shared.num_train_ex))
		print('for the record, first 10 batches: {0}'.format(train_idx[:10]))
	
	if val_idx is not None:
		print('final val set has {0} batches {1} examples'.format(len(val_idx), val_num_ex))
		print('for the record, first 10 batches: {0}'.format(val_idx[:10]))
	
	start = 0
	for i in range(start, opt.epochs):
		train_perf, extra_train_perf, loss, num_ex = train_epoch(opt, shared, m, optim, train_data, i, train_idx)
		train_perfs.append(train_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_train_perf])
		print('Train {0:.4f} All {1}'.format(train_perf, extra_perf_str))

		if opt.skip_eval == 1:
			print('saving model to {0}.hdf5'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
			continue

		# evaluate
		#	and save if it's the best model
		val_perf, extra_val_perf, val_loss, num_ex = validate(opt, shared, m, val_data, val_idx)
		val_perfs.append(val_perf)
		extra_perfs.append(extra_val_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_val_perf])
		print('Val {0:.4f} All {1}'.format(val_perf, extra_perf_str))

		perf_table_str = ''
		cnt = 0
		print('Epoch  | Train | Val ...')
		for train_perf, extra_perf in zip(train_perfs, extra_perfs):
			extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
			perf_table_str += '{0}\t{1:.4f}\t{2}\n'.format(cnt+1, train_perf, extra_perf_str)
			cnt += 1
		print(perf_table_str)

		if val_perf > best_val_perf:
			best_val_perf = val_perf
			print('saving model to {0}'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
			save_opt(opt, '{0}.opt'.format(opt.save_file))

		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, m, val_data, val_idx):

	val_loss = 0.0
	num_ex = 0

	loss = None
	if opt.loss == 'boundary':
		loss = BoundaryLoss(opt, shared)
	else:
		assert(False)

	print('validating on the {0} batches...'.format(len(val_idx)))

	m.train(False)
	val_data.begin_pass()
	loss.begin_pass()
	m.begin_pass()
	for i in range(len(val_idx)):

		(data_name, batch_ex_idx, concated, batch_l, concated_l, context_l, query_l,
				concated_span, context_start, res_map) = val_data[val_idx[i]]

		#print(batch_l, concated_l)
		wv_idx = Variable(concated, requires_grad=False)
		y_gold = Variable(concated_span, requires_grad=False)

		m.update_context(batch_ex_idx, batch_l, concated_l, context_l, query_l, context_start, res_map)

		# forward pass
		with torch.no_grad():
			output = m.forward(wv_idx)

		# loss
		batch_loss = loss(output, y_gold, lambd=torch.ones(1))

		# stats
		val_loss += float(batch_loss.data)
		num_ex += batch_l

	perf, extra_perf = loss.get_epoch_metric()
	m.end_pass()
	loss.end_pass()
	val_data.end_pass()

	return (perf, extra_perf, val_loss / num_ex, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	shared.num_update = 0		# number of updates (total)

	# fix some hyperparameters automatically
	if '-base' in opt.transformer_type:
		opt.hidden_size = 768
	elif '-large' in opt.transformer_type:
		opt.hidden_size = 1024

	# 
	opt.train_data = opt.dir + opt.train_data
	opt.val_data = opt.dir + opt.val_data
	opt.train_res = '' if opt.train_res == ''  else ','.join([opt.dir + path for path in opt.train_res.split(',')])
	opt.val_res = '' if opt.val_res == ''  else ','.join([opt.dir + path for path in opt.val_res.split(',')])

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	print(opt)

	# build model
	m = Pipeline(opt, shared)
	optim = get_optimizer(opt, shared)

	# initializing from pretrained
	if opt.load_file != '':
		m.init_weight()
		print('loading pretrained model from {0}...'.format(opt.load_file))
		param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
		m.set_param_dict(param_dict)
	else:
		m.init_weight()
		model_parameters = filter(lambda p: p.requires_grad, m.parameters())
		num_params = sum([np.prod(p.size()) for p in model_parameters])
		print('total number of trainable parameters: {0}'.format(num_params))
	
	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu
	m = optim.build_optimizer(m)	# build optimizer after distributing model to devices

	# loading data
	train_data = None
	if opt.train_data != opt.dir:
		train_data = Data(opt, opt.train_data, None if opt.train_res == '' else opt.train_res.split(','))
	
	val_data = None
	if opt.val_data != opt.dir:
		val_data = Data(opt, opt.val_data, None if opt.val_res == '' else opt.val_res.split(','))

	train(opt, shared, m, optim, train_data, val_data)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
