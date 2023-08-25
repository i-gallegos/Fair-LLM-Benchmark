import sys
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np

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


def isnan(x):
    return (x != x).sum() > 0


def rand_tensor(shape, r1, r2):
	return (r1 - r2) * torch.rand(shape) + r2


def pick_label(dist):
	return np.argmax(dist, axis=1)

def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()

def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.items():
		file.create_dataset(name, data=p)

	file.close()

def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f

def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			if len(l.rstrip().split()) < 3:
				print('weird line found: {0}'.format(l.rstrip()))
				continue
			w, idx, cnt = l.rstrip().split()
			rs[int(idx)] = w
	return rs


def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))

def simple_init(opt, m):
	classname = m.__class__.__name__
	if hasattr(m, 'weight'):
		m.weight.data.copy_(torch.randn(m.weight.data.shape)).mul_(opt.param_init)

	if hasattr(m, 'bias') and m.bias is not None:
		m.bias.data.copy_(torch.randn(m.bias.data.shape)).mul_(opt.param_init)

def save_optim(optim, path):
	file = h5py.File(path, 'w')
	for i, p in optim.parameters():
		file.create_dataset('{0}'.format(i), data=p)

	file.close()

def load_optim(path):
	f = h5py.File(path, 'r')
	# CLOSE??
	return f