import sys
sys.path.append('../allennlp')
import h5py
import torch
from torch import nn
from torch import cuda
from holder import *
from util import *
from contractionFunc import *
from torch.autograd import Variable
from contraction import *
import numpy as np

class EmbeddingBias(torch.nn.Module):
	def __init__(self, opt, shared):
		super(EmbeddingBias, self).__init__()
		self.opt = opt
		self.shared = shared

		if opt.bias_glove != opt.dir:
			print('loading embedding bias from {0}'.format(opt.bias_glove))
			f = h5py.File(opt.bias_glove, 'r')
			bias_glove = f['bias'][:]

		if opt.contract_v1 != opt.dir:
			print('loading embedding contract1 from {0}'.format(opt.contract_v1))
			f = h5py.File(opt.contract_v1, 'r')
			v1 = f['bias'][:]
		if opt.contract_v2 != opt.dir:
			print('loading embedding contract2 from {0}'.format(opt.contract_v2))
			f = h5py.File(opt.contract_v2, 'r')
			v2 = f['bias'][:]

		assert(opt.fix_word_vecs == 1)	# this will not work with dynamic word embeddings, natually

		if opt.bias_glove != opt.dir:
			if opt.bias_type == 'removal1':
				self.bias_glove = nn.Parameter(torch.ones(1, 1, opt.word_vec_size), requires_grad=False)
				self.bias_glove.data = torch.from_numpy(bias_glove).float().view(1, 1, opt.word_vec_size)
				self.bias_glove.skip_init = 1
				self.bias_glove.skip_save = 1
			elif opt.bias_type == 'removal2':
				self.bias_glove = nn.Parameter(torch.ones(1, 2, opt.word_vec_size), requires_grad=False)
				self.bias_glove.data = torch.from_numpy(bias_glove).float().view(1, 2, opt.word_vec_size)
				self.bias_glove.skip_init = 1
				self.bias_glove.skip_save = 1
			elif opt.bias_type == 'removal3':
				self.bias_glove = nn.Parameter(torch.ones(1, 3, opt.word_vec_size), requires_grad=False)
				self.bias_glove.data = torch.from_numpy(bias_glove).float().view(1, 3, opt.word_vec_size)
				self.bias_glove.skip_init = 1
				self.bias_glove.skip_save = 1
			else:
				raise Exception('unrecognized bias_type {0}'.format(self.opt.bias_type))
	
		if opt.contract_v1 != opt.dir:
			if opt.bias_type == 'contract':
				v1 = torch.from_numpy(v1).view(-1, opt.word_vec_size).numpy()
				v2 = torch.from_numpy(v2).view(-1, opt.word_vec_size).numpy()
	
				v1, v2 = maxSpan(v1, v2)
				U = np.identity(opt.word_vec_size)
				U = gsConstrained(U, v1, basis(np.vstack((v1, v2))))
				
				self.contract_glove1 = nn.Parameter(torch.from_numpy(v1).float().view(1,1,opt.word_vec_size), requires_grad=False)
				self.contract_glove1.skip_init = 1
				self.contract_glove1.skip_save = 1
				self.contract_glove2 = nn.Parameter(torch.from_numpy(v2).float().view(1,1,opt.word_vec_size), requires_grad=False)
				self.contract_glove2.skip_init = 1
				self.contract_glove2.skip_save = 1
				self.contract_U = nn.Parameter(torch.from_numpy(U).float().view(1,opt.word_vec_size,opt.word_vec_size), requires_grad=False)
				self.contract_U.skip_init = 1
				self.contract_U.skip_save = 1
	
			else:
				raise Exception('unrecognized bias_type {0}'.format(self.opt.bias_type))


	def contraction_correct(self, enc):
		rec = enc
		enc = correction(self.opt, self.contract_U, self.contract_glove1, self.contract_glove2, enc)
		
		return enc


	def forward(self, glove_enc):
		batch_l, sent_l, glove_size = glove_enc.shape

		if self.opt.bias_glove != self.opt.dir:
			if self.opt.bias_type == 'removal1':
				bias = self.bias_glove.expand(batch_l, 1, glove_size)
				proj = glove_enc.bmm(bias.transpose(1,2))	# batch_l, sent_l, 1
				return glove_enc - (proj * bias)
			elif self.opt.bias_type == 'removal2':
				bias1 = self.bias_glove[:, 0:1, :].expand(batch_l, 1, glove_size)
				bias2 = self.bias_glove[:, 1:2, :].expand(batch_l, 1, glove_size)
				proj1 = glove_enc.bmm(bias1.transpose(1,2))
				proj2 = glove_enc.bmm(bias2.transpose(1,2))
				return glove_enc - (proj1*bias1) - (proj2*bias2)
			elif self.opt.bias_type == 'removal3':
				bias1 = self.bias_glove[:, 0:1, :].expand(batch_l, 1, glove_size)
				bias2 = self.bias_glove[:, 1:2, :].expand(batch_l, 1, glove_size)
				bias3 = self.bias_glove[:, 2:3, :].expand(batch_l, 1, glove_size)
				proj1 = glove_enc.bmm(bias1.transpose(1,2))
				proj2 = glove_enc.bmm(bias2.transpose(1,2))
				proj3 = glove_enc.bmm(bias3.transpose(1,2))
				return glove_enc - (proj1*bias1) - (proj2*bias2) - (proj3*bias3)
			else:
				raise Exception('unrecognized bias_type {0}'.format(self.opt.bias_type))

		if self.opt.contract_v1 != self.opt.dir:
			if self.opt.bias_type == 'contract':
				return self.contraction_correct(glove_enc)
			else:
				raise Exception('unrecognized bias_type {0}'.format(self.opt.bias_type))


if __name__ == '__main__':
	pass
