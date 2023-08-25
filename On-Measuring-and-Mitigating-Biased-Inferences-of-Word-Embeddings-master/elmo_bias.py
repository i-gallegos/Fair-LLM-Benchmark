import sys
import h5py
import torch
from torch import nn
from torch import cuda
from holder import *
from util import *


class ElmoBias(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ElmoBias, self).__init__()
		self.opt = opt
		self.shared = shared

		print('loading embedding bias from {0}'.format(opt.bias_elmo))
		f = h5py.File(opt.bias_elmo, 'r')
		bias_elmo = f['bias'][:]

		assert(opt.fix_word_vecs == 1)	# this will not work with dynamic word embeddings, natually
	
		if opt.bias_type == 'removal':
			bias_elmo = np.ascontiguousarray(bias_elmo[:, :])	# the original elmo bias are [l2, l1, l0], we might need [l0, l1, l2]
			self.bias_elmo = nn.Parameter(torch.from_numpy(bias_elmo).float().view(1, opt.num_bias, -1), requires_grad=False)
			self.bias_elmo.skip_init = 1
			self.bias_elmo.skip_save = 1
		else:
			raise Exception('unrecognized bias_type {0}'.format(self.opt.bias_type))


	def debias(self, enc, bias):
		batch_l, sent_l, elmo_size = enc.shape

		if self.opt.bias_type == 'removal':
			bias = bias.expand(batch_l, 1, elmo_size)
			proj = enc.bmm(bias.transpose(1,2))
			return enc - (proj * bias)
		elif self.opt.bias_type == 'nationality':
			assert(False)
		else:
			raise Exception('unrecognized bias_type {0}'.format(self.opt.bias_type))


	# input elmo_enc should have shape (batch_l, sent_l, elmo_in_size*3)
	def forward(self, elmo_enc):
		batch_l, sent_l, enc_size = elmo_enc.shape
		assert(enc_size == self.opt.elmo_in_size*3)

		debiased = []
		for i in range(3):
			start = self.opt.elmo_in_size * i
			end = self.opt.elmo_in_size * (i+1)

			# each time we get (batch_l, sent_l, elmo_in_size)
			debiased.append(self.debias(elmo_enc[:, :, start:end], self.bias_elmo[:, :, start:end]))

		debiased = torch.cat(debiased, 2)
		return debiased



if __name__ == '__main__':
	pass
