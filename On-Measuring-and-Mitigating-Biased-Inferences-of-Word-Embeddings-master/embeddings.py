import sys
import h5py
import torch
from torch import nn
from torch import cuda
from holder import *
from util import *
from embedding_bias import *

class Embeddings(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Embeddings, self).__init__()
		self.opt = opt
		self.shared = shared

		if opt.debias == 1:
			self.embedding_bias = EmbeddingBias(opt, shared)
	
		print('loading word vector from {0}'.format(opt.word_vecs))
		f = h5py.File(opt.word_vecs, 'r')
		word_vecs = f['word_vecs'][:]
		assert(opt.word_vec_size == word_vecs.shape[1])
		num_tok = word_vecs.shape[0]

		print('loading word dict from {0}'.format(opt.dict))
		if opt.dict != '':
			self.vocab = load_dict(opt.dict)

		# assumes <blank> is the first, the second is the oov
		# 	and assumes there is exactly one oov
		assert(self.vocab[0] == '<blank>')
		assert(self.vocab[1] == '<s>')
		assert(self.vocab[2] == '<oov0>')

		self.embeddings = nn.Embedding(num_tok, opt.word_vec_size)
		self.embeddings.weight.data[0,:] = torch.zeros(1, opt.word_vec_size).float()
		# load all w2v including oov from preprocessed hdf5
		#self.embeddings.weight.data[1:] = rand_tensor((1, opt.word_vec_size), -0.05, 0.05).float()
		self.embeddings.weight.data[1:] = torch.from_numpy(word_vecs[1:]).float()
		self.embeddings.weight.requires_grad = opt.fix_word_vecs == 0
		self.embeddings.weight.skip_init = 1
		self.embeddings.weight.skip_save = 1

		# concat to form embedding variable
		#self.embeddings = torch.cat([self.blank_weight, self.oov_weight, self.word_vec_weight], 0)

	# incoming idx of shape (batch_l, seq_l)
	def forward(self, idx):
		batch_l, seq_l = idx.shape
		idx = idx.contiguous().view(-1)	# flatten to form a single vector (pytorch 0.3.1 does not support tensor idx)
		emb = self.embeddings(idx).view(batch_l, seq_l, self.opt.word_vec_size)

		# if to debias
		if hasattr(self, 'embedding_bias'):
			emb = self.embedding_bias(emb)

		return emb


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass
