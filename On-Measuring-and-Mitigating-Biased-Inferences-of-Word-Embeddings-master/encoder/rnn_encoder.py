import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from locked_dropout import *


class RNNEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(RNNEncoder, self).__init__()

		self.bidir = opt.birnn == 1
		hidden_state = opt.hidden_size if not self.bidir else opt.hidden_size//2

		self.rnn = build_rnn(
			opt.rnn_type,
			input_size=opt.word_vec_size, 
			hidden_size=hidden_state, 
			num_layers=opt.rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=self.bidir)

		self.drop = LockedDropout(opt.dropout)

		# bookkeeping
		self.opt = opt
		self.shared = shared


	def rnn_over(self, rnn, enc):
		enc, _ = rnn(self.drop(enc))
		return enc


	def forward(self, sent1, sent2):
		lstm_enc1 = self.rnn_over(self.rnn, sent1)
		lstm_enc2 = self.rnn_over(self.rnn, sent2)
		lstm_enc1 = lstm_enc1.contiguous()
		lstm_enc2 = lstm_enc2.contiguous()

		# record
		#	take lstm encoding as embeddings for classification
		#	take post-lstm encoding as encodings for attention
		self.shared.input_emb1 = lstm_enc1
		self.shared.input_emb2 = lstm_enc2
		self.shared.input_enc1 = lstm_enc1
		self.shared.input_enc2 = lstm_enc2

		return [self.shared.input_emb1, self.shared.input_emb2, self.shared.input_enc1, self.shared.input_enc2]

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass




	
