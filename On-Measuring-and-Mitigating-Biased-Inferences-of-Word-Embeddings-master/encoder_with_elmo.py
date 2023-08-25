import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from elmo_encoder import *
from elmo_loader import *


class EncoderWithElmo(torch.nn.Module):
	def __init__(self, opt, shared):
		super(EncoderWithElmo, self).__init__()

		# bookkeeping
		self.opt = opt
		self.shared = shared

		self.elmo_drop = nn.Dropout(opt.elmo_dropout)
		self.drop = LockedDropout(opt.dropout)

		if opt.dynamic_elmo == 1:
			self.elmo = ElmoEncoder(opt, shared)
		else:
			self.elmo = ElmoLoader(opt, shared)


		# rnn merger
		bidir = opt.birnn == 1
		rnn_in_size = opt.word_vec_size + opt.elmo_in_size if opt.use_elmo_only == 0 else opt.elmo_in_size
		rnn_hidden_size = opt.hidden_size if not bidir else opt.hidden_size//2
		self.rnn = build_rnn(
			opt.rnn_type,
			input_size=rnn_in_size,
			hidden_size=rnn_hidden_size,
			num_layers=opt.rnn_layer,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir)

		if opt.elmo_blend == 'concat':
			self.sampler_pre = nn.Linear(opt.elmo_in_size*3, opt.elmo_in_size)
			self.sampler_post = nn.Linear(opt.elmo_in_size*3, opt.elmo_in_size)

		elif opt.elmo_blend == 'interpolate':
			self.gamma_pre = nn.Parameter(torch.ones(1), requires_grad=True)
			self.gamma_pre.skip_init = 1
			self.gamma_post = nn.Parameter(torch.ones(1), requires_grad=True)
			self.gamma_post.skip_init = 1
	
			self.w_pre = nn.Parameter(torch.ones(3), requires_grad=True)
			self.w_pre.skip_init = 1
			self.w_post = nn.Parameter(torch.ones(3), requires_grad=True)
			self.w_post.skip_init = 1

			self.softmax = nn.Softmax(0)


	def rnn_over(self, x):
		x = self.drop(x)
		x, h = self.rnn(x)
		return x, h


	def interpolate_elmo(self, elmo_layers1, elmo_layers2, w, gamma):
		weights = nn.Softmax(0)(w)
		# interpolate
		if self.opt.elmo_layer == 3:
			sent1 = elmo_layers1[0] * weights[0] + elmo_layers1[1] * weights[1] + elmo_layers1[2] * weights[2]
			sent2 = elmo_layers2[0] * weights[0] + elmo_layers2[1] * weights[1] + elmo_layers2[2] * weights[2]
		elif self.opt.elmo_layer == 2:
			sent1 = elmo_layers1[0] * weights[0] + elmo_layers1[1] * weights[1]
			sent2 = elmo_layers2[0] * weights[0] + elmo_layers2[1] * weights[1]
		elif self.opt.elmo_layer == 1:
			sent1 = elmo_layers1[0] * weights[0]
			sent2 = elmo_layers2[0] * weights[0]
		return sent1*gamma, sent2*gamma


	def concat_elmo(self, elmo_layers1, elmo_layers2):
		return torch.cat(elmo_layers1, 2), torch.cat(elmo_layers2, 2)


	def sample_elmo(self, sampler, elmo1, elmo2):
		elmo1 = sampler(elmo1.view(-1, self.opt.elmo_in_size*3)).view(self.shared.batch_l, self.shared.sent_l1, -1)
		elmo2 = sampler(elmo2.view(-1, self.opt.elmo_in_size*3)).view(self.shared.batch_l, self.shared.sent_l2, -1)
		return elmo1, elmo2


	def forward(self, sent1, sent2):
		# elmo pass
		elmo1, elmo2 = self.elmo()

		# pre-rnn elmo
		elmo_pre1, elmo_pre2 = None, None
		if self.opt.elmo_blend == 'interpolate':
			elmo_pre1, elmo_pre2 = self.interpolate_elmo(elmo1, elmo2, self.w_pre, self.gamma_pre)
		elif self.opt.elmo_blend == 'concat':
			elmo_pre1, elmo_pre2 = self.concat_elmo(elmo1, elmo2)
			elmo_pre1, elmo_pre2 = self.sample_elmo(self.sampler_pre, elmo_pre1, elmo_pre2)

		elmo_pre1, elmo_pre2 = self.elmo_drop(elmo_pre1), self.elmo_drop(elmo_pre2)

		enc1, enc2 = elmo_pre1, elmo_pre2
		if self.opt.use_elmo_only == 0:
			enc1 = torch.cat([sent1, enc1], 2)
			enc2 = torch.cat([sent2, enc2], 2)

		# read
		enc1, _ = self.rnn_over(enc1)
		enc2, _ = self.rnn_over(enc2)

		# post-rnn elmo
		if self.opt.use_elmo_post == 1:
			elmo_post1, elmo_post2 = None, None
			if self.opt.elmo_blend == 'interpolate':
				elmo_post1, elmo_post2 = self.interpolate_elmo(elmo1, elmo2, self.w_post, self.gamma_post)
			elif self.opt.elmo_blend == 'concat':
				elmo_post1, elmo_post2 = self.concat_elmo(elmo1, elmo2)
				elmo_post1, elmo_post2 = self.sample_elmo(self.sampler_post, elmo_post1, elmo_post2)

			elmo_post1, elmo_post2 = self.elmo_drop(elmo_post1), self.elmo_drop(elmo_post2)
			enc1 = torch.cat([enc1, elmo_post1], 2)
			enc2 = torch.cat([enc2, elmo_post2], 2)

		# record
		#	take lstm encoding as embeddings for classification
		#	take post-lstm encoding as encodings for attention
		self.shared.input_emb1 = enc1
		self.shared.input_emb2 = enc2
		self.shared.input_enc1 = enc1
		self.shared.input_enc2 = enc2

		return [self.shared.input_emb1, self.shared.input_emb2, self.shared.input_enc1, self.shared.input_enc2]

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass




	
