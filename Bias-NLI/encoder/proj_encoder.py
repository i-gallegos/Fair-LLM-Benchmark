import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from holder import *
from util import *

class ProjEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ProjEncoder, self).__init__()
		self.proj = nn.Linear(opt.word_vec_size, opt.hidden_size, False)

		self.f = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())

		# bookkeeping
		self.opt = opt
		self.shared = shared


	def forward(self, sent1, sent2):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2
		input_size = self.opt.word_vec_size
		hidden_size = self.opt.hidden_size

		self.shared.input_emb1 = self.proj(sent1.view(batch_l * sent_l1, input_size))
		self.shared.input_emb2 = self.proj(sent2.view(batch_l * sent_l2, input_size))

		self.shared.input_enc1 = self.f(self.shared.input_emb1).view(batch_l, sent_l1, hidden_size)
		self.shared.input_enc2 = self.f(self.shared.input_emb2).view(batch_l, sent_l2, hidden_size)

		self.shared.input_emb1 = self.shared.input_emb1.view(batch_l, sent_l1, hidden_size)
		self.shared.input_emb2 = self.shared.input_emb2.view(batch_l, sent_l2, hidden_size)

		return [self.shared.input_emb1, self.shared.input_emb2, self.shared.input_enc1, self.shared.input_enc2]


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	from torch.autograd import Variable

	opt = Holder()
	opt.word_vec_size = 3
	opt.hidden_size = 4
	opt.dropout = 0.0
	shared = Holder()
	shared.batch_l = 1
	shared.sent_l1 = 5
	shared.sent_l2 = 8
	shared.input1 = Variable(torch.randn(shared.batch_l, shared.sent_l1, opt.word_vec_size), True)
	shared.input2 = Variable(torch.randn(shared.batch_l, shared.sent_l2, opt.word_vec_size), True)

	# build network
	encoder = ProjEncoder(opt, shared)

	# update batch info
	shared.batch_l = 1
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	# run network
	rs = encoder(shared.input1, shared.input2)
	print(rs)





	
