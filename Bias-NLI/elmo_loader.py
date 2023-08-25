import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from elmo_bias import *


# the elmo loader
#	it takes no input but the current example idx
#	encodings are actually loaded from cached embeddings
class ElmoLoader(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ElmoLoader, self).__init__()
		self.opt = opt
		self.shared = shared
		self.null_token = torch.zeros(opt.elmo_in_size).float()
		if opt.gpuid != -1:
			self.null_token = self.null_token.cuda()
		#self.null_token.skip_init = 1
		#self.null_token.requires_grad = True

		if opt.debias == 1 and opt.bias_elmo != '':
			self.elmo_bias = ElmoBias(opt, shared)


	# fetch a specific layer of elmo, 0/1/2
	def get_layer(self, idx):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2

		elmo1 = self.shared.res_map['elmo_src']
		elmo2 = self.shared.res_map['elmo_tgt']

		# if to debias
		if self.opt.debias == 1 and self.opt.bias_elmo != '':
			elmo1 = self.elmo_bias(elmo1)
			elmo2 = self.elmo_bias(elmo2)

		sent1 = torch.zeros(batch_l, sent_l1, self.opt.elmo_in_size)
		sent2 = torch.zeros(batch_l, sent_l2, self.opt.elmo_in_size)
		if self.opt.gpuid != -1:
			sent1 = sent1.cuda()
			sent2 = sent2.cuda()

		start = self.opt.elmo_in_size * idx
		end = self.opt.elmo_in_size * (idx+1)

		for i in range(len(elmo1)):
			assert(elmo1[i].shape[0] == sent_l1-1)
			sent1[i, 0, :] = self.null_token
			sent1[i, 1:, :] = elmo1[i][:, start:end]

		for i in range(len(elmo2)):
			assert(elmo2[i].shape[0] == sent_l2-1)
			sent2[i, 0, :] = self.null_token
			sent2[i, 1:, :] = elmo2[i][:, start:end]

		sent1 = Variable(sent1, requires_grad=False)
		sent2 = Variable(sent2, requires_grad=False)

		return sent1, sent2



	# load cached ELMo embeddings for the current batch
	def forward(self):
		sent1_l0, sent2_l0 = self.get_layer(0)
		sent1_l1, sent2_l1 = self.get_layer(1)
		sent1_l2, sent2_l2 = self.get_layer(2)

		return [[sent1_l0, sent1_l1, sent1_l2], [sent2_l0, sent2_l1, sent2_l2]]


	def begin_pass(self):
		pass

	def end_pass(self):
		pass

