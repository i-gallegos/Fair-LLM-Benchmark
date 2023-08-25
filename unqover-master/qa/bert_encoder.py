import sys
import os
import torch
from torch import cuda
from transformers import *
from torch import nn
from torch.autograd import Variable
from utils.holder import *

# encoder with Elmo
class BertEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BertEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.zero = Variable(torch.zeros(1).half(), requires_grad=False)

		print('loading Transformer...')
		self.transformer, self.tokenizer = self._get_transformer(self.opt.transformer_type)

		print('verifying Transformer...')
		self.transformer.eval()

		if opt.gpuid != -1:
			self.zero = self.zero.cuda()

		for n in self.transformer.children():
			for p in n.parameters():
				p.skip_init = True

		self.customize_cuda_id = self.opt.gpuid
		#self.shared.tokenizer = self.tokenizer

	def _get_transformer(self, key):
		return AutoModel.from_pretrained(key), AutoTokenizer.from_pretrained(key)

	def forward(self, concated, att_mask=None):

		if 'distil' not in self.opt.transformer_type:
			enc, pooled = self.transformer(concated, att_mask)
			enc = enc + pooled.unsqueeze(1) * self.zero 	#hacky a bit
		else:
			rs = self.transformer(concated, att_mask)
			enc = rs[0]

		return enc


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


