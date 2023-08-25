import sys
import os
import math
import torch
from torch import nn
from utils.util import *
from apex import amp
from transformers.optimization import *
#from apex.fp16_utils import FP16_Optimizer
from transformers import AdamW


# the huggingface's adam for bert
class AdamBert:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.betas = [float(b) for b in opt.adam_betas.split(',')]

	def build_optimizer(self, m):
		params = [p for n, p in m.named_parameters() if p.requires_grad]
		self.optim = BertAdam(params, lr=self.opt.learning_rate, max_grad_norm=self.opt.clip, b1=self.betas[0], b2=self.betas[1])
		return m

	def backward(self, m, loss):
		loss.backward()
		params = [p for n, p in m.named_parameters() if p.requires_grad]

	def step(self, m):
		grad_norm2 = nn.utils.clip_grad_norm_(params, self.clip, norm_type=2)

		self.optim.step()

		for n, p in m.named_parameters():
			if has_nan(p.data):
				print(n, p.data)
				assert(False)

		return grad_norm2


# the apex's adam for fp16 with huggingface AdamW
class AdamWFp16:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.scheduler = None
		
	def build_optimizer(self, m):
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		optimizer_grouped_parameters = [{'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.opt.weight_decay},
			{'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		adamw = AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate)
		m, self.optim = amp.initialize(m, adamw, opt_level='O1')
		return m

	def get_lr(self):
		if self.opt.warmup_perc <= 0:
			return self.opt.learning_rate
		acc_l = 20
		normalizer = self.shared.num_train_ex / acc_l * self.opt.epochs
		if self.opt.warmup_type == 'linear_flat':
			return self.opt.learning_rate * warmup_linear_flat(self.shared.num_update / normalizer, self.opt.warmup_perc)
		elif self.opt.warmup_type == 'linear':
			return self.opt.learning_rate * warmup_linear(self.shared.num_update / normalizer, self.opt.warmup_perc)
		else:
			raise Exception('unrecognized warmup_type: {0}'.format(self.opt.warmup_type))

	def step(self, m):
		cur_lr = self.get_lr()
		for param_group in self.optim.param_groups:
			param_group['lr'] = cur_lr

		grad_norm2 = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.opt.clip)

		self.optim.step()
		return grad_norm2

	# this interface is only for apex's optimizer
	def backward(self, m, loss):
		with amp.scale_loss(loss, self.optim) as scaled_loss:
			scaled_loss.backward()



def get_optimizer(opt, shared):
	optim = None
	if opt.optim == 'adam_bert':
		optim = AdamBert(opt, shared)
	elif opt.optim == 'adamw_fp16':
		optim = AdamWFp16(opt, shared)
	else:
		print('unrecognized optim: {0}'.format(opt.optim))
		assert(False)
	return optim


def grad_sanity_check(optim, m, batch_size):
	optim.__SANITY_FLAG = False
	for n, p in m.named_parameters():
		if p.requires_grad:
			if p.grad is None:
				if optim.__SANITY_FLAG == False:
					print('{0} requires grad but has no grad, double check your graph'.format(n))
			else:
				if p.grad.is_sparse:
					print('sparse gradient found.')
					assert(False)
				p.grad.data.div_(batch_size)

	optim.__SANITY_FLAG = True


