import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *


# Multiclass Loss
class MulticlassLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(MulticlassLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.num_correct = 0
		self.num_ex = 0
		self.verbose = False
		# NOTE, do not creat loss node globally

		self.all_preds = []
		

	def forward(self, pred, gold):
		log_p = pred
		batch_l = self.shared.batch_l
		assert(pred.shape == (batch_l, self.opt.num_label))

		# loss
		crit = torch.nn.NLLLoss(reduction='sum')	# for pytorch < 0.4.1, use size_average=False
		if self.opt.gpuid != -1:
			crit = crit.cuda()
		loss = crit(log_p, gold[:])

		# stats
		self.num_correct += np.equal(pick_label(log_p.data.cpu()), gold.cpu()).sum()
		self.num_ex += batch_l

		if hasattr(self.opt, 'pred_output'):
			preds = pick_label(log_p.data.cpu())
			for k, (ex_idx, p) in enumerate(zip(self.shared.batch_ex_idx, preds)):
				self.all_preds.append([ex_idx, p, gold.cpu()[k]])


		return loss


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Acc {0:.3f} '.format(float(self.num_correct) / self.num_ex)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		acc = float(self.num_correct) / self.num_ex
		return acc, [acc] 	# and any other scalar metrics	


	def begin_pass(self):
		# clear stats
		self.num_correct = 0
		self.num_ex = 0
		self.all_preds = []

	def end_pass(self):
		if hasattr(self.opt, 'pred_output'):
			pred_path = self.opt.pred_output + '.pred.txt'
			print('writing predictions to {0}'.format(pred_path))
			with open(pred_path, 'w') as f:
				for idx, p, g in self.all_preds:
					f.write('{0}\t{1}\t{2}\n'.format(idx, p, g))

