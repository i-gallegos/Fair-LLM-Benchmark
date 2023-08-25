import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

class LocalClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LocalClassifier, self).__init__()

		# bookkeeping
		self.opt = opt
		self.shared = shared

		self.enc_size = opt.hidden_size*2 + opt.elmo_size*2 if opt.use_elmo_post == 1 else opt.hidden_size*2
		self.g = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(self.enc_size, opt.cls_hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.cls_hidden_size, opt.cls_hidden_size),
			nn.ReLU())
		
		self.h = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.cls_hidden_size * 2, opt.cls_hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.cls_hidden_size, opt.cls_hidden_size),
			nn.ReLU(),
			nn.Linear(opt.cls_hidden_size, opt.num_label))


	def forward(self, sent1, sent2, att1, att2):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2
		cls_hidden_size = self.opt.cls_hidden_size

		attended2 = att1.bmm(sent2)
		attended1 = att2.bmm(sent1)

		cat1 = torch.cat([sent1, attended2], 2)
		cat2 = torch.cat([sent2, attended1], 2)

		phi1 = self.g(cat1.view(batch_l * sent_l1, self.enc_size)).view(batch_l, sent_l1, cls_hidden_size)
		phi2 = self.g(cat2.view(batch_l * sent_l2, self.enc_size)).view(batch_l, sent_l2, cls_hidden_size)

		flat_phi1 = phi1.sum(1)
		flat_phi2 = phi2.sum(1)

		phi = torch.cat([flat_phi1, flat_phi2], 1)
		y_scores = self.h(phi)

		# record
		self.shared.y_scores = y_scores
		
		self.shared.out = nn.LogSoftmax(1)(y_scores)

		return self.shared.out


	def begin_pass(self):
		pass

	def end_pass(self):
		pass

if __name__ == '__main__':
	sys.path.insert(0, '../attention/')
	from torch.autograd import Variable
	from local_attention import *

	opt = Holder()
	opt.hidden_size = 3
	opt.dropout = 0.0
	opt.num_label = 3
	shared = Holder()
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8
	shared.input1 = Variable(torch.randn(shared.batch_l, shared.sent_l1, opt.hidden_size), True)
	shared.input2 = Variable(torch.randn(shared.batch_l, shared.sent_l2, opt.hidden_size), True)

	# build network
	attender = LocalAttention(opt, shared)
	classifier = LocalClassifier(opt, shared)

	# update batch info
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	# run network
	shared.att1, shared.att2 = attender(shared.input1, shared.input2)
	shared.out = classifier(shared.input1, shared.input2, shared.att1, shared.att2)

	print(shared.att1)
	print(shared.att1.sum(2))
	print(shared.att2)
	print(shared.att2.sum(2))
	print(shared.out)
	#print(classifier)
	#print(classifier.g[1].weight)
	#print(classifier.g[1].bias)
	#classifier.apply(classifier.weights)
#
	#for i, p in enumerate(classifier.parameters()):
	#	print(p.data)
	#	print(p.grad)
	#