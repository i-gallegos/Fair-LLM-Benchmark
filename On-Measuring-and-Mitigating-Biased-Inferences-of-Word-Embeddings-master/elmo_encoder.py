
import sys
sys.path.append('../allennlp')
sys.path.append('./elmo')
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from locked_dropout import *
from allennlp.modules.elmo import _ElmoCharacterEncoder as ElmoEmbedder
from elmo_bias import *
from elmo_embedder_debias import *

# the dynamic ELMo encoder, instead of loading cached ELMo, here use pretrained ELMo to scan over current batch on the fly.
class ElmoEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ElmoEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.null_token = torch.zeros(opt.elmo_in_size).float()
		if opt.gpuid != -1:
			self.null_token = self.null_token.cuda()

		# initialize from these
		options_file = None
		weight_file = None
		if opt.elmo_in_size == 1024:
			options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
			weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		else:
			raise Exception("unsupported elmo_in_size {0}".format(opt.elmo_in_size))
		
		if opt.debias == 1:
			bias = None
			contraction = None
			if opt.bias_elmo != opt.dir:
				self.elmo_bias = ElmoBias(opt, shared)
				bias = self.elmo_bias.bias_elmo[:, :, 0:512]
			if opt.contract_v1 != opt.dir:
				f = h5py.File(opt.contract_v1, 'r')
				self.contract_v1 = torch.from_numpy(f['bias'][:].astype(np.float32)).view(1, 1, -1)[:,:,:512]
				f = h5py.File(opt.contract_v2, 'r')
				self.contract_v2 = torch.from_numpy(f['bias'][:].astype(np.float32)).view(1, 1, -1)[:,:,:512]
				contraction = (self.contract_v1, self.contract_v2)
			self.elmo = ElmoEmbedderDebias(bias, opt.num_bias, contraction, options_file, weight_file, cuda_device=opt.gpuid)
		else:
			self.elmo = ElmoEmbedder(options_file, weight_file, cuda_device=opt.gpuid)	# by default all 3 layers are output


	def elmo_over(self, toks):
		emb_ls = self.elmo.embed_batch(toks)	# each element has shape (3, seq_l, 1024)
		emb = torch.cat([torch.from_numpy(e).transpose(0,1).unsqueeze(0) for e in emb_ls], 0)	# (batch_l, seq_l, 3, 1024)
		if self.opt.gpuid != -1:
			emb = emb.cuda()
		return emb


	# fetch a specific layer of elmo, 0/1/2
	def get_layer(self, elmo1, elmo2, idx):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2

		# if to debias
		#if hasattr(self, 'elmo_bias'):
		#	elmo1 = self.elmo_bias(elmo1)
		#	elmo2 = self.elmo_bias(elmo2)

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


	def forward(self):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2

		sent1 = self.shared.res_map['sent1']
		sent2 = self.shared.res_map['sent2']

		assert(batch_l == len(sent1) and batch_l == len(sent2))

		elmo1 = self.elmo_over(sent1)	# (batch_l, seq_l, 3, 1024)
		elmo2 = self.elmo_over(sent2)

		elmo1 = elmo1.view(batch_l, sent_l1-1, 3072)	# (batch_l, seq_l, 3072)
		elmo2 = elmo2.view(batch_l, sent_l2-1, 3072)

		sent1_l0, sent2_l0 = self.get_layer(elmo1, elmo2, 0)
		sent1_l1, sent2_l1 = self.get_layer(elmo1, elmo2, 1)
		sent1_l2, sent2_l2 = self.get_layer(elmo1, elmo2, 2)

		return [[sent1_l0, sent1_l1, sent1_l2], [sent2_l0, sent2_l1, sent2_l2]]
