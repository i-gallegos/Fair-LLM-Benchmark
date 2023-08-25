import sys
sys.path.append('../allennlp')
sys.path.append('./elmo')
import argparse
import h5py
import torch
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from elmo_embedder_debias import *

def load_elmo(opt):
	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

	bias = None
	if opt.bias != '':
		f = h5py.File(opt.bias, 'r')
		bias = torch.from_numpy(f['bias'][:].astype(np.float32)).view(1, 1, -1)
		bias = bias[:, :, 0:512]

	contraction = None
	if opt.contract_v1 != '' and opt.contract_v2 != '':
		f = h5py.File(opt.contract_v1, 'r')
		contract_v1 = torch.from_numpy(f['bias'][:].astype(np.float32)).view(1, 1, -1)
		f = h5py.File(opt.contract_v2, 'r')
		contract_v2 = torch.from_numpy(f['bias'][:].astype(np.float32)).view(1, 1, -1)
		contraction = (contract_v1[:, :, :512], contract_v2[:, :, :512])

	if bias is None and contraction is None:
		elmo = ElmoEmbedder(options_file, weight_file, cuda_device=opt.gpuid)	# by default all 3 layers are output
	else:
		elmo = ElmoEmbedderDebias(bias, opt.num_bias, contraction, options_file, weight_file, cuda_device=opt.gpuid)

	return elmo


def load_token(path):
	tokens = []
	with open(path, 'r+') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			tokens.append(l.strip().split(' '))
	return tokens


def elmo_over(opt, elmo, toks):
	emb_ls = elmo.embed_batch(toks)
	return emb_ls	# each element has shape (3, seq_l, 1024)


def load_batched(opt):
	f = h5py.File(opt.batched, 'r')
	source_l = f['source_l'][:].astype(np.int32)	# (batch_l,)
	target_l = f['target_l'][:].astype(np.int32)	# (batch_l,)
	batch_l = f['batch_l'][:].astype(np.int32)
	batch_idx = f['batch_idx'][:].astype(np.int32)
	ex_idx = f['ex_idx'][:].astype(np.int32)

	return ex_idx, batch_idx, batch_l, source_l, target_l


def load_elmo_unbatched(opt):
	f = h5py.File(opt.elmo_unbatched, 'r')
	return f


def process(opt, src, tgt, batched, elmo, output):
	assert(len(src) == len(tgt))

	ex_idx, batch_idx, batch_l, source_l, target_l = batched

	f = h5py.File(output, 'w')

	print_every = 100
	batch_cnt = 0
	num_batch = batch_l.shape[0]
	print('processing {0} batches...'.format(num_batch))

	for i in range(num_batch):
		start = batch_idx[i]
		end = start + batch_l[i]

		batch_ex_idx = [ex_idx[k] for k in range(batch_idx[i], batch_idx[i] + batch_l[i])]

		batch_src = [src[k] for k in batch_ex_idx]
		batch_tgt = [tgt[k] for k in batch_ex_idx]

		elmo_ls1 = elmo_over(opt, elmo, batch_src)
		elmo_ls2 = elmo_over(opt, elmo, batch_tgt)

		seq_l1 = source_l[i]-1
		seq_l2 = target_l[i]-1

		# sanity check, sentences within a batch are supposed to have the same length
		assert(len(batch_src[0]) == len(batch_src[-1]))
		assert(len(batch_src[0]) == seq_l1)
		assert(len(batch_tgt[0]) == seq_l2)

		batch_elmo1 = torch.zeros(batch_l[i], seq_l1, 3072)
		for k, e in enumerate(elmo_ls1):
			e = torch.from_numpy(e).transpose(1, 0).contiguous()	# (seq_l, 3, 1024)
			batch_elmo1[k] = e.view(seq_l1, 3072)

		batch_elmo2 = torch.zeros(batch_l[i], seq_l2, 3072)
		for k, e in enumerate(elmo_ls2):
			e = torch.from_numpy(e).transpose(1, 0).contiguous()	# (seq_l, 3, 1024)
			batch_elmo2[k] = e.view(seq_l2, 3072)

		f['{0}.src_batch'.format(i)] = batch_elmo1.numpy().astype(np.float32)
		f['{0}.tgt_batch'.format(i)] = batch_elmo2.numpy().astype(np.float32)

		batch_cnt += 1
		if batch_cnt % print_every == 0:
			print('processed {0} batches'.format(batch_cnt))

	f.close()


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpuid', help="The gpuid", type=int, default=-1)
	parser.add_argument('--src', help="Path to the tokenized premise", default="data/nli_bias/dev.sent1.txt")
	parser.add_argument('--tgt', help="Path to the tokenized hypothesis", default="data/nli_bias/dev.sent2.txt")
	parser.add_argument('--bias', help="Path to the elmo bias file, (optional)", default="")
	parser.add_argument('--num_bias', help="The number of bias directions", type=int, default=1)
	parser.add_argument('--lowercase', help="Whether to embed input in lower case", type=int, default=1)
	parser.add_argument('--contract_v1', help="Path to the elmo contraction dir 1, (optional)", default="")
	parser.add_argument('--contract_v2', help="Path to the elmo contraction dir 2, (optional)", default="")
	parser.add_argument('--batched', help="The batched hdf5 file from preprocess.py", default='data/nli_bias/snli-val.hdf5')
	parser.add_argument('--output', help="Prefix of output files", default="data/nli_bias/dev")
	opt = parser.parse_args(arguments)

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
	
	
	elmo = load_elmo(opt)
	src = load_token(opt.src)
	tgt = load_token(opt.tgt)

	if opt.lowercase == 1:
		src = [[t.lower() for t in s] for s in src]
		tgt = [[t.lower() for t in s] for s in tgt]

	batched = load_batched(opt)
	process(opt, src, tgt, batched, elmo, opt.output+'.elmo.hdf5')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

