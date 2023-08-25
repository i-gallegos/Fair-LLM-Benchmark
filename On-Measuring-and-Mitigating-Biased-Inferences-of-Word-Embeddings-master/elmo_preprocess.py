import sys
sys.path.append('../allennlp')
import argparse
import h5py
import torch
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

def load_elmo(opt):
	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

		
	elmo = Elmo(options_file, weight_file, 3, dropout=0, requires_grad=False)	# by default all 3 layers are output
	if opt.gpuid != -1:
		elmo = elmo.cuda()
	return elmo


def load_sent(path):
	par = []
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			par.append(split_par(l.rstrip()))
	return par


def load_token(path):
	tokens = []
	with open(path, 'r+') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			tokens.append(l.strip().split(' '))
	return tokens


def split_par(par):
	sents = par.strip().split('|||')
	sents = [s for s in sents if s.strip() != '']
	sents = [s.strip().split(' ') for s in sents]
	return sents


def elmo_over(opt, elmo, toks):
	char_idx = batch_to_ids(toks)

	if opt.gpuid != -1:
		char_idx = char_idx.cuda()

	emb = elmo(char_idx)['elmo_representations']
	return torch.cat([t.data for t in emb], 2)


def process(opt, elmo, src, tgt, output):
	assert(len(src) == len(tgt))

	# output 3 components:
	#	context indices, elmo embeddings for unique contexts, elmo embeddings for queries
	f = h5py.File(output, 'w')

	batch_size = opt.batch_size
	print_every = 100

	print('processing with batch size {0}...'.format(batch_size))

	batch_cnt = 0
	for i in range(0, len(src), batch_size):
		batch_src = src[i:i+batch_size]
		batch_tgt = tgt[i:i+batch_size]

		batch_elmo1 = elmo_over(opt, elmo, batch_src).cpu()
		batch_elmo2 = elmo_over(opt, elmo, batch_tgt).cpu()

		assert(batch_elmo1.shape[0] == len(batch_src))

		for k in range(batch_elmo1.shape[0]):
			f['{0}.src'.format(i + k)] = batch_elmo1[k].numpy()[:len(batch_src[k])]
			f['{0}.tgt'.format(i + k)] = batch_elmo2[k].numpy()[:len(batch_tgt[k])]

		batch_cnt += 1
		if batch_cnt % print_every == 0:
			print('processed {0} examples'.format(batch_cnt * batch_size))
	
	f.close()


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpuid', help="The gpuid", type=int, default=-1)
	parser.add_argument('--batch_size', help="The batch size", type=int, default=40)
	parser.add_argument('--src', help="Path to the tokenized premise", default="data/snli_bias/dev.sent1.txt")
	parser.add_argument('--tgt', help="Path to the tokenized hypothesis", default="data/nli_bias/dev.sent2.txt")
	parser.add_argument('--output', help="Prefix of output files", default="data/snli_bias/dev")
	opt = parser.parse_args(arguments)
	
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
	
	elmo = load_elmo(opt)
	src = load_token(opt.src)
	tgt = load_token(opt.tgt)

	process(opt, elmo, src, tgt, opt.output+'.elmo.hdf5')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

