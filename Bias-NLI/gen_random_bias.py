import os
import sys
import argparse
import numpy as np
from sklearn.decomposition import PCA
import torch
from random import gauss

def gen_gaussian_bias(dim):
	vec = [gauss(0, 1) for i in range(dim)]
	mag = sum(x**2 for x in vec) ** .5
	return [x/mag for x in vec]


def gen_random_bias(word_vecs):
	word_vec_size = word_vecs[0].size

	p = PCA(n_components = 20)
	idx = np.random.permutation(len(word_vecs))[0:20]

	A = []
	for i in idx:
		A.append(torch.from_numpy(word_vecs[i]).view(1, word_vec_size))
	A = torch.cat(A, 0).numpy()

	p.fit(A)
	V = p.components_
	return V[0]


def load_glove(fname):
    dim = 0
    word_vecs = []
    word_vec_size = None
    for line in open(fname, 'r'):
        d = line.split()

        # get info from the first word
        if word_vec_size is None:
          word_vec_size = len(d) - 1

        word = ' '.join(d[:len(d)-word_vec_size])
        vec = d[-word_vec_size:]
        vec = np.array(list(map(float, vec)))

        if len(d) - word_vec_size != 1:
          print('multi word token found: {0}'.format(line))

        word_vecs.append(vec)
    return word_vecs


def write_bias(path, bias):
	with open(path, 'w') as f:
		for b in bias:
			log = ' '.join([str(v) for v in b])
			f.write(log + '\n')


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--glove', help="Path to the glove file", default='./data/glove.840B.300d.txt')
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "data/nli_bias/random")
	parser.add_argument('--num_bias', help="The number of bias vectors to generate", type=int, default=1)
	parser.add_argument('--num_file', help="The number of bias files to generate", type=int, default=10)
	parser.add_argument('--type', help="The type of random bias, random/gaussian", default='random')
	parser.add_argument('--dim', help="The dim of gaussian vector", type=int, default=300)
	parser.add_argument('--seed', help="The random seed", type=int, default=1)
	opt = parser.parse_args(arguments)

	np.random.seed(opt.seed)

	print('loading glove from {0}'.format(opt.glove))
	glove = load_glove(opt.glove)

	for i in range(opt.num_file):
		print('generating random bias...')
		bias = []
		for k in range(opt.num_bias):
			if opt.type == 'random':
				bias.append(gen_random_bias(glove))
			elif opt.type == 'gaussian':
				bias.append(gen_gaussian_bias(opt.dim))
	
		output_path = opt.output + '{0}.txt'.format(i+1)
		print('writing random bias to {0}'.format(output_path))
		write_bias(output_path, bias)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))