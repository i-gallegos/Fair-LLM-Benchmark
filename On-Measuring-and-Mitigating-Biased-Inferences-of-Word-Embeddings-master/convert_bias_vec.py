import os
import sys
import argparse
import numpy as np
import h5py


def load_vec(path):
	with open(path, 'r') as f:
		vals = []
		for l in f:
			if l.rstrip() == '':
				continue
			vals.append([float(k) for k in l.rstrip().split()])
		return np.asarray(vals)


def write_hdf5(path, map):
	print('writing data to {0}'.format(path))
	f = h5py.File(path, "w")
	for key, val in map.items():
		f[key] = val



def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input', help="Path to the vec text file, one value per line", default='')
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "data/nli_bias/bias")
	opt = parser.parse_args(arguments)

	vec = load_vec(opt.input)
	m = {'bias': vec}
	write_hdf5(opt.output + '.hdf5', m)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))