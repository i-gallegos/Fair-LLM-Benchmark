import io
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from util import *

class Data():
	def __init__(self, opt, data_file, res_files=None):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.source = f['source'][:]	# indices to glove tokens
		self.target = f['target'][:]
		self.all_source = f['all_source'][:]	# indices to all tokens
		self.all_target = f['all_target'][:]
		self.source_l = f['source_l'][:].astype(np.int32)	# (batch_l,)
		self.target_l = f['target_l'][:].astype(np.int32)	# (batch_l,)
		self.label = f['label'][:]
		self.batch_l = f['batch_l'][:].astype(np.int32)
		self.batch_idx = f['batch_idx'][:].astype(np.int32)
		self.ex_idx = f['ex_idx'][:].astype(np.int32)
		self.length = self.batch_l.shape[0]

		self.all_source = torch.from_numpy(self.all_source)
		self.all_target = torch.from_numpy(self.all_target)
		self.source = torch.from_numpy(self.source)
		self.target = torch.from_numpy(self.target)
		self.label = torch.from_numpy(self.label)

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]

			# get example token indices
			all_source_i = self.all_source[start:end, 0:self.source_l[i]]
			all_target_i = self.all_target[start:end, 0:self.target_l[i]]
			source_i = self.source[start:end, 0:self.source_l[i]]
			target_i = self.target[start:end, 0:self.target_l[i]]
			label_i = self.label[start:end]

			# sanity check
			assert(self.source[start:end, self.source_l[i]:].sum() == 0)
			assert(self.target[start:end, self.target_l[i]:].sum() == 0)

			# src, tgt, all_src, all_tgt, batch_l, src_l, tgt_l, label, raw info
			self.batches.append((source_i, target_i, all_source_i, all_target_i,
				int(self.batch_l[i]), int(self.source_l[i]), int(self.target_l[i]), label_i))

		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]

		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				if f.endswith('txt'):
					res_names = self.__load_txt(f)

				elif f.endswith('elmo.hdf5'):
					res_names = self.__load_elmo(f)

				elif f.endswith('json'):
					res_names = self.__load_json_res(f)

				else:
					assert(False)
				self.res_names.extend(res_names)


	def subsample(self, ratio, minimal_num=0):
		target_num_ex = int(float(self.num_ex) * ratio)
		target_num_ex = max(target_num_ex, minimal_num)
		sub_idx = torch.LongTensor(range(self.size()))
		sub_num_ex = 0

		if ratio != 1.0:
			rand_idx = torch.randperm(self.size())
			i = 0
			while sub_num_ex < target_num_ex and i < self.batch_l.shape[0]:
				sub_num_ex += self.batch_l[rand_idx[i]]
				i += 1
			sub_idx = rand_idx[:i]

		else:
			sub_num_ex = self.batch_l.sum()

		return sub_idx, sub_num_ex


	def __load_txt(self, path):
		lines = []
		print('loading resource from {0}'.format(path))
		# read file in unicode mode!!!
		with io.open(path, 'r+', encoding="utf-8") as f:
			for l in f:
				lines.append(l.rstrip())
		# the second last extension is the res name
		res_name = path.split('.')[-2]
		res_data = lines[:]

		# some customized parsing
		parsed = []
		if res_name == 'sent1' or res_name == 'sent2' or res_name == 'x_pair':
			print('customized parsing...')
			for l in res_data:
				parsed.append(l.split())
		else:
			parsed = res_data

		setattr(self, res_name, parsed)
		return [res_name]


	def __load_elmo(self, path):
		print('loading resources from {0}'.format(path))
		f = h5py.File(path, 'r')
		self.elmo_file = f

		# the attributes will not be assigned to self, instead, they are customized in __get_res
		return ['elmo_src', 'elmo_tgt']


	def __load_json_res(self, path):
		print('loading resource from {0}'.format(path))
		
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 2)
		res_type = next(iter(j_obj))

		res_name = None
		if j_obj[res_type] == 'map':
			res_name = self.__load_json_map(path)
		elif j_obj[res_type] == 'list':
			res_name = self.__load_json_list(path)
		else:
			assert(False)

		return [res_name]

	
	def __load_json_map(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)

		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			lut = {}
			for i, j in v.items():
				if i == res_name:
					lut[res_name] = [int(l) for l in j]
				else:
					# for token indices, shift by 1 to incorporate the nul-token at the beginning
					lut[int(i)] = ([l+1 for l in j[0]], [l+1 for l in j[1]])

			res[int(k)] = lut
		
		setattr(self, res_name, res)
		return res_name


	def __load_json_list(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)
		
		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			p = v['p']
			h = v['h']

			# for token indices, shift by 1 to incorporate the nul-token at the beginning
			res[int(k)] = ([l+1 for l in p], [l+1 for l in h])
		
		setattr(self, res_name, res)
		return res_name



	def size(self):
		return self.length


	def __get_res_elmo(self, res_name, idx, batch_ex_idx):
		if res_name == 'elmo_src':
			embs = torch.from_numpy(self.elmo_file['{0}.src_batch'.format(idx)][:])
			if self.opt.gpuid != -1:
				embs = embs.cuda()
			#embs = [torch.from_numpy(self.elmo_file['{0}.src'.format(i)][:]) for i in batch_ex_idx]
			#if self.opt.gpuid != -1:
			#	embs = [p.cuda() for p in embs]
			return embs
		elif res_name == 'elmo_tgt':
			embs = torch.from_numpy(self.elmo_file['{0}.tgt_batch'.format(idx)][:])
			if self.opt.gpuid != -1:
				embs = embs.cuda()
			#embs = [torch.from_numpy(self.elmo_file['{0}.tgt'.format(i)][:]) for i in batch_ex_idx]
			#if self.opt.gpuid != -1:
			#	embs = [p.cuda() for p in embs]
			return embs
		else:
			raise Exception('unrecognized res {0}'.format(res_name))


	def __getitem__(self, idx):
		(source, target, all_source, all_target, 
			batch_l, source_l, target_l, label) = self.batches[idx]
		token_l = self.opt.token_l

		# transfer to gpu if needed
		if self.opt.gpuid != -1:
			source = source.cuda()
			target = target.cuda()
			label = label.cuda()

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		res_map = self.__get_res(idx)

		return (self.data_name, source, target,
			batch_ex_idx, batch_l, source_l, target_l, label, res_map)


	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None

		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			# some customization for elmo is needed here for lazy loading
			if 'elmo' in res_n:
				batch_res = self.__get_res_elmo(res_n, idx, batch_ex_idx)
				all_res[res_n] = batch_res
			else:
				res = getattr(self, res_n)

				batch_res = [res[ex_id] for ex_id in batch_ex_idx]
				all_res[res_n] = batch_res

		return all_res



if __name__ == '__main__':
	pass