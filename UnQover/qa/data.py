import io
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from utils.util import *

class Data():
	def __init__(self, opt, data_file, res_files=None):
		self.opt = opt
		self.data_name = data_file


		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')

		self.concated = f['concated'][:]	# indices to glove tokens
		self.concated_l = f['concated_l'][:].astype(np.int32)
		self.context_l = f['context_l'][:].astype(np.int32)
		self.query_l = f['query_l'][:].astype(np.int32)
		self.concated_span = f['concated_span'][:]
		self.context_start = f['context_start'][:]
		self.batch_l = f['batch_l'][:].astype(np.int32)
		self.batch_idx = f['batch_idx'][:].astype(np.int32)
		self.ex_idx = f['ex_idx'][:].astype(np.int32)
		self.length = self.batch_l.shape[0]


		self.concated = torch.from_numpy(self.concated)
		self.concated_span = torch.from_numpy(self.concated_span)

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]
			concated_l_i = self.concated_l[i]

			context_l_i = self.context_l[start:end]
			query_l_i = self.query_l[start:end]
			concated_i = self.concated[start:end, :concated_l_i]
			concated_span_i = self.concated_span[start:end]
			context_start_i = self.context_start[start:end]


			self.batches.append((concated_i, int(self.batch_l[i]), concated_l_i, context_l_i, query_l_i,
				concated_span_i, context_start_i))

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

				elif f.endswith('json'):
					res_names = self.__load_json_res(f)

				else:
					assert(False)
				self.res_names.extend(res_names)


	def subsample(self, ratio, random=True, filter_idx=None):
		target_num_ex = int(float(self.num_ex) * ratio)
		sub_idx = [int(idx) for idx in torch.randperm(self.size())] if random else [i for i in range(self.size())]
		if filter_idx is not None:
			sub_idx = [k for k in sub_idx if k in filter_idx]
		cur_num_ex = 0
		i = 0
		while cur_num_ex < target_num_ex and i < len(sub_idx):
			cur_num_ex += self.batch_l[sub_idx[i]]
			i += 1
		return sub_idx[:i], cur_num_ex


	def split(self, sub_idx, ratio):
		num_ex = sum([self.batch_l[i] for i in sub_idx])
		target_num_ex = int(float(num_ex) * ratio)

		cur_num_ex = 0
		cur_pos = 0
		for i in range(len(sub_idx)):
			cur_pos = i
			cur_num_ex += self.batch_l[sub_idx[i]]
			if cur_num_ex >= target_num_ex:
				break

		return sub_idx[:cur_pos+1], sub_idx[cur_pos+1:], cur_num_ex, num_ex - cur_num_ex


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
		if res_name == 'raw_answer' or res_name == 'tok_answer':
			print('customized parsing...')
			for l in res_data:
				parsed.append(l.rstrip().split('|||'))	# a list of strings that are all ground truth answers
		elif res_name == 'context' or res_name == 'query' or res_name == 'context_tok_map':
			print('customized parsing...')
			for l in res_data:
				parsed.append(l.rstrip().split(' '))
		elif res_name == 'context_pos' or res_name == 'context_ner':
			print('customized parsing...')
			for l in res_data:
				parsed.append(l.strip().split(' '))
		else:
			parsed = res_data

		setattr(self, res_name, parsed)
		return [res_name]


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
					lut[int(i)] = ([l for l in j[0]], [l for l in j[1]])

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
			res[int(k)] = ([l for l in p], [l for l in h])
		
		setattr(self, res_name, res)
		return res_name


	def size(self):
		return self.length


	def __getitem__(self, idx):
		(concated_i, batch_l, concated_l_i, context_l_i, query_l_i,
			concated_span_i, context_start_i) = self.batches[idx]

		# get char indices
		# 	the back forth data transfer should be eliminated
		#char_concated = self.char_idx[all_concated_i.contiguous().view(-1)].view(batch_l, concated_l_i, token_l)

		# transfer to gpu if needed
		if self.opt.gpuid != -1:
			#char_concated = char_concated.cuda()
			#concated_i = concated_i.cuda()
			concated_i = concated_i.long().cuda()
			concated_span_i = concated_span_i.cuda()

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		res_map = self.__get_res(idx)

		return (self.data_name, batch_ex_idx, concated_i, batch_l, concated_l_i, context_l_i, query_l_i,
				concated_span_i, context_start_i, res_map)


	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None

		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			# some customization for elmo is needed here for lazy loading
			res = getattr(self, res_n)

			batch_res = [res[ex_id] for ex_id in batch_ex_idx]
			all_res[res_n] = batch_res
		return all_res


	# something at the beginning of each pass of training/eval
	#	e.g. setup preloading
	def begin_pass(self):
		pass


	def end_pass(self):
		pass
