import sys
import torch
from torch import nn
from torch.autograd import Variable
from utils.util import *
import math

# Boundary Loss
class BoundaryLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BoundaryLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		# do not creat loss node globally
		self.idx1_correct = 0
		self.idx2_correct = 0
		self.total_em_bow = 0.0
		self.total_f1_bow = 0.0
		self.num_ex = 0
		self.verbose = False
		self.start_off_cnt = {}
		self.end_off_cnt = {}

		self.all_query_id = []
		self.all_pred = []
		self.all_accuracy = []
		self.all_gold = []
		self.all_start = []
		self.all_end = []
		self.all_gold_start_p = []
		self.all_gold_end_p = []
		self.all_raw_context = []
		self.all_raw_query = []


	def forward(self, pred, gold, lambd):
		assert(lambd.numel() == 1)
		log_p1, log_p2 = pred
		# loss
		crit = torch.nn.NLLLoss(reduction='sum')	# for pytorch < 0.4.1, use size_average=False
		if self.opt.gpuid != -1:
			crit = crit.cuda()
		loss1 = crit(log_p1, gold[:,0])	# loss on start idx
		loss2 = crit(log_p2, gold[:,1])	# loss on end idx
		loss = (loss1 + loss2) * lambd

		# stats
		ans_span = pick_best_span_bounded(log_p1.data.cpu().float(), log_p2.data.cpu().float(), self.opt.span_l)
		idx1, idx2 = ans_span[:,0], ans_span[:,1]
		self.idx1_correct += count_correct_idx(idx1, gold[:,0].data.cpu())
		self.idx2_correct += count_correct_idx(idx2, gold[:,1].data.cpu())
		self.num_ex += self.shared.batch_l

		# shift context if necessary
		query_l = torch.from_numpy(self.shared.query_l).long()
		context_start = torch.from_numpy(self.shared.context_start).long()
		context_idx1 = idx1 - query_l - 2	# -2 because of [CLS] and [SEP]
		context_idx2 = idx2 - query_l - 2
		context_idx1 = context_idx1.clamp(0) + context_start
		context_idx2 = context_idx2.clamp(0) + context_start

		# f1 and em
		pred_ans = get_answer_tokenized(context_idx1, context_idx2, self.shared.res_map['context'])
		gold_ans = self.shared.res_map['tok_answer']
		gold_ans = [[cleanup_G(k) for k in p] for p in gold_ans]
		em_bow = get_em_bow(pred_ans, gold_ans)
		f1_bow = get_f1_bow(pred_ans, gold_ans)
		self.total_em_bow += sum(em_bow)
		self.total_f1_bow += sum(f1_bow)

		# if to output official format of predictions (i.e. id: ans)
		self.all_query_id.extend(self.shared.res_map['query_id'])
		self.all_gold.extend([a[0] for a in gold_ans])
		if self.opt.output_official != '':
			self.all_pred.extend(pred_ans)
			self.all_accuracy.extend(get_contain_bow(pred_ans, gold_ans))
			self.all_start.extend(idx1)
			self.all_end.extend(idx2)
			self.all_gold_start_p.extend(log_p1.data.gather(1, gold[:,0].view(self.shared.batch_l,1)).view(self.shared.batch_l).float().cpu().exp())
			self.all_gold_end_p.extend(log_p2.data.gather(1, gold[:,1].view(self.shared.batch_l,1)).view(self.shared.batch_l).float().cpu().exp())
			self.all_raw_context.extend(self.shared.res_map['raw_context'])
			self.all_raw_query.extend(self.shared.res_map['raw_query'])

		# verbose
		if self.verbose:
			#raw_query = self.shared.res_map['raw_query']
			k = 0
			print('*************************** pred gold')
			for p, g, em, f1 in zip(pred_ans, gold_ans, em_bow, f1_bow):
				if f1 != 1:
					print(u'{0} {1} {2:.4f} {3:.4f}'.format(p, g, em, f1).encode('utf-8'))
				k += 1

		return loss


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Span {0:.3f}/{1:.3f} '.format(
			float(self.idx1_correct) / self.num_ex, float(self.idx2_correct) / self.num_ex)
		stats += 'EM {0:.3f} F1 {1:.3f} '.format(
			self.total_em_bow / self.num_ex,
			self.total_f1_bow / self.num_ex,)
		return stats


	# get training metric (scalar metric, extra metric)
	#	the scalar metric will be used to pick the best model
	#	the extra metric a list of scalars for extra info
	def get_epoch_metric(self):
		acc1 = float(self.idx1_correct) / self.num_ex
		acc2 = float(self.idx2_correct) / self.num_ex
		em_bow = float(self.total_em_bow) / self.num_ex
		f1_bow = self.total_f1_bow / self.num_ex

		if self.opt.output_official != '':
			print('writting to {0}'.format(self.opt.output_official))
			with open(self.opt.output_official, 'w') as f:
				f.write('{\n')
				for k, (q_id, a, g, accuracy, s, e, ps, pe, c, q) in enumerate(zip(self.all_query_id, self.all_pred, self.all_gold, self.all_accuracy, self.all_start, self.all_end, self.all_gold_start_p, self.all_gold_end_p, self.all_raw_context, self.all_raw_query)):
					a = a.replace('"', '\\"')
					f.write('"{0}": {{"ans":"{1}", "gold": "{2}", "accuracy": "{3:.4f}", "start": "{4}", "end": "{5}", "p_start": "{6}", "p_end": "{7}", "context": "{8}", "query": "{9}"}}'.format(q_id, a, g, accuracy, s, e, ps, pe, c, q))
					if k != len(self.all_pred)-1:
						f.write(',\n')
				f.write('\n}\n')

			# count the grouped consistency
			grouped_consistency = {}
			for k, (q_id, consistency) in enumerate(zip(self.all_query_id, self.all_accuracy)):
				q_id = q_id.split('-')[0]
				if q_id not in grouped_consistency:
					grouped_consistency[q_id] = []
				grouped_consistency[q_id].append(consistency)

			consistent_ls = [1.0 for q_id, consistency_ls in grouped_consistency.items() if sum(consistency_ls) == len(consistency_ls)]

			print('general consistency: {0:.4f}'.format(sum(self.all_accuracy)/len(self.all_accuracy)))
			print('grouped consistency: {0:.4f}'.format(sum(consistent_ls)/len(grouped_consistency)))

		return f1_bow, [(acc1 + acc2) / 2.0, em_bow, f1_bow]


	def begin_pass(self):
		# clear stats
		self.idx1_correct = 0
		self.idx2_correct = 0
		self.num_ex = 0
		self.total_em_bow = 0.0
		self.total_f1_bow = 0.0

		self.all_query_id = []
		self.all_pred = []
		self.all_accuracy = []
		self.all_gold = []
		self.all_start = []
		self.all_end = []
		self.all_ans_cand = []
		self.all_gold_span = []
		self.cand_recall_cnt = 0

	def end_pass(self):
		pass

