import torch
from torch import nn
from torch.autograd import Variable

# liner classifier
class LinearClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LinearClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		# weights will be initialized later
		self.linear = nn.Sequential(
			nn.Dropout(opt.dropout),	# no dropout here according to huggingface
			nn.Linear(opt.hidden_size, 2))	# 1 for start, 1 for end


	def forward(self, concated):
		batch_l, concated_l, enc_size = concated.shape

		scores = self.linear(concated.view(-1, enc_size)).view(batch_l, concated_l, 2)

		# ugly hack

		log_p = nn.LogSoftmax(1)(scores)

		log_p1 = log_p[:, :, 0]
		log_p2 = log_p[:, :, 1]

		self.shared.y_scores = scores

		return [log_p1, log_p2]


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		
