import torch
import torch.nn as nn
from torch.autograd import Variable

# apply dropout on sequence of encodings
#	assuming input shape (batch_l, seq_l, hidden_size)
# dropout is applied the same across timestep (i.e. wipeout certain dimensions)
class LockedDropout(nn.Module):
    def __init__(self, p):
        super(LockedDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        m = x.data.new(x.shape[0], 1, x.shape[2]).bernoulli_(1.0 - self.p)
        mask = Variable(m / (1 - self.p), requires_grad=False)
        mask = mask.expand_as(x)

        if x.is_cuda:
        	mask = mask.cuda()

        return mask * x