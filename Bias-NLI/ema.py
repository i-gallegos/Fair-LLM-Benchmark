import torch
from torch import nn

class EMA():
    def __init__(self, opt, shared):
        self.mu = opt.mu
        self.avg = {}    # keeps a copy of parameter averages

    def step(self, m):
        # recompute the averages
        for n, p in m.named_parameters():
            if p.requires_grad:
                if n not in self.avg:
                    # intialize with the model itself
                    self.avg[n] = torch.Tensor().type_as(p.data).resize_as_(p.data).zero_()
                    self.avg[n].copy_(p.data)
    
                new_avg = (1.0 - self.mu) * p.data + self.mu * self.avg[n]
                self.avg[n].copy_(new_avg)
        
        # copy to model
        #for n, p in m.named_parameters():
        #    if p.requires_grad:
        #        p.data.copy_(self.avg[n])

    def get_param_dict(self):
        param_dict = {}
        for n, p in self.avg.items():
            param_dict[n] = p.cpu().numpy()
        return param_dict
        