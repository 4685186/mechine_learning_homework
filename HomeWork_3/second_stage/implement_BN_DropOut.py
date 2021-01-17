# Coding: UTF-8 
# Created by 11 at 2021/1/16
# This "implement_BN_DropOut.py" will implement function about: 实现BN层与dropout层

import torch
from torch import nn


class mBN2d(nn.Module):
    def __init__(self, num_features):
        super(mBN2d, self).__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.ones(shape))

    def forward(self, x):
        return self.batch_norm(x)

    def batch_norm(self, x, eps=1e-5, momentum=0.9):
        if not self.training:
            x_new = (x - self.moving_mean) / torch.sqrt(self.moving_var + eps)
        else:
            mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            self.moving_mean = momentum * self.moving_mean + (1.0 - momentum) * mean
            self.moving_var = momentum * self.moving_var + (1.0 - momentum) * var
            x_new = (x - mean) / torch.sqrt(var + eps)
        x = self.gamma * x_new + self.beta
        return x


class mDropOut1d(nn.Module):
    def __init__(self, p):
        super(mDropOut1d, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        else:
            judge_Tensor = torch.rand_like(x) > self.p
            return judge_Tensor.float() * x / (1-self.p)
