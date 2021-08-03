# autoregressive flow model (with RNN module)
import math
from math import log, pi, exp
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from scipy import linalg as la

from model.rnn import *
logabs = lambda x: torch.log(torch.abs(x))

    
class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class embedding_linear(nn.Module):
    def __init__(self, in_channel, filter_size=1024):
        super(embedding_linear, self).__init__()
        
        self.net_e = nn.Sequential(
                nn.Linear(in_channel, filter_size),
                GELU(),
                nn.Linear(filter_size, filter_size),
                GELU(),
                nn.Linear(filter_size, in_channel*2)
            )  
        
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)
        
    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight
    
    def forward(self, coup_a, coup_b): 
        
        log, t = self.net_e(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)
        weight = self.calc_weight() 
        
        out_b = (F.linear(coup_b, weight) + t) * s
                        
        logdet = torch.sum(torch.log(s).view(coup_a.shape[0], -1), 1) + torch.sum(self.w_s)

        return out_b, logdet
    
    def reverse(self, coup_a, coup_b):
        weight = self.calc_weight() 
        
        log, t = self.net_e(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)               
        in_b = F.linear(coup_b / s - t, weight.inverse())
        
        return in_b
           

class lstm_linear(nn.Module):
    def __init__(self, in_channel, filter_size=1024):
        super(lstm_linear, self).__init__()
        
        self.net = nn.Sequential(
                nn.Linear(in_channel, filter_size),
                GELU(),
                nn.Linear(filter_size, filter_size),
                GELU(),
                nn.Linear(filter_size, filter_size),
                GELU(),
                nn.Linear(filter_size, in_channel*2)
            )
        
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight
    
    def forward(self, coup_a, coup_b): 
        log, t = self.net(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)
        weight = self.calc_weight() 
        out_b = ((F.linear(coup_b, weight)) + t) * s 
        
        logdet = torch.sum(torch.log(s).view(coup_a.shape[0], -1), 1) + torch.sum(self.w_s)
        
        return out_b, logdet
    
    def reverse(self, coup_a, coup_b):
        log, t = self.net(coup_a).chunk(2, 2)
        s = torch.sigmoid(log + 2)     
        weight = self.calc_weight() 
        in_b = F.linear(coup_b / s - t, weight.inverse())
                
        return in_b

    
class ActNorm(nn.Module):
    def __init__(self, in_channel, in_seqlen, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, 1, in_channel))
        self.scale = nn.Parameter(torch.ones(1, 1, in_channel))
        
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(2, 0, 1).contiguous().view(input.shape[2], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            std = (
                flatten.std(1)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-12))

    def forward(self, input):
        _, seq_length, _ = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = seq_length * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc
    
    def decode(self, output):
        return output / self.scale - self.loc

class InvConv1dLU(nn.Module):
    def __init__(self, in_seq):
        super().__init__()
        
    def forward(self, input):
        return input, 0
    
    def reverse(self, output):
        return output

    def decode(self, output, conv_layer):
        return output
        
# class InvConv1dLU(nn.Module):
#     def __init__(self, in_seq):
#         super().__init__()

#         weight = np.random.randn(in_seq, in_seq)
#         q, _ = la.qr(weight)
#         w_p, w_l, w_u = la.lu(q.astype(np.float32))
#         w_s = np.diag(w_u)
#         w_u = np.triu(w_u, 1)
#         u_mask = np.triu(np.ones_like(w_u), 1)
#         l_mask = u_mask.T

#         w_p = torch.from_numpy(w_p)
#         w_l = torch.from_numpy(w_l)
#         w_s = torch.from_numpy(w_s)
#         w_u = torch.from_numpy(w_u)

#         self.register_buffer('w_p', w_p)
#         self.register_buffer('u_mask', torch.from_numpy(u_mask))
#         self.register_buffer('l_mask', torch.from_numpy(l_mask))
#         self.register_buffer('s_sign', torch.sign(w_s))
#         self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
#         self.w_l = nn.Parameter(w_l)
#         self.w_s = nn.Parameter(logabs(w_s))
#         self.w_u = nn.Parameter(w_u)

#     def forward(self, input):
#         _, _, in_channel = input.shape
        
#         weight = self.calc_weight()     
#         weight = torch.tril(weight)
#         out = F.linear(input.transpose(2, 1), weight).transpose(2, 1)
#         logdet = in_channel * torch.sum(self.w_s)
    
#         return out, logdet

#     def calc_weight(self):
#         weight = (
#             self.w_p
#             @ (self.w_l * self.l_mask + self.l_eye)
#             @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
#         )
#         return weight
    
#     def reverse(self, output):
#         weight = self.calc_weight() 
#         weight = torch.tril(weight.double()).inverse().float()

#         return F.linear(output.transpose(2, 1), weight).transpose(2, 1)

#     def decode(self, output, conv_layer):
#         if conv_layer is not None:
#             output = torch.cat([conv_layer, output], 1)
#         length = output.size(1)
#         weight = torch.tril(self.calc_weight().double()).inverse().float()[:length, :length]

#         return F.linear(output.transpose(2, 1), weight).transpose(2, 1)[:, -1:]
               
class ZeroLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.linear = nn.Linear(in_channel, out_channel)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, 1, out_channel))

    def forward(self, input):
        out = self.linear(input)
        out = out * torch.exp(self.scale * 3)

        return out

    
