# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-27 21:25
# @FILE    :matching_networks.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils

class PrototypicalNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = ''):
        super(PrototypicalNet, self).__init__(model_func,  n_way, n_support)

        self.loss_type = loss_type  #'softmax'# 'mse'
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()  
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)
        z_proto     = z_support.reshape(self.n_support,self.n_way,-1).mean(0) 
        logits = euclidean_metric(z_query.squeeze(1), z_proto)
        
        return logits

   
    def set_forward_loss(self, x):

        y = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        scores = self.set_forward(x)
        loss = F.cross_entropy(scores, y.cuda())

        return loss


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
