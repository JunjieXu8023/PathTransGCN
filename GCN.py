from torch import nn
import torch.nn.functional as F
import torch
import math
from torch import nn
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, infeas, outfeas, bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = infeas
        self.out_features = outfeas
        self.weight = Parameter(torch.FloatTensor(infeas, outfeas))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outfeas))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
        '''
        for name, param in GraphConvolution.named_parameters(self):
            if 'weight' in name:
                #torch.nn.init.constant_(param, val=0.1)
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)
        '''

    def forward(self, x, adj):
        x1 = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)


        x = self.fc(x)
