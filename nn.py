import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from torch.nn import init

class Attention(nn.Module):
    def __init__(self, hidden_size, method='general', cuda='cpu'):
        super(Attention, self).__init__()
        self.method = 'general'
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        elif self.method == 'concat':
            self.linear = nn.Linear(2*hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size)).to(cuda)

    def init_weight(self):
        if self.method == 'general':
            init.xavier_uniform_(self.linear.weight)
        elif self.method == 'concat':
            init.xavier_uniform_(self.linear.weight)
            init.xavier_uniform_(self.weight)

    def score(self, hidden, encoder_outputs):

        if self.method == 'dot':
            score = encoder_outputs.bmm(hidden.view(1, -1, 1)).squeeze(-1)
            return score

        elif self.method == 'general':
            out = self.linear(hidden)
            score = encoder_outputs.bmm(out.unsqueeze(-1)).squeeze(-1)
            return score

        elif self.method == 'concat':
            out = self.linear(torch.cat((hidden, encoder_outputs), 1))
            score = out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)
            return score

    def forward(self, hidden, encoder_outputs, mask=None):
        score = self.score(hidden, encoder_outputs)
        if mask is not None:
            score = score * mask
        att_w = f.softmax(score, -1)
        if mask is not None:
            att_w = att_w * mask
        att_w = att_w.unsqueeze(-1)
        out = encoder_outputs.transpose(-1, -2).bmm(att_w).squeeze(-1)
        return out