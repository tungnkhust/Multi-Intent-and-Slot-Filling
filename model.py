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


class MultiIntentModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            n_labels,
            embed_size=100,
            hidden_size=128,
            n_rnn_layers=2,
            dropout=0.3,
            att_method='general',
            word2vec=None,
            padding_idx=1,
            device='cpu'
    ):
        super(MultiIntentModel, self).__init__()
        self.n_rnn_layer = n_rnn_layers
        self.hidden_size = hidden_size
        self.device = device
        if word2vec is None:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=int(padding_idx))
        else:
            self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_size, hidden_size, n_rnn_layers,
                           dropout=dropout, bias=True, bidirectional=True)
        self.att = Attention(hidden_size*2, method=att_method)
        self.drop = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_size*4, n_labels)

        self.init_weight()

    def init_weight(self):
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.rnn.__getattr__(p))
                    
        self.att.init_weight()
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def init_hidden(self, batch_size):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        h0 = torch.randn(self.n_rnn_layer * 2, batch_size, self.hidden_size)
        c0 = torch.randn(self.n_rnn_layer * 2, batch_size, self.hidden_size)
        return h0.to(device), c0.to(device)

    def forward(self, word_vector: Tensor, mask: Tensor = None):
        bs, seq_len = word_vector.shape
        embed = self.embedding(word_vector)
        h_0, c_0 = self.init_hidden(bs)
        embed = torch.transpose(embed, 0, 1)
        encoder_outputs, (h_n, c_n) = self.rnn(embed, (h_0, c_0))
        hidden_state = encoder_outputs
        encoder_outputs = encoder_outputs.transpose(0, 1)
        hidden_state = hidden_state.view(seq_len, bs, 2, self.hidden_size)
        forward = hidden_state[-1, :, 0, :]
        backward = hidden_state[0, :, 1, :]
        context = torch.cat([forward, backward], dim=-1)

        att_context = self.att(context, encoder_outputs, mask)
        hidden = torch.cat([att_context, context], -1)
        # hidden = context
        out = self.linear(self.drop(hidden))
        out = torch.sigmoid(out)

        return out

    def forward_loss(self, word_vector, label, mask: Tensor = None):
        out = self.forward(word_vector, mask)
        loss = nn.BCELoss(reduction='sum')(out, label)
        return loss, out

