import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from torch.nn import init
from nn import Attention
from metrics import get_multi_label_metrics
from torch.utils.data import DataLoader


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

        self.to(device)

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

    def compute_loss(self, out, label, mask: Tensor = None):
        loss = nn.BCELoss(reduction='sum')(out, label)
        return loss

    def evaluate(self, test_dataset, thresh=0.8, batch_size=32):
        self.eval()

        device = self.device
        y_pred = []
        y_true = []

        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        val_loss = 0.0

        for i, (x_vector, y_vector, x_mask) in enumerate(test_loader):
            out = self.forward(x_vector.to(device), x_mask.to(device))
            loss = self.compute_loss(out, y_vector.to(device))

            y_pred.append((out > thresh).long())
            y_true.append(y_vector.long())
            val_loss += (loss.item() - val_loss) / (i + 1)

        y_true = torch.cat(y_true, dim=-1).cpu().detach().numpy()
        y_pred = torch.cat(y_pred, dim=-1).cpu().detach().numpy()

        acc, sub_acc, f1, precision, recall, hamming_loss = get_multi_label_metrics(y_true=y_true, y_pred=y_pred)
        print('+----------+-----------+----------+---------+-------------+-------------+')
        print('|f1_score  |precision  |recall    |accuracy |sub accuracy |hamming loss |')
        print('+----------+-----------+----------+---------+-------------+-------------+')
        print('|{:.4f}    |{:.4f}     |{:.4f}    |{:.4f}   |{:.4f}       |{:.4f}       |'.format(f1, precision, recall, acc, sub_acc, hamming_loss))
        print('+----------+-----------+----------+---------+-------------+-------------+')
        return val_loss, f1, acc

    
    def from_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))