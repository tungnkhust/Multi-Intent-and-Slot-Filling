import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from torch.nn import init
from nn import Attention


class LSTMDecoder(nn.Module):
    def __init__(
            self,
            input_size,
            n_out,
    ):
        super(LSTMDecoder, self).__init__()
        self.rnn = nn.LSTM(input_size, n_out, num_layers=1, bias=False, bidirectional=False)

        self.init_weight()

    def init_weight(self):
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.rnn.__getattr__(p))

    def forward(self, input_hidden, mask=None):
        """
        input_hidden: (bs, seq_len, input_size)
        mask: (bs, seq_len)
        """
        out, (_, _) = self.rnn(torch.transpose(input_hidden, 0, 1))
        out = out.transpose(0, 1)
        return out

    def forward_loss(self, input_hidden, labels, tags, mask=None):
        """
        input_hidden: (bs, seq_len, input_size)
        mask: (bs, seq_len)
        """
        out = self.forward(init_hidden)
        loss_element = nn.NLLLoss(reduction='none')(out, label.view(-1))
        if mask is not None:
            loss_element = loss_element * mask.view(-1)
        loss_element = loss_element.view(-1, seq_len)
        loss_element = loss_element.sum(dim=1)
        loss = torch.mean(loss_element)
        return loss


class MultiIntentTagModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            n_labels,
            n_tags,
            embed_size=100,
            hidden_size=128,
            n_rnn_layers=2,
            dropout=0.3,
            att_method='general',
            word2vec=None,
            padding_idx=1,
            device='cpu',
            mode='all'
    ):
        super(MultiIntentTagModel, self).__init__()
        self.n_rnn_layer = n_rnn_layers
        self.hidden_size = hidden_size
        self.device = device
        self.n_tags = n_tags
        if word2vec is None:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=int(padding_idx))
        else:
            self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_size, hidden_size, n_rnn_layers,
                           dropout=dropout, bias=True, bidirectional=True)
        self.att = Attention(hidden_size*2, method=att_method)
        self.drop = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_size*4, n_labels)

        self.tags_decoder = LSTMDecoder(input_size=hidden_size*6, n_out=n_tags)
        self.init_weight()

        self.mode = mode

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

    def forward(self, x, mask=None):
        word_vector = x.to(self.device)
        
        if mask is not None:
            mask = mask.to(self.device)
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

        label_out = self.linear(self.drop(hidden))
        label_out = torch.sigmoid(label_out)

        tags_context = torch.stack([self.att(encoder_outputs[:, i, :], encoder_outputs, mask) for i in range(seq_len)], dim=-1).transpose(-2, -1)
        contexts = torch.stack([context]*seq_len, dim=-1).transpose(-2, -1)
        tags_input = torch.cat([tags_context, contexts, encoder_outputs], dim=-1)
        
        tags_out = self.tags_decoder(tags_input)

        return (label_out, tags_out)

    def compute_loss(self, out, y, mask=None):
        (label, tags) = y
        label = label.to(self.device)
        tags = tags.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        (label_out, tags_out) = out
        bs, seq_len = tags.shape

        label_loss = nn.BCELoss(reduction='sum')(label_out, label)

        tags_out = torch.reshape(tags_out, (-1, self.n_tags))
        tags_out = nn.LogSoftmax(dim=-1)(tags_out)
        loss_element = nn.NLLLoss(reduction='none')(tags_out, tags.view(-1))

        if mask is not None:
            loss_element = loss_element * mask.view(-1)
        loss_element = loss_element.view(-1, seq_len)
        loss_element = loss_element.sum(dim=1)
        tags_loss = torch.mean(loss_element)

        if self.mode == 'tag':
            loss = tags_loss
        elif self.mode == 'label':
            loss = label_loss
        elif self.mode == 'all':
            loss = tags_loss + label_loss
        else:
            loss = tags_loss + label_loss
            
        return loss

    def evaluate(self, val_loader, thresh=0.8, device='cpu'):
        self.eval()
        y_pred = []
        y_true = []

        y_pred = []
        y_true = []

        val_loss = 0.0

        for i, (x_vector, y_vector, yt_vector, x_mask) in enumerate(val_loader):
            out = self.forward(x_vector.to(device), x_mask.to(device))
            loss = self.compute_loss(out, y_vector.to(device), yt_vector.to(device), mask.to(device))

            label_out, tags_out = out
            y_pred.append((label_out > thresh).long())
            y_true.append(y_vector.long())
            val_loss += (loss.item() - val_loss) / (i + 1)

        y_true = torch.cat(y_true, dim=-1).cpu().detach().numpy()
        y_pred = torch.cat(y_pred, dim=-1).cpu().detach().numpy()

        acc, sub_acc, f1, precision, recall, hamming_loss = get_multi_label_metrics(y_true=y_true, y_pred=y_pred)
        print("val loss:", val_loss)
        print('f1: {}    precision: {}    recall: {}'.format(f1, precision, recall))
        print('accuracy: {}    sub accuracy : {}    hamming loss: {}'.format(acc, sub_acc, hamming_loss))
        print()

        return val_loss, f1, acc

    def from_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))