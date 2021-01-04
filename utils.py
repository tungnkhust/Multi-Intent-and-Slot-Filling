import os
from typing import List
from dictionary import Dictionary, TokenDictionary
import pandas as pd
import torch
import numpy as np

def load_from_txt(file_path, label_delimiter='#'):
    with open(file_path, 'r') as pf:
        lines = pf.readlines()

    text = []
    tags = []
    final_list = []
    for line in lines:
        line = line.strip('\n')
        if line == '':
            continue
        line = line.split(' ')
        if len(line) == 1:
            final_list.append({'text': ' '.join(text), 'label': ' '.join(line[0].split(label_delimiter)), 'tags': ' '.join(tags)})
            text = []
            tags = []
        else:
            text.append(line[0])
            tags.append(line[1])


    return pd.DataFrame(final_list)


def build_vocab(texts: List[str]):
    seq_vocab = TokenDictionary()
    for text in texts:
        text = text.replace('\n', '')
        tokens = text.split(' ')
        seq_vocab.add_items(tokens)
    return seq_vocab


def get_label_dict(labels: List[str], delimiter=' '):
    label_dict = Dictionary()
    for label in labels:
        label = label.replace('\n', '')
        label_dict.add_items(label.split(delimiter))
    return label_dict


def load_w2c(w2c_path: str, token2idx, embed_size):
    w2c = {}
    with open(w2c_path, 'r') as pf:
        lines = pf.readlines()
        lines = [line.replace('\n', '').split(' ') for line in lines]
    for l in lines:
        w2c[l[0]] = torch.tensor([float(t) for t in l[1:]], dtype=torch.float32)
    weight = torch.randn(len(token2idx), embed_size)
    for w in token2idx:
        if w in w2c:
            weight[token2idx[w]] = w2c[w]
    return weight


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def set_seed(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_train_state(args):
    return {
        'stop_early': False,
        'early_stop_num_epoch': 0,
        'early_stop_max_epochs': args.early_stop_max_epochs,
        'early_stop_best_val_loss': 1e8,
        'epoch_index': 0,
        'model_dir': args.model_dir,
        'learning_rate': args.lr,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }


def update_train_state(model, train_state, save_all_checkpoint=False):
    if save_all_checkpoint:
            torch.save(model.state_dict(),
                train_state['model_dir'] + '/checkpoint{}.pth'.format(train_state['epoch_index']))

    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_dir'] + '/last_checkpoint.pth')
    else:
        torch.save(model.state_dict(), train_state['model_dir'] + '/last_checkpoint.pth')
        loss_t = train_state['val_loss'][-1]
        if loss_t < train_state['early_stop_best_val_loss']:
            print('Save best model at', train_state['model_dir'] + '/best_model.pth')
            torch.save(model.state_dict(), train_state['model_dir'] + '/best_model.pth')
            train_state['early_stop_num_epoch'] = 0
            train_state['early_stop_best_val_loss'] = loss_t
        else:
            train_state['early_stop_num_epoch'] += 1

        if train_state['early_stop_num_epoch'] >= train_state['early_stop_max_epochs']:
            train_state['stop_early'] = True

    return train_state