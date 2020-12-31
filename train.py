import os
import time
import pandas as pd
from argparse import Namespace
from torch.utils.data import DataLoader
import torch.optim as optim
from model import MultiIntentModel
from dictionary import Dictionary, TokenDictionary
import torch
from utils import build_vocab, get_label_dict, load_from_txt
from utils import make_train_state, update_train_state
from utils import load_w2c, get_n_params
from dataset import LabelDataset
from metrics import get_multi_label_metrics


def train(
        model,
        train_dataset,
        val_dataset,
        args: Namespace
    ):
    """
    :param model: model pytorch
    :param train_dataset:
    :param val_dataset:
    :param args:
    :return:
    """

    checkpoint_path = args.model_dir + '/last_checkpoint.pth'
    if args.checkpoint and os.path.exists(checkpoint_path):
        print("load model from " + checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

    if os.path.exists(args.model_dir) is False:
        os.mkdir(args.model_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    train_state = make_train_state(args)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        print("Epoch: ", epoch + 1)
        train_state['epoch_index'] = epoch
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loss = 0.
        model.train()
        for i, (x_vector, y_vector, x_mask) in enumerate(train_loader):
            optimizer.zero_grad()
            loss, out = model.forward_loss(x_vector.to(device), y_vector.to(device), x_mask.to(device))
            loss.backward()
            optimizer.step()
            train_loss += (loss.item() - train_loss) / (i + 1)
            if (i + 1) % 10 == 0:
                print("\tStep: {} train_loss: {}".format(i + 1, loss.item()))

        # writer.add_scalar('Train/Loss', train_loss, epoch)
        train_state['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.
        y_pred = []
        y_true = []

        for i, (x_vector, y_vector, x_mask) in enumerate(val_loader):
            optimizer.zero_grad()
            loss, out = model.forward_loss(x_vector.to(device), y_vector.to(device), x_mask.to(device))
            val_loss += (loss.item() - val_loss) / (i + 1)
            scheduler.step(val_loss)
            y_pred.append((out > args.thresh).long())
            y_true.append(y_vector.long())

            train_state['val_loss'].append(val_loss)
            train_state = update_train_state(model, train_state)

        y_true = torch.cat(y_true, dim=-1).cpu().detach().numpy()
        y_pred = torch.cat(y_pred, dim=-1).cpu().detach().numpy()

        acc, sub_acc, f1, precision, recall, hamming_loss = get_multi_label_metrics(y_true=y_true, y_pred=y_pred)
        print('f1: {}    precision: {}    recall: {}'.format(f1, precision, recall))
        print('accuracy: {}    accuracy : {}    hamming loss: {}'.format(acc, sub_acc, hamming_loss))
        # save best model.
        torch.save(model.state_dict(), args.model_dir + '/' + args.model_name)


if __name__ == '__main__':
    args = Namespace(
        data_path='data/ATIS',
        train_path='data/ATIS/train.txt',
        val_path='data/ATIS/dev.txt',
        test_path='data/ATIS/test.txt',
        model_dir='models/',
        w2c_path='models/pre_trained',
        model_name='multi_intent.pth',
        vocab_path='data/ATIS/vocab.txt',
        intent_path='data/ATIS/intent.txt',
        batch_size=128,
        max_seq_len=80,
        early_stop_max_epochs=3,
        lr=0.1,
        num_epochs=100,
        checkpoint=False,
        thresh=0.6,
        # model
        word_embed_size=100,
        hidden_size=100,
        n_rnn_layers=2,
        dropout=0.3,
        mode='all'
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load data from file .txt
    train_df = load_from_txt(args.train_path)
    val_df = load_from_txt(args.val_path)
    test_df = load_from_txt(args.test_path)

    train_df.to_csv(args.data_path + '/train.csv', index=False)
    val_df.to_csv(args.data_path + '/val.csv', index=False)
    test_df.to_csv(args.data_path + '/test.csv', index=False)
    # Load vocab and label from file, if not exist create from data.
    if os.path.exists(args.vocab_path) is True:
        seq_vocab = TokenDictionary.load(args.vocab_path)
    else:
        seq_vocab = build_vocab(train_df['text'].tolist())
        seq_vocab.save(args.data_path + '/vocab.txt', delimiter='\t')

    if os.path.exists(args.intent_path) is True:
        label_dict = Dictionary.load(args.intent_path)
    else:
        label_dict = get_label_dict(pd.concat([train_df, test_df, val_df])['label'].tolist())
        label_dict.save(args.data_path + '/intent.txt', delimiter='\t')

    # Create train, val, test dataset.
    train_dataset = LabelDataset(train_df, seq_vocab, label_dict, multi_label=True, max_seq_len=args.max_seq_len)
    max_seq_len = train_dataset.max_len_seq

    val_dataset = LabelDataset(val_df, seq_vocab, label_dict, multi_label=True, max_seq_len=max_seq_len)
    test_dataset = LabelDataset(test_df, seq_vocab, label_dict, multi_label=True, max_seq_len=max_seq_len)

    print('Num training samples  :', len(train_df))
    print('Num validation samples:', len(val_df))
    print('Num test samples:', len(test_df))

    # Load word2vec embedding
    w2c = load_w2c(args.w2c_path + '/viglove_{}D.txt'.format(args.word_embed_size), seq_vocab.item2idx,
                   embed_size=args.word_embed_size)

    n_labels = len(label_dict)
    padding_idx = seq_vocab.padding_idx
    vocab_size = len(seq_vocab)

    print('Num labels:', n_labels)
    print("Max sequence length:", max_seq_len)
    model = MultiIntentModel(
        n_labels=n_labels,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        embed_size=args.word_embed_size,
        hidden_size=args.hidden_size,
        n_rnn_layers=args.n_rnn_layers,
        dropout=args.dropout,
        word2vec=w2c,
    )
    # print(model)
    print('Total parameter       :', get_n_params(model))
    try:
        train(model, train_dataset, val_dataset, args)
    except InterruptedError:
        torch.save(model.state_dict(), args.model_dir + '/' + args.model_name)
