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

    checkpoint_path = args.model_dir + '/best_model.pth'
    if args.checkpoint and os.path.exists(checkpoint_path):
        print("load model from " + checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

    if os.path.exists(args.model_dir) is False:
        os.mkdir(args.model_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=args.num_epochs)

    train_state = make_train_state(args)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        print(32*'_' + ' '*9 + 31*'_')
        print(32*'_' + 'EPOCH {:3}'.format(epoch+1) + 31*'_')
        train_state['epoch_index'] = epoch
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_loss = 0.
        model.train()
        for i, (x_vector, y_vector, x_mask) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x_vector.to(device), x_mask.to(device))
            loss = model.compute_loss(out, y_vector.to(device), x_mask.to(device))
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            train_loss += (loss.item() - train_loss) / (i + 1)
            if (i + 1) % 10 == 0:
                print("|{:4}| train_loss = {:.4f}".format(i + 1, loss.item()))

        # writer.add_scalar('Train/Loss', train_loss, epoch)
        train_state['train_loss'].append(train_loss)
        optimizer.zero_grad()
        val_loss, f1, acc = model.evaluate(val_dataset, thresh=args.thresh, batch_size=args.batch_size)
        # scheduler.step(val_loss)
        print('Val loss: {:.4f}'.format(val_loss))
        scheduler.step()

        train_state['val_loss'].append(-f1)

        ## update train state
        train_state = update_train_state(model, train_state)

        if train_state['stop_early']:
            print('Stop early.......!')
            break

    return model
    

if __name__ == '__main__':
    args = Namespace(
        data_path='data/MixATIS',
        train_name='train.txt',
        val_name='dev.txt',
        test_name='test.txt',
        model_dir='models/MixATIS',
        w2c_path='models/pre_trained',
        vocab_name='vocab.txt',
        intent_name='intent.txt',

        # model
        word_embed_size=64,
        hidden_size=128,
        n_rnn_layers=3,
        dropout=0.4,
        mode='all',

        # train
        batch_size=64,
        max_seq_len=80,
        early_stop_max_epochs=5,
        lr=0.0001,
        num_epochs=50,
        checkpoint=True,
        thresh=0.7,
        clip=5,
        stop_thresh=0.01

        
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

    # Load data from file .txt
    train_df = load_from_txt(args.data_path + '/'+ args.train_name)
    val_df = load_from_txt(args.data_path +'/'+ args.val_name)
    test_df = load_from_txt(args.data_path + '/'+ args.test_name)

    train_df.to_csv(args.data_path + '/train.csv', index=False)
    val_df.to_csv(args.data_path + '/val.csv', index=False)
    test_df.to_csv(args.data_path + '/test.csv', index=False)
    # Load vocab and label from file, if not exist create from data.
    if os.path.exists(args.data_path + '/' + args.vocab_name) is True:
        seq_vocab = TokenDictionary.load(args.data_path + '/' + args.vocab_name)
    else:
        seq_vocab = build_vocab(train_df['text'].tolist())
        seq_vocab.save(args.data_path + '/' + args.vocab_name, delimiter='\t')

    if os.path.exists(args.data_path + '/' + args.intent_name) is True:
        label_dict = Dictionary.load(args.data_path + '/' + args.intent_name)
    else:
        label_dict = get_label_dict(pd.concat([train_df, test_df, val_df])['label'].tolist())
        label_dict.save(args.data_path + '/' + args.intent_name , delimiter='\t')

    # Create train, val, test dataset.
    train_dataset = LabelDataset(train_df, seq_vocab, label_dict, multi_label=True, max_seq_len=args.max_seq_len)
    max_seq_len = train_dataset.max_len_seq

    val_dataset = LabelDataset(val_df, seq_vocab, label_dict, multi_label=True, max_seq_len=max_seq_len)
    test_dataset = LabelDataset(test_df, seq_vocab, label_dict, multi_label=True, max_seq_len=max_seq_len)

    # # Load word2vec embedding
    # w2c = load_w2c(args.w2c_path + '/viglove_{}D.txt'.format(args.word_embed_size), seq_vocab.item2idx,
    #                embed_size=args.word_embed_size)

    n_labels = len(label_dict)
    padding_idx = seq_vocab.padding_idx
    vocab_size = len(seq_vocab)

    model = MultiIntentModel(
        n_labels=n_labels,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        embed_size=args.word_embed_size,
        hidden_size=args.hidden_size,
        n_rnn_layers=args.n_rnn_layers,
        dropout=args.dropout,
        att_method='general',
        device=device
    )
    
    print('_'*72)
    print()
    print(model)
    print('_'*72)
    print()
    print('Num training samples  :', len(train_df))
    print('Num validation samples:', len(val_df))
    print('Num test samples      :', len(test_df))
    print('Num labels            :', n_labels)
    print("Max sequence length   :", max_seq_len)
    print('Total parameter       :', get_n_params(model))
    print('_'*72)
    print()
    try:
        model = train(model, train_dataset, val_dataset, args)
        print()
    except KeyboardInterrupt as e:
        print(e)
        print('\nSave last model at {}\n'.format(args.model_dir + '/final_model.pth'))
        torch.save(model.state_dict(), args.model_dir + '/final_model.pth')
    
    finally:
        model = MultiIntentModel(
            n_labels=n_labels,
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            embed_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            n_rnn_layers=args.n_rnn_layers,
            dropout=args.dropout,
            att_method='general',
            device=device
        )
        print("Load best model to evaluate......!")
        model.from_pretrained(args.model_dir + '/best_model.pth')
        model.to(device)
        print(31*'_' + '          ' + 31*'_')
        print(31*'_' + 'EVALUATION' + 31*'_')
        print()
        model.evaluate(test_dataset, args.thresh)
        print(72*'_')
