import os
import time
import pandas as pd
from argparse import Namespace
from torch.utils.data import DataLoader
import torch.optim as optim
from label_tag_model import MultiIntentTagModel
from dictionary import Dictionary, TokenDictionary
import torch
from utils import build_vocab, get_label_dict, load_from_txt, get_tag_dict
from utils import make_train_state, update_train_state
from utils import load_w2c, get_n_params
from dataset import LabelTagDataset
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
        for i, samples in enumerate(train_loader):
            x = samples['x']
            y = samples['y']
            mask = samples['mask']
            optimizer.zero_grad()
            out = model(x, mask)
            loss = model.compute_loss(out, y, mask)
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
        train_file='train.txt',
        val_file='dev.txt',
        test_file='test.txt',
        model_dir='LS_Models/MixATIS',
        w2c_path='models/pre_trained',
        vocab_file='vocab.txt',
        intent_file='intent.txt',
        slot_file='slot.txt',

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
        checkpoint=False,
        thresh=0.7,
        clip=5,
        stop_thresh=0.01

        
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

    # Load data from file .txt
    train_df = load_from_txt(args.data_path + '/'+ args.train_file)
    val_df = load_from_txt(args.data_path +'/'+ args.val_file)
    test_df = load_from_txt(args.data_path + '/'+ args.test_file)

    train_df.to_csv(args.data_path + '/train.csv', index=False)
    val_df.to_csv(args.data_path + '/val.csv', index=False)
    test_df.to_csv(args.data_path + '/test.csv', index=False)
    # Load vocab and label from file, if not exist create from data.
    if os.path.exists(args.data_path + '/' + args.vocab_file) is True:
        seq_vocab = TokenDictionary.load(args.data_path + '/' + args.vocab_file)
    else:
        seq_vocab = build_vocab(train_df['text'].tolist())
        seq_vocab.save(args.data_path + '/' + args.vocab_file, delimiter='\t')

    if os.path.exists(args.data_path + '/' + args.intent_file) is True:
        label_dict = Dictionary.load(args.data_path + '/' + args.intent_file)
    else:
        label_dict = get_label_dict(pd.concat([train_df, test_df, val_df])['label'].tolist())
        label_dict.save(args.data_path + '/' + args.intent_file , delimiter='\t')

    if os.path.exists(args.data_path + '/' + args.slot_file) is True:
        tag_dict = TokenDictionary.load(args.data_path + '/' + args.slot_file)
    else:
        tag_dict = get_tag_dict(pd.concat([train_df, test_df, val_df])['tags'].tolist())
        label_dict.save(args.data_path + '/' + args.slot_file , delimiter='\t')


    # Create train, val, test dataset.
    train_dataset = LabelTagDataset(train_df, seq_vocab, label_dict, label_dict, multi_label=True, max_seq_len=args.max_seq_len)
    max_seq_len = train_dataset.max_len_seq

    val_dataset = LabelTagDataset(val_df, seq_vocab, label_dict, label_dict, multi_label=True, max_seq_len=max_seq_len)
    test_dataset = LabelTagDataset(test_df, seq_vocab, label_dict, label_dict,multi_label=True, max_seq_len=max_seq_len)

    

    # # Load word2vec embedding
    # w2c = load_w2c(args.w2c_path + '/viglove_{}D.txt'.format(args.word_embed_size), seq_vocab.item2idx,
    #                embed_size=args.word_embed_size)

    n_labels = len(label_dict)
    n_tags = len(tag_dict)
    padding_idx = seq_vocab.padding_idx
    vocab_size = len(seq_vocab)

    model = MultiIntentTagModel(
        n_labels=n_labels,
        n_tags=n_tags,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        embed_size=args.word_embed_size,
        hidden_size=args.hidden_size,
        n_rnn_layers=args.n_rnn_layers,
        dropout=args.dropout,
        att_method='general',
        device=device,
        word2vec=None,
        mode='all'
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
    try:
        model = train(model, train_dataset, val_dataset, args)
        print()
    except KeyboardInterrupt as e:
        print(e)
        print('\nSave last model at {}\n'.format(args.model_dir + '/final_model.pth'))
        torch.save(model.state_dict(), args.model_dir + '/final_model.pth')
    
    finally:
        model = MultiIntentTagModel(
            n_labels=n_labels,
            n_tags=n_tags,
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            embed_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            n_rnn_layers=args.n_rnn_layers,
            dropout=args.dropout,
            att_method='general',
            device=device,
            word2vec=None,
            mode='all'
        )
        print("Load best model to evaluate......!")
        model.from_pretrained(args.model_dir + '/best_model.pth')
        model.to(device)
        print(31*'_' + '          ' + 31*'_')
        print(31*'_' + 'EVALUATION' + 31*'_')
        print()
        model.evaluate(test_dataset, args.thresh)
        print(72*'_')
