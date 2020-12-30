from dictionary import Dictionary, TokenDictionary
from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import List


class LanguageModelDataset(Dataset):
    def __init__(
            self,
            data: List[str],
            max_seq_len: int,
            max_token_len: int,
            seq_vocab: TokenDictionary,
            char_vocab: TokenDictionary,
    ):
        self.max_len_seq = max_seq_len + 1
        self.max_len_word = max_token_len
        self.data = data
        self.seq_vocab = seq_vocab
        self.char_vocab = char_vocab

    def __len__(self):
        return len(self.data)

    def n_tokens(self):
        return len(self.seq_vocab)

    def n_chars(self):
        return len(self.char_vocab)

    def __getitem__(self, index):
        text = self.data[index]
        x = self.seq_vocab.bos_token + " " + text
        y = text + " " + self.seq_vocab.eos_token
        x_tokens = x.split(' ')
        y_tokens = y.split(' ')
        x_vector, x_mask = self.seq_vocab.encode(x_tokens, max_len=self.max_len_seq)
        y_vector, y_mask = self.seq_vocab.encode(y_tokens, max_len=self.max_len_seq)
        x_chars_vector = torch.zeros((self.max_len_seq, self.max_len_word), dtype=torch.int64)
        mask_chars = torch.zeros((self.max_len_seq, self.max_len_word), dtype=torch.int64)
        for i, token in enumerate(x_tokens):
            char = [c for c in token]
            char_vector, mask_c = self.char_vocab.encode(char, max_len=self.max_len_word)
            x_chars_vector[i] = char_vector
            mask_chars[i] = mask_c

        return x_vector, x_chars_vector, y_vector, x_mask

    @classmethod
    def from_csv(cls, file_path: str):
        data_df = pd.read_csv(file_path, usecols=['text'])

        seq_vocab = TokenDictionary()
        char_vocab = TokenDictionary()

        max_seq_len = 0
        max_token_len = 0
        for i, row in data_df.iterrows():
            tokens = row['text'].split(' ')
            chars = list(set([c for c in row['text']]))

            max_seq_len = max(max_seq_len, len(tokens))
            max_token_len = max(max_token_len, max([len(token) for token in tokens]))

            seq_vocab.add_items(tokens)
            char_vocab.add_items(chars)

        return cls(
            data_df,
            max_seq_len,
            max_token_len,
            seq_vocab,
            char_vocab
        )


class LabelDataset(Dataset):
    def __init__(
            self,
            data_df,
            seq_vocab: TokenDictionary,
            label_dict: Dictionary,
            max_seq_len: int = None,
            multi_label=False
    ):

        self.data_df = data_df
        self.seq_vocab = seq_vocab
        self.label_dict = label_dict
        self.multi_label = multi_label
        if max_seq_len is not None:
            self.max_len_seq = max_seq_len
        else:
            self.max_len_seq = self.compute_max_seq_len() + 1

    def __len__(self):
        return len(self.data_df)

    def n_tokens(self):
        return len(self.seq_vocab)

    def n_labels(self):
        return len(self.label_dict)

    def compute_max_seq_len(self):
        texts = self.data_df['text'].tolist()
        max_seq_len = 0
        for text in texts:
            max_seq_len = max(max_seq_len, len(text.split(' ')))
        return max_seq_len

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        x_tokens = row['text'].split(' ')
        y_labels = row['label'].split(' ')
        x_vector, x_mask = self.seq_vocab.encode(x_tokens, max_len=self.max_len_seq)
        y_vector = self.label_dict.encode(y_labels, self.multi_label)
        return x_vector, y_vector, x_mask

    @classmethod
    def from_csv(cls, file_path: str):
        data_df = pd.read_csv(file_path, usecols=['text'])

        seq_vocab = TokenDictionary()
        label_dict = Dictionary()

        max_seq_len = 0
        for i, row in data_df.iterrows():
            tokens = row['text'].split(' ')
            max_seq_len = max(max_seq_len, len(tokens))

            seq_vocab.add_items(tokens)
            labels = row['label'].split(' ')
            label_dict.add_items(labels)

        return cls(
            data_df,
            max_seq_len,
            seq_vocab,
            label_dict
        )


class TagDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        max_len_seq,
        max_len_word,
        seq_vocab: TokenDictionary,
        char_vocab: TokenDictionary,
        tagger_dict: TokenDictionary,
    ):
        self.max_len_seq = max_len_seq
        self.max_len_word = max_len_word
        self.data_df = data_df
        self.seq_vocab = seq_vocab
        self.char_vocab = char_vocab
        self.tagger_dict = tagger_dict

    def cap_tag(self, word: str):
        if word.isupper():
            return 1
        if word.islower():
            return 2
        if word[0].isupper() and word[1:].islower():
            return 3
        return 4

    def cap_tags(self, text: str = None, max_len_seq=None):
        tokens = text.split(' ')
        if max_len_seq is None:
            max_len_seq = len(tokens)
        caps = [self.cap_tag(word) for word in tokens]

        cap_vec = torch.zeros(max_len_seq)
        cap_vec[:len(tokens)] = caps
        return cap_vec

    def __get_df__(self):
        return self.data_df

    def __len__(self):
        return len(self.data_df)

    def n_tokens(self):
        return len(self.seq_vocab)

    def n_chars(self):
        return len(self.char_vocab)

    def n_tags(self):
        return len(self.tagger_dict)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        tags = sample.tag.split(' ')
        tokens = sample.text.split(' ')

        tokens_vector = self.seq_vocab.encode(tokens, max_len=self.max_len_seq)
        chars_vector = []
        for token in tokens:
            chars = [c for c in token]
            chars_vector.append(self.char_vocab.encode(chars, max_len=self.max_len_word))
        tags_idx = self.tagger_dict.encode(tags, max_len=self.max_len_seq)
        return (tokens_vector, chars_vector), tags_idx, len(tokens)

    @classmethod
    def from_csv(cls, file_path: str):
        data_df = pd.read_csv(file_path)
        data_df.columns = ['text', 'tags', 'label']

        seq_vocab = TokenDictionary()
        char_vocab = TokenDictionary()
        tagger_dict = TokenDictionary()

        max_seq_len = 0
        max_token_len = 0
        for i, row in data_df.iterrows():
            tokens = row['text'].split(' ')
            tags = row['tags'].split(' ')
            chars = list(set([c for c in row['text']]))

            max_seq_len = max(max_seq_len, len(tokens))
            max_token_len = max(max_token_len, max([len(token) for token in tokens]))

            seq_vocab.add_items(tokens)
            char_vocab.add_items(chars)
            tagger_dict.add_items(tags)

        return cls(
            data_df,
            max_seq_len,
            max_token_len,
            seq_vocab,
            char_vocab,
            tagger_dict
        )