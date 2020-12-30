import torch
import torch.nn as nn
from torch import Tensor


class LanguageModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder
    ):
        super(LanguageModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor, encode_hidden, mask=None):
        hidden = self.encoder(x, encode_hidden, mask)
        out = self.decoder(hidden)
        return out

    def forward_loss(self, word_vector: Tensor, chars_vector: Tensor, y, mask=None):
        hidden = self.encoder(word_vector, chars_vector)
        word_embed = self.encoder.word_embed(word_vector)
        loss = self.decoder.forward_loss(hidden, y, word_embed, mask)
        return loss

    def save(self, model_path):
        torch.save(self.encoder.state_dict(), model_path)
