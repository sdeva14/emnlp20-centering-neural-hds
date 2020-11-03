# -*- coding: utf-8 -*-

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tree_trans.attention import *
from torch.nn import CrossEntropyLoss
from models.tree_trans.modules import *

    
class Encoder(nn.Module):
    # def __init__(self, layer, N, d_model, vocab_size):
    def __init__(self, layer, N, d_model, dropout):
        super(Encoder, self).__init__()
        # self.word_embed = word_embed
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        # self.proj = nn.Linear(d_model, vocab_size)
        self.pos_emb = PositionalEncoding(d_model, dropout)

    # def forward(self, inputs, mask):
    def forward(self, inputs, mask, adj_mat):
        break_probs = []
        # x = self.word_embed(inputs)  # we do not need an embedding layer
        x = inputs

        # put positional encoding
        n_sents = inputs.shape[1]
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = x + pos_emb

        ##
        group_prob = 0.
        for layer in self.layers:
            # x,group_prob,break_prob = layer(x, mask, group_prob)
            x,group_prob,break_prob = layer(x, mask, group_prob, adj_mat)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)
        # return self.proj(x),break_probs  # original
        return x, break_probs  # modified, because we do not need a projection layer to the vocabulary


    def masked_lm_loss(self, out, y):
        fn = CrossEntropyLoss(ignore_index=-1)
        return fn(out.view(-1, out.size()[-1]), y.view(-1))


    def next_sentence_loss(self):
        pass


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, self_attn, feed_forward, group_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.group_attn = group_attn
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask, group_prob, adj_mat):
        # group_prob,break_prob = self.group_attn(x, mask, group_prob)
        group_prob,break_prob = self.group_attn(x, mask, group_prob, adj_mat)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, group_prob, mask))
        return self.sublayer[1](x, self.feed_forward), group_prob, break_prob

