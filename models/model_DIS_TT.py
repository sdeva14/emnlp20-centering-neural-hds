# -*- coding: utf-8 -*-

# Copyright 2020 Sungho Jeon and Heidelberg Institute for Theoretical Studies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# import torch.distributions.normal as normal
import logging
import math

import networkx as nx
import collections

import w2vEmbReader

from models.encoders.encoder_main import Encoder_Main

import models.model_base
from models.model_base import masked_softmax

import utils
from utils import FLOAT, LONG, BOOL

import torch.nn.utils.weight_norm as weightNorm

import fairseq.modules as fairseq

from models.transformer.encoder import TransformerInterEncoder

import models.tree_trans.attention as tt_attn
import models.tree_trans.models as tt_model
import models.tree_trans.modules as tt_module
import copy

# from apex.normalization.fused_layer_norm import FusedLayerNorm

logger = logging.getLogger()

class Model_DIS_TT(models.model_base.BaseModel):
    def __init__(self, config, corpus_target, embReader):
        super().__init__(config)

        ####
        # init parameters
        self.corpus_target = config.corpus_target
        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
        self.avg_num_sents = config.avg_num_sents
        self.batch_size = config.batch_size

        self.avg_len_doc = config.avg_len_doc

        self.vocab = corpus_target.vocab  # word2id
        self.rev_vocab = corpus_target.rev_vocab  # id2word
        self.pad_id = corpus_target.pad_id
        self.num_special_vocab = corpus_target.num_special_vocab

        self.embed_size = config.embed_size
        self.dropout_rate = config.dropout
        self.rnn_cell_size = config.rnn_cell_size
        self.path_pretrained_emb = config.path_pretrained_emb
        self.num_layers = 1
        self.output_size = config.output_size  # the number of final output class
        self.pad_level = config.pad_level

        self.use_gpu = config.use_gpu
        self.gen_logs = config.gen_logs

        if not hasattr(config, "freeze_step"):
            config.freeze_step = 5000

        ########

        #
        self.base_encoder = Encoder_Main(config, embReader)

        #
        self.sim_cosine_d0 = torch.nn.CosineSimilarity(dim=0)
        self.sim_cosine_d2 = torch.nn.CosineSimilarity(dim=2)

        ## tree-transformer
        c = copy.deepcopy
        N=4  # num of layers
        d_model=self.base_encoder.encoder_out_size
        d_ff=self.base_encoder.encoder_out_size
        num_heads=4
        dropout=self.dropout_rate
        
        attn = tt_attn.MultiHeadedAttention(num_heads, d_model)
        group_attn = tt_attn.GroupAttention(d_model)
        ff = tt_module.PositionwiseFeedForward(d_model, d_ff, dropout)
        position = tt_module.PositionalEncoding(d_model, dropout)
        # word_embed = nn.Sequential(Embeddings(d_model, vocab_size), c(position))
        self.tt_encoder = tt_model.Encoder(tt_model.EncoderLayer(d_model, c(attn), c(ff), group_attn, dropout), 
                N, d_model, dropout)  # we do not need an embedding layer here
        
        for p in self.tt_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
        if self.use_gpu:
            self.tt_encoder.cuda()

        
        self.context_weight = nn.Parameter(torch.zeros(self.base_encoder.encoder_out_size,1))
        nn.init.xavier_uniform_(self.context_weight)
        
        #####################
        fc_in_size = self.base_encoder.encoder_out_size

        linear_1_out = fc_in_size // 2
        linear_2_out = linear_1_out // 2

        self.linear_1 = nn.Linear(fc_in_size, linear_1_out)
        nn.init.xavier_uniform_(self.linear_1.weight)

        self.linear_2 = nn.Linear(linear_1_out, linear_2_out)
        nn.init.xavier_uniform_(self.linear_2.weight)

        self.linear_out = nn.Linear(linear_2_out, self.output_size)
        if corpus_target.output_bias is not None:  # bias
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        nn.init.xavier_uniform_(self.linear_out.weight)
        # nn.init.xavier_normal_(self.linear_out.weight)

        #
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.dropout_01 = nn.Dropout(0.1)
        self.dropout_02 = nn.Dropout(0.2)

        self.softmax = nn.Softmax(dim=1)

        self.layer_norm1 = nn.LayerNorm(linear_1_out, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(linear_2_out, eps=1e-6)


        return
    # end __init__

    #
    def forward(self, text_inputs, mask_input, len_seq, len_sents, tid, len_para=None, list_rels=None, mode=""):

        batch_size = text_inputs.size(0)

        #### stage1: sentence level representations
        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        sent_mask = utils.cast_type(sent_mask, FLOAT, self.use_gpu)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        avg_sents_repr = torch.zeros(batch_size, self.max_num_sents, self.base_encoder.encoder_out_size)  # averaged sents repr in the sent level encoding
        avg_sents_repr = utils.cast_type(avg_sents_repr, FLOAT, self.use_gpu)

        cur_ind = torch.zeros(batch_size, dtype=torch.int64)
        cur_ind = utils.cast_type(cur_ind, LONG, self.use_gpu)
        len_sents = utils.cast_type(len_sents, LONG, self.use_gpu)
        for sent_i in range(self.max_num_sents):
            cur_sent_lens = len_sents[:, sent_i]
            cur_max_len = int(torch.max(cur_sent_lens))
            
            if cur_max_len > 0:
                cur_sent_ids = torch.zeros(batch_size, cur_max_len, dtype=torch.int64)
                cur_sent_ids = utils.cast_type(cur_sent_ids, LONG, self.use_gpu)
                cur_mask = torch.zeros(batch_size, cur_max_len, dtype=torch.int64)
                cur_mask = utils.cast_type(cur_mask, FLOAT, self.use_gpu)

                prev_ind = cur_ind
                cur_ind = cur_ind + cur_sent_lens

                for batch_ind, sent_len in enumerate(cur_sent_lens):
                    cur_loc = cur_ind[batch_ind]
                    prev_loc = prev_ind[batch_ind]
                    cur_sent_ids[batch_ind, :cur_loc-prev_loc] = text_inputs[batch_ind, prev_loc:cur_loc]
                    cur_mask[batch_ind, :cur_loc-prev_loc] = mask_input[batch_ind, prev_loc:cur_loc]

            cur_encoded = self.base_encoder(cur_sent_ids, cur_mask, cur_sent_lens)

            encoded_sent = cur_encoded[0]  # encoded output for the current sent

            cur_sent_lens = cur_sent_lens + 1e-9 # prevent zero division
            cur_avg_repr = torch.div(torch.sum(encoded_sent, dim=1), cur_sent_lens.unsqueeze(1))

            avg_sents_repr[:, sent_i] = cur_avg_repr

        
        # encoder sentence 
        encoded_sents = avg_sents_repr
        mask_sent = torch.arange(self.max_num_sents, device=num_sents.device).expand(len(num_sents), self.max_num_sents) < num_sents.unsqueeze(1)
        mask_sent = utils.cast_type(mask_sent, BOOL, self.use_gpu)
        num_sents = utils.cast_type(num_sents, FLOAT, self.use_gpu)

        #### stage2: update sentence representations using the tree transformer
        encoded_sents, break_probs = self.tt_encoder(encoded_sents, mask_sent)  # ['features'], ['node_order'], ['adjacency_list'], ['edge_order']
        
        #### stage3: document attention
        context_weight = self.context_weight.expand(encoded_sents.shape[0], encoded_sents.shape[2], 1)
        attn_weight = torch.bmm(encoded_sents, context_weight).squeeze(2)
        attn_weight = self.tanh(attn_weight)
        attn_weight = masked_softmax(attn_weight, sent_mask)
        # attention applied
        attn_vec = torch.bmm(encoded_sents.transpose(1, 2), attn_weight.unsqueeze(2))
        ilc_vec = attn_vec.squeeze(2)


        #### FC layer
       
        fc_out = self.linear_1(ilc_vec)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_2(fc_out)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)

        fc_out = self.linear_out(fc_out)
        
        if self.output_size == 1:
            fc_out = self.sigmoid(fc_out)

        outputs = []
        outputs.append(fc_out)

        # return fc_out
        return outputs


    # end forward
