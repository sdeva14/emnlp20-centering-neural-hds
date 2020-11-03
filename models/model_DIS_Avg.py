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

class Model_DIS_Avg(models.model_base.BaseModel):
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
        
        #####################
        fc_in_size = self.base_encoder.encoder_out_size

        linear_1_out = fc_in_size // 2
        linear_2_out = linear_1_out // 2

        self.linear_1 = nn.Linear(fc_in_size, linear_1_out)
        nn.init.xavier_uniform_(self.linear_1.weight)

        self.linear_2 = nn.Linear(linear_1_out, linear_2_out)
        nn.init.xavier_uniform_(self.linear_2.weight)

        self.linear_out = nn.Linear(linear_2_out, self.output_size)
        if corpus_target.output_bias is not None:  # if a bias is given
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        nn.init.xavier_uniform_(self.linear_out.weight)

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
   
    # return sentence representation by averaging of all words
    def sent_repr_avg(self, batch_size, encoder_out, len_sents):
        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        sent_repr = torch.zeros(batch_size, self.max_num_sents, self.encoder_coh.encoder_out_size)
        sent_repr = utils.cast_type(sent_repr, FLOAT, self.use_gpu)
        for cur_ind_doc in range(batch_size):
            list_sent_len = len_sents[cur_ind_doc]
            cur_sent_num = int(num_sents[cur_ind_doc])
            cur_loc_sent = 0
            list_cur_doc_sents = []

            for cur_ind_sent in range(cur_sent_num):
                cur_sent_len = int(list_sent_len[cur_ind_sent])
                # cur_local_words = local_output_words[cur_batch, cur_ind_sent:end_sent, :]
                
                # cur_sent_repr = encoder_out[cur_ind_doc, cur_loc_sent+cur_sent_len-1, :]  # pick the last representation of each sentence
                cur_sent_repr = torch.div(torch.sum(encoder_out[cur_ind_doc, cur_loc_sent:cur_loc_sent+cur_sent_len], dim=0), cur_sent_len)  # avg version
                cur_sent_repr = cur_sent_repr.view(1, 1, -1)  # restore to (1, 1, xrnn_cell_size)
                
                list_cur_doc_sents.append(cur_sent_repr)
                cur_loc_sent = cur_loc_sent + cur_sent_len

            # end for cur_len_sent

            cur_sents_repr = torch.stack(list_cur_doc_sents, dim=1)  # (batch_size, num_sents, rnn_cell_size)
            cur_sents_repr = cur_sents_repr.squeeze(2)  # not needed when the last repr is used

            sent_repr[cur_ind_doc, :cur_sent_num, :] = cur_sents_repr
        # end for cur_doc

        return sent_repr

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
        mask_sent = torch.arange(self.max_num_sents, device=num_sents.device).expand(len(num_sents), self.max_num_sents) < num_sents.unsqueeze(1)
        mask_sent = utils.cast_type(mask_sent, BOOL, self.use_gpu)
        num_sents = utils.cast_type(num_sents, FLOAT, self.use_gpu)
        encoded_sents = avg_sents_repr

        #### Stage2: Avg
        ilc_vec = torch.div(torch.sum(encoded_sents, dim=1), num_sents.unsqueeze(1))


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
