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
import torch
import numpy as np
# import torch.distributions.normal as normal
import logging

import w2vEmbReader

from models.encoders.encoder_main import Encoder_Main
from models.encoders.encoder_rnn import Encoder_RNN
from models.encoders.StructuredAttention import StructuredAttention

import models.model_base
import utils
from utils import FLOAT, LONG


class Model_Latent_Doc_Stru(models.model_base.BaseModel):
    """ class for TACL18 implementation
        Title: Learning Structured Text Representations
        Ref: https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00005
    """
    def __init__(self, config, corpus_target, embReader):
        super().__init__(config)

        ####
        # init parameters
        self.corpus_target = config.corpus_target
        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
        self.batch_size = config.batch_size

        self.vocab = corpus_target.vocab  # word2id
        self.rev_vocab = corpus_target.rev_vocab  # id2word
        self.vocab_size = len(self.vocab)
        self.pad_id = corpus_target.pad_id
        self.num_special_vocab = corpus_target.num_special_vocab

        self.embed_size = config.embed_size
        self.dropout_rate = config.dropout
        self.path_pretrained_emb = config.path_pretrained_emb
        self.num_layers = 1
        self.output_size = config.output_size  # the number of final output class
        self.pad_level = config.pad_level

        self.use_gpu = config.use_gpu
        
        if not hasattr(config, "freeze_step"):
            config.freeze_step = 5000

        config.rnn_bidir = True ## fix bi-dir to follow original paper of NAACL19
        if config.rnn_bidir:
            self.sem_dim_size = 2 * config.sem_dim_size
        else:
            self.sem_dim_size = config.sem_dim_size

        self.rnn_cell_size = config.rnn_cell_size

        self.pooling_sent = config.pooling_sent  # max or avg
        self.pooling_doc = config.pooling_doc  # max or avg

        ####
        self.encoder_base = Encoder_Main(config, embReader)
        config.rnn_bidir = False
        self.encoder_sent = Encoder_RNN(config, embReader, self.rnn_cell_size*2, self.rnn_cell_size*2)
        config.rnn_bidir = True
        self.structure_att = StructuredAttention(config)

        #
        fc_in_size = self.encoder_base.encoder_out_size
        linear_1_out = fc_in_size // 2
        linear_2_out = linear_1_out // 2


        self.linear_out = nn.Linear(self.sem_dim_size, self.output_size)
        if corpus_target.output_bias is not None:  # bias
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        # nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.xavier_normal_(self.linear_out.weight)

        #
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.leak_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.softmax = nn.Softmax(dim=1)

        return

    # end __init__
   

    def _pooling_avg(self, encoded_sentences, mask, len_seq_sent):
        mask = mask.unsqueeze(3).repeat(1, 1, 1, encoded_sentences.size(3))
        encoded_sentences = encoded_sentences * mask
        len_seq_sent = utils.cast_type(len_seq_sent, FLOAT, self.use_gpu)
        encoded_sentences = torch.div(torch.sum(encoded_sentences, dim=2), len_seq_sent.unsqueeze(2))

        return encoded_sentences, mask

    def _pooling_sent_max(self, encoded_sentences, mask, target_dim):
        mask = ((mask - 1) * 999).unsqueeze(target_dim+1).repeat(1, 1, 1, encoded_sentences.size(target_dim+1))
        encoded_sentences = encoded_sentences + mask
        encoded_sentences = encoded_sentences.max(dim=target_dim)[0]  # Batch * sent * dim

        return encoded_sentences, mask

    def _pooling_doc_max(self, encoded_docs, mask, target_dim):
        mask = utils.cast_type(mask, FLOAT, self.use_gpu)

        mask = ((mask - 1) * 999).unsqueeze(target_dim + 1).repeat(1, 1, encoded_docs.size(target_dim + 1))
        encoded_docs = encoded_docs + mask
        encoded_docs = encoded_docs.max(dim=target_dim)[0]  # Batch * sent * dim

        return encoded_docs, mask


    # return sentence representation by averaging of all words
    def sent_repr_avg(self, batch_size, encoder_out, len_sents):
        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        sent_repr = torch.zeros(batch_size, self.max_num_sents, self.encoder_base.encoder_out_size)
        sent_repr = utils.cast_type(sent_repr, FLOAT, self.use_gpu)
        for cur_ind_doc in range(batch_size):
            list_sent_len = len_sents[cur_ind_doc]
            cur_sent_num = int(num_sents[cur_ind_doc])
            cur_loc_sent = 0
            list_cur_doc_sents = []

            for cur_ind_sent in range(cur_sent_num):
                # cur_sent_len = cur_sent_len + 1e-9 # prevent zero division
                cur_sent_len = int(list_sent_len[cur_ind_sent])

                # cur_local_words = local_output_words[cur_batch, cur_ind_sent:end_sent, :]
                
                # cur_sent_repr = encoder_out[cur_ind_doc, cur_loc_sent+cur_sent_len-1, :]  # pick the last representation of each sentence
                cur_sent_repr = torch.div(torch.sum(encoder_out[cur_ind_doc, cur_loc_sent:cur_loc_sent+cur_sent_len], dim=0), cur_sent_len + 1e-9)  # avg version
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

        #### word level representations
        encoder_out = self.encoder_base(text_inputs, mask_input, len_seq)

        #### make sentence representations
        batch_size = text_inputs.size(0)

        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        sent_mask = utils.cast_type(sent_mask, FLOAT, self.use_gpu)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        ### sentence level representations
        sent_repr = self.sent_repr_avg(batch_size, encoder_out, len_sents)

        # do not consider sentence level attention in this implementation
        encoded_sentences = self.encoder_sent.forward_skip(sent_repr, sent_mask, num_sents)

        # structured attention for document level
        encoded_documents, doc_attention_matrix = self.structure_att(encoded_sentences)  # get structured attn for sent

        # pooling for sentence
        if self.pooling_sent.lower() == "avg":
            encoded_documents, mask = self._pooling_avg(encoded_documents, sent_mask, len_sent_seq)
        elif self.pooling_sent.lower() == "max":
            encoded_documents, mask = self._pooling_doc_max(encoded_documents, sent_mask, 1)

        ## Fully Connected layers
        fc_out = self.linear_out(encoded_documents)

        if self.corpus_target.lower() == "asap":
            fc_out = self.sigmoid(fc_out)

        model_outputs = []
        model_outputs.append(fc_out)

        # return fc_out
        return model_outputs
    # end forward
