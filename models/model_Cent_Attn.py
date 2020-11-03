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
import logging
import math

import networkx as nx
import collections

from models.encoders.encoder_main import Encoder_Main

import models.model_base
from models.model_base import masked_softmax

import utils
from utils import FLOAT, LONG, BOOL

import models.stru_trans.attention as tt_attn
import models.stru_trans.models as tt_model
import models.stru_trans.modules as tt_module
import copy

# from apex.normalization.fused_layer_norm import FusedLayerNorm

logger = logging.getLogger()

class Coh_Model_Cent_Attn(models.model_base.BaseModel):
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

        self.dropout_rate = config.dropout
        self.output_size = config.output_size  # the number of final output class

        self.use_gpu = config.use_gpu
        self.gen_logs = config.gen_logs

        if not hasattr(config, "freeze_step"):
            config.freeze_step = 5000

        self.output_attentions = config.output_attentions  # flag for huggingface impl

        self.topk_fwr = config.topk_fwr
        self.threshold_sim = config.threshold_sim
        # self.topk_back = config.topk_back
        self.topk_back = 1

        ########
        #
        self.base_encoder = Encoder_Main(config, embReader)

        #
        self.sim_cosine_d0 = torch.nn.CosineSimilarity(dim=0)
        self.sim_cosine_d1 = torch.nn.CosineSimilarity(dim=1)
        self.sim_cosine_d2 = torch.nn.CosineSimilarity(dim=2)

        ## tree-transformer
        c = copy.deepcopy
        num_heads = 4
        N=4  # num of layers
        d_model=self.base_encoder.encoder_out_size
        d_ff=self.base_encoder.encoder_out_size 
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

        # self.layer_norm1 = nn.LayerNorm(linear_1_out, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(linear_2_out, eps=1e-6)

        return
    # end __init__

#### Functions
#####################################################################################

    def sent_repr_avg(self, batch_size, encoder_out, len_sents):
        """return sentence representation by averaging of all words."""

        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        sent_repr = torch.zeros(batch_size, self.max_num_sents, self.base_encoder.encoder_out_size)
        sent_repr = utils.cast_type(sent_repr, FLOAT, self.use_gpu)
        for cur_ind_doc in range(batch_size):
            list_sent_len = len_sents[cur_ind_doc]
            cur_sent_num = int(num_sents[cur_ind_doc])
            cur_loc_sent = 0
            list_cur_doc_sents = []

            for cur_ind_sent in range(cur_sent_num):
                cur_sent_len = int(list_sent_len[cur_ind_sent])
                
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
    # end def sent_repr_avg

    def get_fwrd_centers(self, text_inputs, mask_input, len_sents):
        """ Determine fowrard-looking centers using an attention matrix in a PLM """
        batch_size = text_inputs.size(0)

        fwrd_repr = torch.zeros(batch_size, self.max_num_sents, self.topk_fwr, self.base_encoder.encoder_out_size)
        fwrd_repr = utils.cast_type(fwrd_repr, FLOAT, self.use_gpu)

        avg_sents_repr = torch.zeros(batch_size, self.max_num_sents, self.base_encoder.encoder_out_size)  # averaged sents repr in the sent level encoding
        avg_sents_repr = utils.cast_type(avg_sents_repr, FLOAT, self.use_gpu)

        batch_cp_ind = torch.zeros(batch_size, self.max_num_sents)  # only used for manual analysis later
        batch_cp_ind = utils.cast_type(batch_cp_ind, LONG, self.use_gpu)

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

                # encode each sentence
                cur_encoded = self.base_encoder(cur_sent_ids, cur_mask, cur_sent_lens)

                encoded_sent = cur_encoded[0]  # encoded output for the current sent
                attn_sent = cur_encoded[1]  # averaged attention for the current sent

                ## filter out: we do not consider special tokens and punctation as a center; <.>, <sep>, and <cls>
                list_diag = []
                for batch_ind, cur_mat in enumerate(attn_sent):
                    cur_diag = torch.diag(cur_mat, diagonal=0)

                    ## masking as the length of each sentence
                    cur_batch_sent_len = int(cur_sent_lens[batch_ind])  # i th sentence with batch
                    if cur_batch_sent_len > 3:
                        cur_diag[cur_batch_sent_len-3:] = 0  # also remove puntation
                    else:
                        cur_diag[cur_batch_sent_len-2:] = 0  # only remove the special tokens
                    list_diag.append(cur_diag)
                # end for
                attn_diag = torch.stack(list_diag)  # because torch.daig does not support batch

                temp_fwr_centers, fwr_sort_ind = torch.sort(attn_diag, dim=1, descending=True)  # forward centers are selected by attn
                temp_fwr_centers = temp_fwr_centers[:, :self.topk_fwr]

                batch_cp_ind[:, sent_i] = fwr_sort_ind[:, 0]  # only consider the top-1 item for Cp

                # # non-batched selecting by indices
                # temp = []
                # for batch_ind, cur_fwr in enumerate(fwr_centers):
                #     cur_fwrd_repr = encoded_sent[batch_ind].index_select(0, cur_fwr)
                #     temp.append(cur_fwrd_repr)
                # cur_fwrd_repr = torch.stack(temp)
                # fwrd_repr[:, sent_i] = cur_fwrd_repr

                # batched version selecting by indices
                fwr_centers = torch.zeros(batch_size, self.topk_fwr)
                fwr_centers = utils.cast_type(fwr_centers, LONG, self.use_gpu)
                fwr_centers[:, :temp_fwr_centers.size(1)] = temp_fwr_centers  # to handle execeptional case when the sent is shorter than topk

                selected = encoded_sent.gather(1, fwr_centers.unsqueeze(-1).expand(batch_size, self.topk_fwr, self.base_encoder.encoder_out_size))
                
                fwrd_repr[:, sent_i, :fwr_centers.size(1)] = selected

                cur_sent_lens = cur_sent_lens + 1e-9 # prevent zero division
                cur_avg_repr = torch.div(torch.sum(encoded_sent, dim=1), cur_sent_lens.unsqueeze(1))
                
                avg_sents_repr[:, sent_i] = cur_avg_repr
            # end if
        # end for sent_i

        return avg_sents_repr, fwrd_repr, batch_cp_ind
    # end def get_fwrd_centers

    def get_back_centers(self, avg_sents_repr, fwrd_repr):
        """ Determine backward-looking centers"""
        batch_size = avg_sents_repr.size(0)
        back_repr = torch.zeros(batch_size, self.max_num_sents, self.topk_back, self.base_encoder.encoder_out_size)
        back_repr = utils.cast_type(back_repr, FLOAT, self.use_gpu)

        for sent_i in range(self.max_num_sents):
            if sent_i == 0 or sent_i == self.max_num_sents-1:
                # there is no backward center in the first sentence
                continue
            # end if
            else:
                prev_fwrd_repr = fwrd_repr[:, sent_i-1, :, :]  # (batch_size, topk_fwrd, dim)
                cur_fwrd_repr = fwrd_repr[:, sent_i, :, :]  # (batch_size, topk_fwrd, dim)
                cur_sent_repr = avg_sents_repr[:, sent_i, :]  # (batch_size, dim)

                sim_rank = self.sim_cosine_d2(prev_fwrd_repr, cur_sent_repr.unsqueeze(1))
                
                max_sim_val, max_sim_ind = torch.max(sim_rank, dim=1)
                
                idx = max_sim_ind.view(-1, self.topk_back, 1).expand(max_sim_ind.size(0), self.topk_back, self.base_encoder.encoder_out_size)
                cur_back_repr = prev_fwrd_repr.gather(1, idx)

                back_repr[:, sent_i] = cur_back_repr                

                # end for topk_i
            # end else

        # end for sent_i

        return back_repr
    # end def get_back_centers

    def get_disco_seg(self, cur_sent_num, ind_batch, fwrd_repr, cur_batch_repr):
        """ construct hierarchical discourse segments """

        cur_seg_list = []  # current segment
        cur_seg_ind = 0
        stack_focus = []  # stack representing focusing

        seg_map = dict()
        adj_list = []  # adjacency list
        list_root_ds = []  # list up the first level segments

        for sent_i in range(cur_sent_num):
            cur_pref_repr = fwrd_repr[ind_batch, sent_i, 0, :]
            cur_pref_repr = cur_pref_repr.unsqueeze(0)
            cur_seg_list = cur_seg_list + [sent_i]

            # for the first two sentences, skip them to make a initial stack
            if sent_i  < 2:
                # for the first and the second sent, just push
                continue
            # handle the last sentence
            elif sent_i == cur_sent_num-1:
                if len(stack_focus) < 1:
                    list_root_ds.append(cur_seg_ind)
                else:
                    top_seg_stack = stack_focus[-1]
                    adj_pair = (top_seg_stack, cur_seg_ind)
                    adj_list.append(adj_pair)
                
                seg_map[cur_seg_ind] = cur_seg_list
                stack_focus.append(cur_seg_ind)
            # end if
            else:
                cur_back_repr = cur_batch_repr[sent_i, :, :]

                isCont = False
                # while len(stack_focus) > 0 and stack_focus[-1]!=0:       
                while len(stack_focus) > 0:                       
                    # consider average of sentences included in the top segment on the stack
                    top_seg_stack = stack_focus[-1]
                    cur_sent_stack = seg_map[top_seg_stack]
                    prev_repr = cur_batch_repr[cur_sent_stack, :, :]  
                    prev_back_repr = torch.div(torch.sum(prev_repr, dim=0), len(cur_sent_stack))

                    # calcualte the similarity between backward
                    sim_back_vec = self.sim_cosine_d1(prev_back_repr, cur_back_repr)
                    sim_avg = torch.div(torch.sum(sim_back_vec, dim=0), sim_back_vec.size(0))
                    
                    # similarity between the current backward and the prefered in the precedding sentence
                    sim_back_pref = self.sim_cosine_d1(cur_back_repr, cur_pref_repr)
                   
                    # if we find a place either continue or retain, then stop the loop
                    if sim_avg > self.threshold_sim:  # stack the item, and move to the next sentence
                        # if sim_back_fwrd > self.threshold_sim:  ## continuing
                        if sim_back_pref > self.threshold_sim:  ## continuing
                            isCont = True
                        else:  ## retaining
                            # push the current segment
                            isCont = False
                            # update stack and segment map
                            if len(stack_focus) < 1:
                                list_root_ds.append(cur_seg_ind)
                            else:
                                top_seg_stack = stack_focus[-1]
                                adj_pair = (top_seg_stack, cur_seg_ind)
                                adj_list.append(adj_pair)
                            
                            seg_map[cur_seg_ind] = cur_seg_list
                            stack_focus.append(cur_seg_ind)

                            cur_seg_ind += 1
                            cur_seg_list = []
                        break                            
                    # shifting: pop the top item in the stack, and iterate to find the location
                    else:  
                        del stack_focus[-1]  # pop the top segment
                        isCont = False
                # end while len(stack_focus)

                if ~isCont and len(stack_focus) < 1:
                    # when loop is over because pop everyting
                        stack_focus.append(cur_seg_ind)
                        seg_map[cur_seg_ind] = cur_seg_list
                        list_root_ds.append(cur_seg_ind)

                        cur_seg_ind += 1
                        cur_seg_list = []
                    # end if
                # end if
            # end else
        # end for sent_i

        return seg_map, adj_list, list_root_ds
    # end get_disco_seg

    def make_tree_stru(self, seg_map, adj_list, list_root_ds):
        """ make a tree structure using the structural information """
        cur_tree = nx.DiGraph()  # tree structure for current document

        # consider root first
        for i in list_root_ds:
            cur_root_seg = seg_map[i]
            cur_tree.add_node(cur_root_seg[0])  # add the first item of segments in the root level

        # connect the first item of each segment in the root level
        for i in range(len(list_root_ds)):
            for j in range(i+1, len(list_root_ds)):
                cur_root_pair = (list_root_ds[i], list_root_ds[j])
                # adj_list.append(cur_root_pair)
                
                src_seg = seg_map[cur_root_pair[0]]
                dst_seg = seg_map[cur_root_pair[1]]        
                cur_tree.add_edge(src_seg[0], dst_seg[0])  # connect the first item of segments
        
        # connect sentences each other within intra segment
        for cur_seg, sents_seg in seg_map.items():
            if len(sents_seg) > 1:
                for i in range(len(sents_seg)-1):
                    cur_tree.add_edge(sents_seg[i], sents_seg[i+1])

        # then between segments
        for cur_pair in adj_list:
            src_seg = seg_map[cur_pair[0]]
            dst_seg = seg_map[cur_pair[1]]

            cur_tree.add_edge(src_seg[0], dst_seg[0])  # first sentence version

        # connect between siblings
        for cur_root in list_root_ds:
            childs = nx.descendants(cur_tree, cur_root)
            for cur_child in childs:
                siblings = list(cur_tree.successors(cur_child))
                if len(siblings) > 1:

                    for i in range(len(siblings)):
                        for j in range(i+1, len(siblings)):
                            cur_tree.add_edge(siblings[i], siblings[j])

        return cur_tree
    # end def make_tree_stru

    #
    def centering_attn(self, text_inputs, mask_input, len_sents, num_sents, tid):

        ## Parser stage1: determine foward-looking centers and preferred centers
        avg_sents_repr, fwrd_repr, batch_cp_ind = self.get_fwrd_centers(text_inputs, mask_input, len_sents)

        ## Parser stage2: decide backward center
        back_repr = self.get_back_centers(avg_sents_repr, fwrd_repr)

        ## Parser stage3: construct hierarchical discourse segments
        batch_segMap = []
        batch_adj_mat = []
        batch_adj_list = []
        batch_root_list = []
        for ind_batch, cur_batch_repr in enumerate(back_repr):
            cur_sent_num = int(num_sents[ind_batch])

            ## Parser stage3-1: get structural information
            seg_map, adj_list, list_root_ds = self.get_disco_seg(cur_sent_num, ind_batch, fwrd_repr, cur_batch_repr)

            ## Parser stage3-2: make a tree structure using the information
            cur_tree = self.make_tree_stru(seg_map, adj_list, list_root_ds)

            ## Parser stage3-3: make a numpy array from networkx tree
            cur_adj_mat = np.zeros((self.max_num_sents, self.max_num_sents))
            undir_tree = cur_tree.to_undirected()  # we make an undirected tree
            np_adj_mat = nx.to_numpy_matrix(undir_tree)

            cur_adj_mat[:np_adj_mat.shape[0], :np_adj_mat.shape[1]]=np_adj_mat

            ## store structures for statistical analysis
            batch_adj_mat.append(cur_adj_mat)
            batch_adj_list.append(adj_list)
            batch_root_list.append(list_root_ds)
            batch_segMap.append(list(seg_map.items()) )

        # end for ind_batch

        # structural information which will be passed to structure-aware transformer
        adj_mat = torch.from_numpy(np.array(batch_adj_mat))
        adj_mat = utils.cast_type(adj_mat, FLOAT, self.use_gpu)
        batch_cp_ind = batch_cp_ind.tolist()

        return  adj_mat, avg_sents_repr, batch_adj_list, batch_root_list, batch_segMap, batch_cp_ind

#### Forward function
#####################################################################################

    #
    def forward(self, text_inputs, mask_input, len_seq, len_sents, tid, len_para=None, list_rels=None, mode=""):
        # mask_input: (batch, max_tokens), len_sents: (batch, max_num_sents)
        batch_size = text_inputs.size(0)

        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        sent_mask = utils.cast_type(sent_mask, FLOAT, self.use_gpu)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)

        #### Stage1 and 2: sentence repr and discourse segments parser
        adj_mat, sent_repr, batch_adj_list, batch_root_list, batch_segMap, batch_cp_ind = self.centering_attn(text_inputs, mask_input, len_sents, num_sents, tid)

        encoder_doc_out = self.base_encoder(text_inputs, mask_input, len_seq)
        encoded_doc = encoder_doc_out[0]
        if self.output_attentions:
            attn_doc_avg = encoder_doc_out[1]  # averaged mh attentions (batch, item, item)

        #### doc-level encoding input text (disable the below 4 lines if GPU memory is not enough)
        sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
        sent_mask = utils.cast_type(sent_mask, FLOAT, self.use_gpu)
        num_sents = sent_mask.sum(dim=1)  # (batch_size)
        sent_repr = self.sent_repr_avg(batch_size, encoded_doc, len_sents)

        #### Stage3: Structure-aware transformer
        mask_sent = torch.arange(self.max_num_sents, device=num_sents.device).expand(len(num_sents), self.max_num_sents) < num_sents.unsqueeze(1)
        mask_sent = utils.cast_type(mask_sent, BOOL, self.use_gpu)
        encoded_sents, break_probs = self.tt_encoder(sent_repr, mask_sent, adj_mat)  # ['features'], ['node_order'], ['adjacency_list'], ['edge_order']

        #### Stage4: Document Attention
        context_weight = self.context_weight.expand(encoded_sents.shape[0], encoded_sents.shape[2], 1)
        attn_weight = torch.bmm(encoded_sents, context_weight).squeeze(2)
        attn_weight = self.tanh(attn_weight)
        attn_weight = masked_softmax(attn_weight, sent_mask)
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

        # prepare output to return
        outputs = []
        outputs.append(fc_out)

        if self.gen_logs:
            outputs.append(batch_adj_list)
            outputs.append(batch_root_list)
            outputs.append(batch_segMap)
            outputs.append(batch_cp_ind)
            outputs.append(num_sents.tolist())

        # return fc_out
        return outputs
    # end def forward

# end class