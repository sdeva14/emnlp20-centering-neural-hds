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


import torch
import torch.nn as nn

import utils
from utils import LONG, FLOAT

from models.transformer.Models import Encoder as Transf_Encoder

class Encoder_Transfomer(nn.Module):
    """ encoders class """

    #
    def __init__(self, config, x_embed):
        super().__init__()

        self.use_gpu = config.use_gpu

        len_max_seq = 0
        if config.pad_level == "doc":
            len_max_seq = config.max_len_doc
        else:
            len_max_seq = config.max_len_sent

        self.d_model = config.embed_size

        self.model = Transf_Encoder(
            x_embed=x_embed.x_embed, n_src_vocab=config.max_vocab_cnt, len_max_seq=len_max_seq,
            d_word_vec=config.embed_size, d_model=self.d_model, d_inner=config.d_inner_hid,
            n_layers=config.transf_n_layers, n_head=config.n_head, d_k=config.d_k, d_v=config.d_v,
            dropout=config.dropout
        )

        #self.encoder_out_size = config.d_model
        self.encoder_out_size = self.d_model

        return
    # end def __init__

    # generate positional input consists of order
    def gen_positional_input(self, seq_x):
        pos_x = []
        for cur_batch in seq_x:
            cur_pos = []
            for ind, val in enumerate(cur_batch):
                if val != 0:
                    cur_pos.append(ind+1)
                else:
                    cur_pos.append(0)
            pos_x.append(cur_pos)

        pos_input = torch.LongTensor(pos_x)
        pos_input = utils.cast_type(pos_input, LONG, self.use_gpu)

        return pos_input
    # end def_gen_positional_input

    #
    def forward(self, text_inputs, mask_input, len_seq, mode=""):
        pos_x = self.gen_positional_input(text_inputs)
        encoder_out, *_ = self.model(text_inputs, pos_x)

        encoder_out = encoder_out * mask_input.unsqueeze(2)

        return encoder_out
    # end forward

