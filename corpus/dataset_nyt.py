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

import numpy as np
import pandas as pd

import utils
from utils import LONG, FLOAT

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from corpus.dataset_base import Dataset_Base

class Dataset_NYT(Dataset_Base):
    def __init__(self, data_id, config, pad_id):

        super().__init__(data_id, config, pad_id)       

        # self.list_rels = data_id["DS_rels"]
        self.origin_score = data_id["origin_score"]

        return

    #
    def __len__(self):
        return len(self.y_label)

    #
    def __getitem__(self, idx):
        cur_x = self.x_data[idx]
        cur_y = self.y_label[idx]
        cur_tid = self.tid[idx]
        
        # cur_len_para = self.len_para[idx]
        # cur_list_rels = [self.list_rels[idx]]  #
        # print(cur_list_rels)
        
        cur_origin_score = self.origin_score[idx]

        vec_text_input = None
        len_sents = None
        if self.pad_level == "sent" or self.pad_level == "sentence":
            vec_text_input, mask_input, seq_lens = self._pad_sent_level(cur_x)  # depreciated, not feasible practically
        else:
            vec_text_input, mask_input, len_seq, len_sents = self._pad_doc_level(cur_x)

        ## both of doc and sent
        # vec_doc_input, mask_input, len_seq, len_sents, vec_sent_input, mask_sent_input = self._pad_both(cur_x)

        label_y = torch.LongTensor([cur_y])


        return vec_text_input, label_y, mask_input, len_seq, len_sents, cur_tid, cur_origin_score
        # return vec_doc_input, label_y, mask_input, len_seq, len_sents, cur_tid, vec_sent_input, mask_sent_input, cur_origin_score

