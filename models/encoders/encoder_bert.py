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

# from pytorch_transformers import XLNetConfig, XLNetModel  # old-version
# from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
from transformers import BertTokenizer, BertModel

class Encoder_BERT(nn.Module):

    def __init__(self, config, x_embed):
        super().__init__()

        pretrained_weights = "bert-base-uncased"
        self.model = BertModel.from_pretrained(pretrained_weights)
        self.encoder_out_size = 768

        return
    # end __init__

    #
    def forward(self, text_inputs, mask_input, len_seq, mode=""):
        encoder_out = []
        self.model.eval()

        with torch.no_grad():
            encoder_out = self.model(text_inputs, attention_mask=mask_input)[0]
            encoder_out = encoder_out * mask_input.unsqueeze(2)

        return encoder_out
