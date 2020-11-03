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
from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
# from transformers import BertTokenizer, BertModel

class Encoder_XLNet(nn.Module):

    def __init__(self, config, x_embed):
        super().__init__()

        # pretrained_weights = "xlnet-base-cased"
        self.output_attentions = config.output_attentions
        self.model = XLNetModel.from_pretrained(config.pretrained_weights, output_attentions=self.output_attentions)
        self.pretrained_config = XLNetConfig.from_pretrained(config.pretrained_weights)
        self.encoder_out_size = self.model.config.d_model

        return
    # end __init__

    #
    def forward(self, text_inputs, mask_input, len_seq, mode=""):
        encoder_out = []
        self.model.eval()

        with torch.no_grad():
            model_output = self.model(text_inputs, attention_mask=mask_input)
            encoded = model_output[0] * mask_input.unsqueeze(2)

            encoder_out.append(encoded)

            if self.output_attentions:
                attn = model_output[1][-1] # only consider mh attentions from the last layer (batch, mh, item, item)
                attn_avg = torch.div(torch.sum(attn, dim=1), attn.shape[1])  # average all mh
                attn_avg = attn_avg * mask_input.unsqueeze(2)  # masking for actual input
                encoder_out.append(attn_avg)  # (batch, item, item)  

        return encoder_out


    #
    def forward_iter(self, raw_sent_inputs, mask_input, len_seq, mode=""):
        # raw_sent_inputs: (batch, sent_num, max_sent_len)
        # return: (batch, sent_num, dim_xlnet)

        encoder_out = []
        self.model.eval()

        encoded_sents = []
        attn_sents = []
        with torch.no_grad():  
            # print(text_inputs)

            for sent_i in range(raw_sent_inputs.shape[1]):
                cur_inputs = raw_sent_inputs[:, sent_i, :].squeeze(1)  # (batch, max_sent_len)
                cur_mask = mask_input[:, sent_i, :].squeeze(1)  # 

                print(cur_inputs.shape)
                print(cur_mask.shape)

                model_output = self.model(cur_inputs, attention_mask=cur_mask)  # (batch, dim_xlnet)

                encoded_sents.append(model_output)

                if self.output_attentions:
                    attn = model_output[1][-1] # only consider mh attentions from the last layer (batch, mh, item, item)
                    attn_avg = torch.div(torch.sum(attn, dim=1), attn.shape[1])  # average all mh
                    attn_sents.append(attn_avg)

            # end for sent_i

            encoded = torch.stack(encoded_sents)
            encoded = encoded.transpose(1, 0, 2, 3)
            encoded = model_output[0] * mask_input.unsqueeze(3)

            encoder_out.append(encoded)
            if self.output_attentions:            
                attn_out = torch.stack(attn_sents)
                attn_out = attn_out.transpose(1, 0, 2, 3)
                encoder_out.append(attn_out)  # (batch, item, item)  


        return encoder_out  # [0]: encoded_out, [1]: averaged_attn


    #
    def forward_skip(self, x_input, mask, len_seq, mode=""):
        # ''' skip embedding part when embedded input is given '''
        encoder_out = x_input

        return encoder_out
    # end forward