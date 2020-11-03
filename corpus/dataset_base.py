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

class Dataset_Base(Dataset):
	def __init__(self, data_id, config, pad_id):
		self.x_data = data_id["x_data"]
		self.y_label = data_id["y_label"]
		self.tid = data_id["tid"]

		self.len_para = data_id["len_para"]

		self.max_num_sents = config.max_num_sents
		self.max_len_sent = config.max_len_sent
		self.max_len_doc = config.max_len_doc
		self.pad_level = config.pad_level

		self.use_gpu = config.use_gpu

		self.pad_id = pad_id # given by w2v reader class

	#
	def __len__(self):
		raise NotImplementedError

	#
	def __getitem__(self, idx):
		raise NotImplementedError

	######
	def _pad_sent_level(self, doc_x):
		""" padding for sentence level """

		vec_text_input = torch.zeros(self.max_num_sents, self.max_len_sent, dtype=torch.int64)
		# mask_input = torch.ones(self.max_num_sents, self.max_len_sent, dtype=torch.float)
		mask_input = torch.zeros(self.max_num_sents, self.max_len_sent, dtype=torch.float)

		for ind_sent in range(len(doc_x)):
			non_padded_sent = torch.LongTensor(doc_x[ind_sent])

			pad_len = self.max_len_sent - len(non_padded_sent)
			padded_sent = F.pad(non_padded_sent, pad=(0, pad_len), mode='constant', value=self.pad_id)
			vec_text_input[ind_sent, 0:len(padded_sent)] = padded_sent

			# mask_input[ind_sent, 0:len(non_padded_sent)] = 0.0
			mask_input[ind_sent, 0:len(non_padded_sent)] = 1.0

		#
		cur_seq_len = 0
		for cur_sent in doc_x:
			cur_seq_len = cur_seq_len + len(cur_sent)

		len_sents = torch.zeros(self.max_num_sents)

		return vec_text_input, mask_input, cur_seq_len, len_sents

	#
	def _pad_doc_level(self, doc_x):
		""" padding for document level """

		flat_sents = [item for sublist in doc_x for item in sublist]
		non_padded = torch.LongTensor(flat_sents)

		vec_text_input = torch.zeros(self.max_len_doc, dtype=torch.int64)

		pad_len = self.max_len_doc - len(non_padded)
		vec_text_input = F.pad(non_padded, pad=(0, pad_len), mode='constant', value=self.pad_id)

		# mask_input = torch.ones(self.max_len_doc, dtype=torch.float)  # mask consists 0 for real tokens, 1 for padding
		# mask_input[0:len(non_padded)] = 0.0
		mask_input = torch.zeros(self.max_len_doc, dtype=torch.float)  # mask consists 0 for real tokens, 1 for padding
		mask_input[0:len(non_padded)] = 1.0

		len_seq = 0
		for cur_sent in doc_x:
			len_seq = len_seq + len(cur_sent)

		#
		len_sents = torch.zeros(self.max_num_sents)
		for ind, cur_sent in enumerate(doc_x):
			len_sents[ind] = len(cur_sent)

		# return vec_text_input, mask_input, cur_seq_len
		return vec_text_input, mask_input, len_seq, len_sents

	#
	def _pad_both(self, doc_x):

		### doc pad
		flat_sents = [item for sublist in doc_x for item in sublist]
		non_padded = torch.LongTensor(flat_sents)

		vec_doc_input = torch.zeros(self.max_len_doc, dtype=torch.int64)

		pad_len = self.max_len_doc - len(non_padded)
		vec_doc_input = F.pad(non_padded, pad=(0, pad_len), mode='constant', value=self.pad_id)

		# mask_input = torch.ones(self.max_len_doc, dtype=torch.float)  # mask consists 0 for real tokens, 1 for padding
		# mask_input[0:len(non_padded)] = 0.0
		mask_input = torch.zeros(self.max_len_doc, dtype=torch.float)  # mask consists 0 for real tokens, 1 for padding
		mask_input[0:len(non_padded)] = 1.0

		len_seq = 0
		for cur_sent in doc_x:
			len_seq = len_seq + len(cur_sent)

		# sent length
		len_sents = torch.zeros(self.max_num_sents)
		for ind, cur_sent in enumerate(doc_x):
			len_sents[ind] = len(cur_sent)

		### sent pad
		vec_sent_input = torch.zeros(self.max_num_sents, self.max_len_sent, dtype=torch.int64)
		# mask_input = torch.ones(self.max_num_sents, self.max_len_sent, dtype=torch.float)
		mask_sent_input = torch.zeros(self.max_num_sents, self.max_len_sent, dtype=torch.float)

		for ind_sent in range(len(doc_x)):
			non_padded_sent = torch.LongTensor(doc_x[ind_sent])

			pad_len = self.max_len_sent - len(non_padded_sent)
			padded_sent = F.pad(non_padded_sent, pad=(0, pad_len), mode='constant', value=self.pad_id)
			vec_sent_input[ind_sent, 0:len(padded_sent)] = padded_sent

			# mask_input[ind_sent, 0:len(non_padded_sent)] = 0.0
			mask_sent_input[ind_sent, 0:len(non_padded_sent)] = 1.0


		# return vec_text_input, mask_input, cur_seq_len
		return vec_doc_input, mask_input, len_seq, len_sents, vec_sent_input, mask_sent_input