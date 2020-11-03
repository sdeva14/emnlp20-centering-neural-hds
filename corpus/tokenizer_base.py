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

import os

import nltk
# from pytorch_pretrained_bert.tokenization import BertTokenizer

# from pytorch_transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer
from transformers import BertTokenizer,XLNetTokenizer, T5Tokenizer #BartTokenizer

#
class Tokenizer_Base(object):

	def __init__(self, config):
		super().__init__()
		
		return
	# end __init__

		#
	def get_tokenizer(self, config):
		tokenizer = None
		# if not configured, then no need to assign
		if config.tokenizer_type.startswith('word'):
			tokenizer = nltk.word_tokenize
		elif config.tokenizer_type.startswith('bert-'):
			tokenizer = BertTokenizer.from_pretrained(config.tokenizer_type, do_lower_case=True)
		elif config.tokenizer_type.startswith('xlnet'):
			tokenizer = XLNetTokenizer.from_pretrained(config.tokenizer_type, do_lower_case=True)
		elif config.tokenizer_type.startswith('t5-'):
			tokenizer = T5Tokenizer.from_pretrained(config.tokenizer_type, do_lower_case=True)
		elif config.tokenizer_type.startswith('bart-'):
			tokenizer = BartTokenizer.from_pretrained(config.tokenizer_type, do_lower_case=True)

		return tokenizer

