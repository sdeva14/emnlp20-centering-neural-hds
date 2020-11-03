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

from __future__ import unicode_literals  # at top of module

import os
import logging
import re
import string
from collections import Counter
from collections import namedtuple
import statistics

import numpy as np
import pandas as pd
import math

from scipy.stats import entropy
from scipy.stats.mstats import gmean
from math import log, e

import nltk

from corpus.tokenizer_base import Tokenizer_Base
import sentencepiece as spm

from transformers import BertConfig, XLNetConfig, T5Config#, BartConfig

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
SEP = "|"
TIME= '<time>'
DATE = '<date>'

# import spacy
# spacy_nlp = spacy.load('en')
# import nltk.tokenize.punkt

class CorpusBase(object):
    """ Corpus class for base """

    #
    def __init__(self, config):
        super(CorpusBase, self).__init__()
        self.config = config

        self.corpus_target = config.corpus_target
        self.tokenizer_type = config.tokenizer_type # nltk or bert-base-uncased

        self.vocab = None  # will be assigned in "_build_vocab" i.e., word2ind
        self.rev_vocab = None  # will be assigned in "_build_vocab" i.e., ind2word
        self.pad_id = 0  # default 0, will be re-assigned depending on tokenizer
        self.unk_id = None  # will be assigned in "_build_vocab"
        self.bos_id = None  # will be assigned in "_build_vocab"
        self.eos_id = None  # will be assigned in "_build_vocab"
        self.time_id = None
        self.vocab_count = -1  # will be assigned in "_build_vocab"
        self.num_special_vocab = None  # number of used additional vocabulary, e.g., PAD, UNK, BOS, EOS

        self.train_corpus = None  # will be assigned in "read_kfold"
        self.valid_corpus = None  # will be assigned in "read_kfold"
        self.test_corpus = None  # will be assigned in "read_kfold"

        self.fold_train = []  # (num_fold, structured_train), # will be assigned in "read_kfold"
        self.fold_test = []  # (num_fold, structured_test), # will be assigned in "read_kfold"
        self.cur_fold_num = -1  #

        self.max_num_sents = -1  # maximum number of sentence in document given corpus, will be assigned in "_read_dataset"
        self.max_len_sent = -1  # maximum length of sentence given corpus, will be assigned in "_read_dataset"
        self.max_len_doc = -1  # maximum length of documents (the number of words), will be assigned in "_read_dataset"

        self.max_num_para = -1  # maximum number of paragraphs

        self.output_bias = None

        self.keep_pronoun = config.keep_pronoun
        self.remove_stopwords = config.remove_stopwords
        self.stopwords = []

        # get tokenizer
        tokenizer_class = Tokenizer_Base(config)
        self.tokenizer = tokenizer_class.get_tokenizer(config)
        self.use_paragraph = config.use_paragraph

        # sentence splitter
        self.sent_tokenzier = nltk.sent_tokenize  # nltk sent tokenizer

        # stopwords (not used)
        self._make_stopwords()

    ##########################

    #
    def set_cur_fold_num(self, cur_fold_num):
        self.cur_fold_num = cur_fold_num
        return

    #
    def get_id_corpus(self, num_fold=-1):
        raise NotImplementedError

    #
    def _tokenize_corpus(self, pd_input):
        raise NotImplementedError

    #
    def _read_dataset(self, config):
        raise NotImplementedError

    #
    def generate_kfold(self, config, seed):
        raise NotImplementedError

    #
    def read_kfold(self, config):
        raise NotImplementedError

    #
    def is_time(self, token):
        is_time = False
        if bool(time_regex1.match(token)): is_time = True
        elif bool(time_regex2.match(token)): is_time = True

        return is_time

    #
    def is_date(self, token):
        is_date = False
        if bool(date_regex1.match(token)): is_date = True
        elif bool(date_regex2.match(token)): is_date = True
        elif bool(date_regex3.match(token)): is_date = True
        elif bool(date_regex4.match(token)): is_date = True

        return is_date

    #
    def _build_vocab(self, max_vocab_cnt):
        # build vocab
        if self.tokenizer_type.startswith('word'):
            self._build_vocab_manual(max_vocab_cnt)
        elif self.tokenizer_type.startswith('bert-'):
            self.pad_id = self.tokenizer.sp_model.piece_to_id("<pad>")
            # self.vocab_count = 30522  # fixed for pretrained BERT vocab (old version)
            config_pretrained = BertConfig.from_pretrained(self.tokenizer_type) 
            self.vocab_count = config_pretrained.vocab_size

            map_vocab = {}
            for ind in range(self.vocab_count):
                map_vocab[ind] = self.tokenizer.sp_model.id_to_piece(ind)

            inv_map = {v: k for k, v in map_vocab.items()}


        elif self.tokenizer_type.startswith('xlnet-'):
            # self.vocab = self.tokenizer.vocab
            # self.rev_vocab = self.tokenizer.ids_to_tokens
            # self.pad_id = self.vocab["[PAD]"]
            self.pad_id = self.tokenizer.sp_model.piece_to_id("<pad>")
            # self.vocab_count = 32000  # fixed for pretrained BERT vocab
            config_pretrained = XLNetConfig.from_pretrained(self.tokenizer_type) 
            self.vocab_count = config_pretrained.vocab_size

            map_vocab = {}
            for ind in range(self.vocab_count):
                map_vocab[ind] = self.tokenizer.sp_model.id_to_piece(ind)

            inv_map = {v: k for k, v in map_vocab.items()}

            self.vocab = map_vocab
            self.rev_vocab = inv_map

        elif self.tokenizer_type.startswith('x5-'):
            self.pad_id = self.tokenizer.sp_model.piece_to_id("<pad>")
            # self.vocab_count = 32000  
            config_pretrained = T5Config.from_pretrained(self.tokenizer_type) 
            self.vocab_count = config_pretrained.vocab_size

            map_vocab = {}
            for ind in range(self.vocab_count):
                map_vocab[ind] = self.tokenizer.sp_model.id_to_piece(ind)

            inv_map = {v: k for k, v in map_vocab.items()}
            self.vocab = map_vocab
            self.rev_vocab = inv_map

        elif self.tokenizer_type.startswith('bart-'):
            self.pad_id = self.tokenizer.sp_model.piece_to_id("<pad>")
            # self.vocab_count = 32000  # fixed for pretrained BERT vocab
            config_pretrained = BartConfig.from_pretrained(self.tokenizer_type) 
            self.vocab_count = config_pretrained.vocab_size

            map_vocab = {}
            for ind in range(self.vocab_count):
                map_vocab[ind] = self.tokenizer.sp_model.id_to_piece(ind)

            inv_map = {v: k for k, v in map_vocab.items()}

        return

    #
    def _build_vocab_manual(self, max_vocab_cnt):
        """tokenize to word level for building vocabulary"""

        all_words = []
        for cur_doc in self.train_corpus:
            for cur_sent in cur_doc:
                tokenized_words = nltk.word_tokenize(cur_sent)
                all_words.extend(tokenized_words)

        vocab_count = Counter(all_words).most_common()
        vocab_count = vocab_count[0:max_vocab_cnt]

        # # create vocabulary list sorted by count for printing
        # raw_vocab_size = len(vocab_count)  # for printing
        # discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])  # for printing
        # print("Load corpus with train size %d, valid size %d, "
        #       "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
        #       % (len(self.train_corpus), len(self.valid_corpus),
        #          len(self.test_corpus),
        #          raw_vocab_size, len(vocab_count), vocab_count[-1][1],
        #          float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, BOS, EOS, TIME, DATE] + [t for t, cnt in
                                                         vocab_count]  # insert BOS and EOS to sentence later actually
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}

        self.pad_id = self.rev_vocab[PAD]
        self.unk_id = self.rev_vocab[UNK]
        self.bos_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.time_id = self.rev_vocab[TIME]
        self.date_id = self.rev_vocab[DATE]

        self.num_special_vocab = len(self.vocab) - max_vocab_cnt
        self.vocab_count = len(self.vocab)

        return
    # end def _build_vocab

    #
    def _get_stat_corpus(self):
        """ get statistics required for seq2seq processing from stored corpus"""
        
        ## get the number of sents in given whole corpus, regardless of train or test
        list_num_sent_doc = [len(doc) for doc in self.train_corpus]
        list_num_sent_doc = list_num_sent_doc + [len(doc) for doc in self.test_corpus]
        if self.valid_corpus is not None:
            list_num_sent_doc = list_num_sent_doc + [len(doc) for doc in self.valid_corpus]

        self.avg_num_sents = statistics.mean(list_num_sent_doc)
        self.std_num_sents = statistics.stdev(list_num_sent_doc)
        self.max_num_sents = np.max(list_num_sent_doc)  # document length (in terms of sentences)
        self.total_num_sents = sum(list_num_sent_doc)

        # print("Num Sents")
        # print(str(self.max_num_sents) + "\t" + str(self.avg_num_sents) + "\t" + str(self.std_num_sents))
        # print()

        ## get length of sentences
        self.max_len_sent = 0
        if self.tokenizer_type.startswith("bert") or self.tokenizer_type.startswith("xlnet"):
            list_len_sent = [len(self.tokenizer.tokenize(sent)) for cur_doc in self.train_corpus for sent in cur_doc]
            list_len_sent = list_len_sent + [len(self.tokenizer.tokenize(sent)) for cur_doc in self.test_corpus for sent in cur_doc]
            if self.valid_corpus is not None:
                list_len_sent = list_len_sent + [len(self.tokenizer.tokenize(sent)) for cur_doc in self.valid_corpus for sent in cur_doc]

        else:
            list_len_sent = [len(nltk.word_tokenize(sent)) for cur_doc in self.train_corpus for sent in cur_doc]
            list_len_sent = list_len_sent + [len(nltk.word_tokenize(sent)) for cur_doc in self.test_corpus for sent in cur_doc]
            if self.valid_corpus is not None:
                list_len_sent = list_len_sent + [len(nltk.word_tokenize(sent)) for cur_doc in self.valid_corpus for sent in cur_doc]

        self.max_len_sent = np.max(list_len_sent)
        if not self.tokenizer_type.startswith("bert") and not self.tokenizer_type.startswith("xlnet"):
            self.max_len_sent = self.max_len_sent + 2  # because of special character BOS and EOS (or SEP)
        self.avg_len_sent = statistics.mean(list_len_sent)
        self.std_len_sent = statistics.stdev(list_len_sent)

        # print("Len Sent")
        # print(str(self.max_len_sent-2) + "\t" + str(self.avg_len_sent) + "\t" + str(self.std_len_sent))
        # print()

        ## get document length (in terms of words number)
        list_len_word_doc = self._get_list_len_word_doc(self.train_corpus)
        list_len_word_doc = list_len_word_doc + self._get_list_len_word_doc(self.test_corpus)
        if self.valid_corpus is not None:
            list_len_word_doc = list_len_word_doc + self._get_list_len_word_doc(self.valid_corpus)

        self.max_len_doc = np.max(list_len_word_doc)
        self.avg_len_doc = statistics.mean(list_len_word_doc)
        self.std_len_doc = statistics.stdev(list_len_word_doc)

        self.med_len_doc = statistics.median(list_len_word_doc)
        self.geo_len_doc = gmean(list_len_word_doc)
        self.har_len_doc = statistics.harmonic_mean(list_len_word_doc)

        # print("Len Doc")
        # print(str(self.max_len_doc) + "\t" + str(self.avg_len_doc) + "\t" + str(self.std_len_doc))
        # print()


        return

    #
    def _get_max_doc_len(self, target_corpus):
        """ get maximum document length for seq2seq """

        doc_len_list = []
        for cur_doc in target_corpus:
            if self.tokenizer_type.startswith("bert") or self.tokenizer_type.startswith("xlnet"):
                len_num_words = len(self.tokenizer.tokenize(' '.join(sent for sent in cur_doc)))
                doc_len_list.append(len_num_words + (len(cur_doc)))
            else:
                cur_text = ' '.join(sent for sent in cur_doc)
                len_num_words = len(nltk.word_tokenize(cur_text))
                doc_len_list.append(len_num_words + (len(cur_doc)*2) )  # should be considered that each sent has bos and eos

        return max(doc_len_list)

    #
    def _get_list_len_word_doc(self, target_corpus):
        """ get maximum document length for seq2seq """

        doc_len_list = []
        for cur_doc in target_corpus:
            if self.tokenizer_type.startswith("bert") or self.tokenizer_type.startswith("xlnet"):
                len_num_words = len(self.tokenizer.tokenize(' '.join(sent for sent in cur_doc)))
                doc_len_list.append(len_num_words + (len(cur_doc)))
                break
            else:
                cur_text = ' '.join(sent for sent in cur_doc)
                len_num_words = len(nltk.word_tokenize(cur_text))
                doc_len_list.append(len_num_words + (len(cur_doc)*2) )  # should be considered that each sent has bos and eos

        return doc_len_list

    #
    def _refine_text(self, input_text, ignore_uni=True, ignore_para=True):
        """ customized function for pre-processing raw text"""
        input_text = input_text.lower()
        out_text = input_text

        return out_text

    #
    def _make_stopwords(self):
        """ make stopwords list (not used now)"""

        # snowball stopwords
        snowball_stopwords = "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing would should could ought i'm you're he's she's it's we're they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't let's that's who's what's here's there's when's where's why's how's a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very"
        stopwords = snowball_stopwords.split()

        if not self.keep_pronoun:
            pronouns = ['i', 'me', 'we', 'us', 'you', 'she', 'her', 'him', 'he', 'it', 'they', 'them', 'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves']
            stopwords = list(set(stopwords) - set(pronouns))

        str_punct = [t for t in string.punctuation]
        stopwords = stopwords + str_punct
        stopwords = stopwords + [u'``',u"''",u"lt",u"gt", u"<NUM>"]

        stopwords.remove('.')

        self.stopwords = stopwords

        return
    # end _make_stopwords

    # use stanza tokenizer
    def _sent_split_corpus(self, arr_input_text):
        """ tokenize corpus given tokenizer by config file"""
        # arr_input_text = pd_input['essay'].values

        # num_over = 0
        # total_sent = 0

        sent_corpus = []  # tokenized to form of [doc, list of sentences]
        for cur_doc in arr_input_text:
            cur_doc = self._refine_text(cur_doc)  # cur_doc: single string
            
            # sent_list = [sent.string.strip() for sent in spacy_nlp(cur_doc).sents] # spacy style

            ## stanza version
            doc_stanza = tokenizer_stanza(cur_doc)

            sent_list = [sentence.text for sentence in doc_stanza.sentences]
           
            # ## normal version
            # sent_list = self.sent_tokenzier(cur_doc)  # following exactly same way with previous works
            
            sent_corpus.append(sent_list)

            
        return sent_corpus

    #
    def _to_id_corpus(self, data):
        """
        Get id-converted corpus
        :param data: corpus data
        :return: id-converted corpus
        """
        results = []
        max_len_doc = -1
        list_doc_len = []
        entropies = []
        kld = []
        for cur_doc in data:
            temp = []
            for raw_sent in cur_doc:
                id_sent = self._sent2id(raw_sent)  # convert to id

                temp.append(id_sent)
            results.append(temp)

            # save max doc len
            flat_doc = [item for sublist in temp for item in sublist]
            list_doc_len.append(len(flat_doc))

        #
        max_len_doc = np.max(list_doc_len)
        # avg_len_doc = math.ceil(statistics.mean(list_doc_len))

        return results, max_len_doc, list_doc_len

    #
    def _sent2id(self, sent):
        """ return id-converted sentence """

        # note that, it is not zero padded yet here
        id_sent = []
        if self.tokenizer_type.startswith("word"):
            tokens_sent = nltk.word_tokenize(sent)  # word level tokenizer

            id_sent = [self.rev_vocab.get(t, self.unk_id) for t in tokens_sent]
            id_sent = [self.bos_id] + id_sent + [self.eos_id]  # add BOS and EOS to make an id-converted sentence
        else:  # assume we use transfomers library
            id_sent = self.tokenizer.encode(sent)  # each sentence has <sep> and <cls>, but <cls> does not influnece according to the other works
            
        return id_sent
