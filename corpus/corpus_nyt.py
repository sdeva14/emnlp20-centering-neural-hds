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

import logging
import os, io

from collections import Counter
import pandas as pd
import numpy as np
import sklearn.model_selection
import nltk
from unidecode import unidecode
import statistics

import corpus.corpus_base
from corpus.corpus_base import CorpusBase
from corpus.corpus_base import PAD, UNK, BOS, EOS, BOD, EOD, SEP, TIME, DATE

logger = logging.getLogger()

class CorpusNYT(CorpusBase):

    #
    def __init__(self, config):
        super().__init__(config)

        # self.score_ranges = {
        #     1: (2, 12), 2: (1, 6), 3: (0, 3), 4: (0, 3), 5: (0, 4), 6: (0, 4), 7: (0, 30), 8: (0, 60)
        # }
        if config.output_size < 0:
            config.output_size = 2  # 0, 1

        self.train_total_pd = None
        self.valid_total_pd = None
        self.test_total_pd = None

        self.train_pd = None
        self.valid_pd = None
        self.test_pd = None

        self.k_fold = config.num_fold

        # self.train_corpus = None  # just for remind, already declared in the corpus_base
        # self.test_corpus = None

        if config.is_gen_cv:
            seed = np.random.seed()
            self.generate_kfold(config=config, seed=seed)  # generate k-fold file

        self._read_dataset(config)  # store whole dataset as pd.dataframe

        self._build_vocab(config.max_vocab_cnt)

        return 
    # end __init__

    #
    def _get_model_friendly_scores(self, text_pd, prompt_id_target):
        """ scale between 0 to 1 for MSE loss function"""

        # scores_array = essay_pd['domain1_score'].values
        scores_array = text_pd['score'].values
        min_rating = 0
        max_rating = 4
        scaled_label = (scores_array - min_rating) / (max_rating - min_rating)

        text_pd.insert(len(text_pd.columns), 'rescaled_label', scaled_label)

        return text_pd

    #
    def _read_dataset(self, config):
        """ read asap dataset, assumed that splitted to "train.tsv", "dev.tsv", "test.tsv" under "fold_" """

        path_data = config.data_dir
        path_fold_dir = config.data_dir_cv
        # cur_path_data = os.path.join(path_data, "fold_" + str(config.cur_fold))

        # # read
        cur_path_data = os.path.join(path_data, path_fold_dir)
        # self.train_pd = pd.read_csv(os.path.join(cur_path_data, "nyt_train_fold_"+str(config.cur_fold)+".csv"), sep=",", header=0, encoding="utf-8", engine='c')  # skip the first line
        # self.valid_pd = pd.read_csv(os.path.join(cur_path_data, "nyt_valid_fold_"+str(config.cur_fold)+".csv"), sep=",", header=0, encoding="utf-8", engine='c')
        # self.test_pd = pd.read_csv(os.path.join(cur_path_data, "nyt_test_fold_"+str(config.cur_fold)+".csv"), sep=",", header=0, encoding="utf-8", engine='c')

        self.train_pd = pd.read_csv(os.path.join(cur_path_data, "elisa_nyt_train.csv"), sep=",", header=0, encoding="utf-8", engine='python')  # skip the first line
        self.valid_pd = pd.read_csv(os.path.join(cur_path_data, "elisa_nyt_valid.csv"), sep=",", header=0, encoding="utf-8", engine='python')
        self.test_pd = pd.read_csv(os.path.join(cur_path_data, "elisa_nyt_test.csv"), sep=",", header=0, encoding="utf-8", engine='python')

        # train_pd = self.train_total_pd.loc[self.train_total_pd['essay_set'] == self.prompt_id_train]
        # valid_pd = self.valid_total_pd.loc[self.valid_total_pd['essay_set'] == self.prompt_id_test]
        # test_pd = self.test_total_pd.loc[self.test_total_pd['essay_set'] == self.prompt_id_test]

        # # scaling score for training due to different score range as prompt_id
        # self.train_pd = self._get_model_friendly_scores(train_pd, self.prompt_id_train)
        # self.valid_pd = self._get_model_friendly_scores(valid_pd, self.prompt_id_test)
        # self.test_pd = self._get_model_friendly_scores(test_pd, self.prompt_id_test)

        self.train_pd["score"] = self.train_pd["score"].apply(lambda x: x - 1)  # scale to 0 to 1
        self.valid_pd["score"] = self.valid_pd["score"].apply(lambda x: x - 1)  # scale to 0 to 1
        self.test_pd["score"] = self.test_pd["score"].apply(lambda x: x - 1)  # scale to 0 to 1

        # self.merged_pd = pd.concat([self.train_pd, self.valid_pd, self.test_pd])
        # self.merged_pd = self.merged_pd.rename({'domain1_score': 'essay_score', 'essay_set': 'prompt'}, axis='columns')

        # put bias as mean of labels
        # self.output_bias = self.train_pd['rescaled_label'].values.mean(axis=0)

        # sentence split
        self.train_corpus, self.num_sents_train = self._sent_split_corpus(self.train_pd['text'].values)  # (doc_id, list of sents)
        self.valid_corpus, self.num_sents_valid = self._sent_split_corpus(self.valid_pd['text'].values)
        self.test_corpus, self.num_sents_test = self._sent_split_corpus(self.test_pd['text'].values)

        # get statistics for training later
        self._get_stat_corpus()

        return
    # end _read_dataset

    # nyt dataset has a clear splitter for sentences
    def _sent_split_corpus(self, arr_input_text):
        """ tokenize corpus given tokenizer by config file"""
        # arr_input_text = pd_input['essay'].values

        # num_over = 0
        # total_sent = 0

        import stanza  # stanford library for tokenizer
        tokenizer_stanza = stanza.Pipeline('en', processors='tokenize', use_gpu=True)

        num_sents = []
        sent_corpus = []  # tokenized to form of [doc, list of sentences]
        for cur_doc in arr_input_text:
            # cur_doc = self._refine_text(cur_doc)  # cur_doc: single string
            cur_doc = cur_doc.lower()
            
            # sent_list = [sent.string.strip() for sent in spacy_nlp(cur_doc).sents] # spacy style
           
            ## normal version
            # sent_list = self.sent_tokenzier(cur_doc)  # following exactly same way with previous works

            sent_list = cur_doc.split("<split2>")  # following exactly same way with previous works

            # stanza ver            
            # para_list = cur_doc.split("<split2>")

            # sent_list = []
            # for para in para_list:
            #     doc_stanza = tokenizer_stanza(para)
            #     cur_sents = [sentence.text for sentence in doc_stanza.sentences]

            #     refined_sents = [item for item in cur_sents if len(item)>1]

            #     sent_list = sent_list + refined_sents

            sent_corpus.append(sent_list)     

            # print(sent_list)
            # print(len(sent_list))
            # print(ewlkjewlef)

            num_sents.append(len(sent_list))

        # return sent_corpus
        return sent_corpus, num_sents

    #
    def get_id_corpus(self, num_fold=-1):
        """
        return id-converted corpus which is read in the earlier stage
        :param num_fold:
        :return: map of id-converted sentence
        """

        train_corpus = None
        valid_corpus = None
        test_corpus = None
        y_train = None
        y_valid = None
        y_test = None

        # ASAP is already divided
        train_corpus = self.train_corpus
        valid_corpus = self.valid_corpus
        test_corpus = self.test_corpus

        # change each sentence to id-parsed
        x_id_train, max_len_doc_train, list_len_train = self._to_id_corpus(train_corpus)
        x_id_valid, max_len_doc_valid, list_len_valid  = self._to_id_corpus(valid_corpus)
        x_id_test, max_len_doc_test, list_len_test = self._to_id_corpus(test_corpus)

        max_len_doc = max(max_len_doc_train, max_len_doc_valid, max_len_doc_test)
        # avg_len_doc = avg_len_doc_train

        # avg_len_doc = avg_len_doc_train*len(self.train_pd) + avg_len_doc_valid*len(self.valid_pd) + avg_len_doc_test*len(self.test_pd)
        # avg_len_doc = avg_len_doc / (len(self.train_pd) + len(self.valid_pd) + len(self.test_pd))

        list_len = list_len_train + list_len_valid + list_len_test
        avg_len_doc = statistics.mean(list_len)
        std_len_doc = statistics.stdev(list_len)

        # print(max_len_doc)
        # print(avg_len_doc)
        # print(std_len_doc)
        # print(welkjfwlewf)


        # paragraph info
        max_num_para = -1
        max_num_sents = -1
        # if self.use_paragraph:
        # list_num_para_train, len_para_train, list_num_sents_train = self._get_para_info(self.train_pd['text'])
        # list_num_para_valid, len_para_valid, list_num_sents_valid = self._get_para_info(self.valid_pd['text'])
        # list_num_para_test, len_para_test, list_num_sents_test = self._get_para_info(self.test_pd['text'])
        # max_num_para = max(max(list_num_para_train), max(list_num_para_valid), max(list_num_para_test))
        # max_num_sents = max(max(list_num_sents_train), max(list_num_sents_valid), max(list_num_sents_test))        

        max_num_para = 0
        len_para_train = []
        len_para_valid = []
        len_para_test = []

        max_num_sents = max(max(self.num_sents_train), max(self.num_sents_valid), max(self.num_sents_test))     

        # # debugging: max doc len in test
        # len_docs = []
        # for cur_doc in x_id_test:
        #     flat_doc = [item for sublist in cur_doc for item in sublist]
        #     len_docs.append(len(flat_doc))
        # max_len = max(len_docs)

        y_train = self.train_pd['score'].values
        y_valid = self.valid_pd['score'].values
        y_test = self.test_pd['score'].values

        # score_train = self.train_pd['domain1_score'].values
        # score_valid = self.valid_pd['domain1_score'].values
        # score_test = self.test_pd['domain1_score'].values

        score_train = self.train_pd['score'].values
        score_valid = self.valid_pd['score'].values
        score_test = self.test_pd['score'].values

        tid_train = self.train_pd['tid'].values
        tid_valid = self.valid_pd['tid'].values
        tid_test = self.test_pd['tid'].values

        # train_data_id = {'x_data': x_id_train, 'y_label': y_train, 'tid': tid_train, 'origin_score': score_train}  # origin score is needed for qwk
        # valid_data_id = {'x_data': x_id_valid, 'y_label': y_valid, 'tid': tid_valid, 'origin_score': score_valid}
        # test_data_id = {'x_data': x_id_test, 'y_label': y_test, 'tid': tid_test, 'origin_score': score_test}

        train_data_id = {'x_data': x_id_train, 'y_label': y_train, 'tid': tid_train, 'len_para': len_para_train, 'origin_score': score_train}  # origin score is needed for qwk
        valid_data_id = {'x_data': x_id_valid, 'y_label': y_valid, 'tid': tid_valid, 'len_para': len_para_valid, 'origin_score': score_valid}
        test_data_id = {'x_data': x_id_test, 'y_label': y_test, 'tid': tid_test,'len_para': len_para_test, 'origin_score': score_test}

        id_corpus = {'train': train_data_id, 'valid':valid_data_id, 'test': test_data_id}

        return id_corpus, max_len_doc, avg_len_doc, max_num_para, max_num_sents
    # end def get_id_corpus

    #
    def read_kfold(self, config):

        """ not used for asap, to follow the same setting with previous work"""

        return

       #
        #
    def generate_kfold(self, config, seed):
        """ Generate k-fold CV"""
        num_fold = config.num_fold

        # prepare target directory and file
        path_data = config.data_dir
        path_fold_dir = config.data_dir_cv
        if not os.path.exists(os.path.join(path_data, path_fold_dir)):
            os.makedirs(os.path.join(path_data, path_fold_dir))

        self.generate_single_split(config, seed)

        # file_name = "nyt_train.csv"
        # pd_train = pd.read_csv(os.path.join(path_data, file_name))
        # file_name = "nyt_valid.csv"
        # pd_valid = pd.read_csv(os.path.join(path_data, file_name))
        # file_name = "nyt_test.csv"
        # pd_test = pd.read_csv(os.path.join(path_data, file_name))

        file_name = "raw_good.csv"
        pd_good = pd.read_csv(os.path.join(path_data, file_name))
        file_name = "raw_typical.csv"
        pd_typical = pd.read_csv(os.path.join(path_data, file_name))
                
        # convert to numpy array
        arr_good = pd_good.values
        arr_typical = pd_typical.values
        arr_input_combined = np.vstack([arr_good, arr_typical])
        col_data = list(pd_good)
        # print(col_data)

        ## splitting by KFold (seperated version)
        # generate chunk
        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        rand_index = np.array(range(len(arr_good)))
        np.random.shuffle(rand_index)
        shuffled_good = arr_good[rand_index]
        list_chunk_good = np.array_split(shuffled_good, num_fold)

        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        rand_index = np.array(range(len(arr_typical)))
        shuffled_typical = arr_typical[rand_index]
        shuffled_typical = shuffled_typical[:len(shuffled_good)]  # extract the same amount of good label
        list_chunk_typical = np.array_split(shuffled_typical, num_fold)

        # concat chunks to num_fold
        # cur_map_fold = dict()
        print(len(arr_input_combined))
        for cur_fold in range(num_fold):
            train_chunks = []
            valid_chunks = []
            test_chunks = []

            for cur_ind in range(num_fold):
                cur_test_ind = (num_fold - 1) - cur_fold
                cur_valid_ind = cur_test_ind - 1
                if cur_valid_ind < 0:
                    cur_valid_ind = cur_valid_ind + config.num_fold

                if cur_ind == cur_test_ind:
                    test_chunks.append(list_chunk_good[cur_ind])
                    test_chunks.append(list_chunk_typical[cur_ind])
                    # continue
                elif cur_ind == cur_valid_ind:
                    valid_chunks.append(list_chunk_good[cur_ind])
                    valid_chunks.append(list_chunk_typical[cur_ind])
                else:
                    train_chunks.append(list_chunk_good[cur_ind])
                    train_chunks.append(list_chunk_typical[cur_ind])
                    
            # end for cur_ind
            print(len(train_chunks))
            print(len(test_chunks))

            cur_train_np = np.concatenate(train_chunks, axis=0)
            cur_valid_np = np.concatenate(valid_chunks, axis=0)
            cur_test_np = np.concatenate(test_chunks, axis=0)
            # cur_map_fold[cur_fold] = {"train": cur_train_np, "test": cur_test_np}

            # save CV partition
            cur_train_file = "nyt_train" + "_fold_" + str(cur_fold) + ".csv"
            cur_valid_file = "nyt_valid" + "_fold_" + str(cur_fold) + ".csv"
            cur_test_file = "nyt_test" + "_fold_" + str(cur_fold) + ".csv"
            # print(pd.DataFrame(input_train, columns=col_gcdc).head())
            pd.DataFrame(cur_train_np, columns=col_data).to_csv(os.path.join(path_data, path_fold_dir, cur_train_file), index=None)
            pd.DataFrame(cur_valid_np, columns=col_data).to_csv(os.path.join(path_data, path_fold_dir, cur_valid_file), index=None)
            pd.DataFrame(cur_test_np, columns=col_data).to_csv(os.path.join(path_data, path_fold_dir, cur_test_file), index=None)


        print(welkfjwlekjewlf)  # stop here

        return

    def generate_single_split(self, config, seed):
        """ Generate k-fold CV"""
        num_fold = config.num_fold

        # prepare target directory and file
        path_data = config.data_dir
        path_fold_dir = config.data_dir_cv
        if not os.path.exists(os.path.join(path_data, path_fold_dir)):
            os.makedirs(os.path.join(path_data, path_fold_dir))

        file_name = "stanza_raw_good.csv"
        pd_good = pd.read_csv(os.path.join(path_data, file_name))
        file_name = "stanza_raw_typical.csv"
        pd_typical = pd.read_csv(os.path.join(path_data, file_name))
                
        # convert to numpy array
        arr_good = pd_good.values
        arr_typical = pd_typical.values
        arr_input_combined = np.vstack([arr_good, arr_typical])
        col_data = list(pd_good)
        # print(col_data)

        ## splitting by KFold (seperated version)
        # generate chunk
        num_fold = 10
        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        rand_index = np.array(range(len(arr_good)))
        np.random.shuffle(rand_index)
        shuffled_good = arr_good[rand_index]
        list_chunk_good = np.array_split(shuffled_good, num_fold)

        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        rand_index = np.array(range(len(arr_typical)))
        shuffled_typical = arr_typical[rand_index]
        shuffled_typical = shuffled_typical[:len(shuffled_good)]  # extract the same amount of good label
        list_chunk_typical = np.array_split(shuffled_typical, num_fold)

        train_chunks = []
        valid_chunks = []
        test_chunks = []
        for cur_ind in range(8):
            train_chunks.append(list_chunk_typical[cur_ind])
            train_chunks.append(list_chunk_good[cur_ind])
            print(cur_ind)

        for cur_ind in range(8, 9):
            valid_chunks.append(list_chunk_typical[cur_ind])
            valid_chunks.append(list_chunk_good[cur_ind])
            print(cur_ind)

        for cur_ind in range(9, 10):
            test_chunks.append(list_chunk_typical[cur_ind])
            test_chunks.append(list_chunk_good[cur_ind])
            print(cur_ind)

        cur_train_np = np.concatenate(train_chunks, axis=0)
        cur_valid_np = np.concatenate(valid_chunks, axis=0)
        cur_test_np = np.concatenate(test_chunks, axis=0)

        # save CV partition
        cur_train_file = "nyt_train" + "_single_split.csv"
        cur_valid_file = "nyt_valid" + "_single_split.csv"
        cur_test_file = "nyt_test" + "_single_split.csv"
        # print(pd.DataFrame(input_train, columns=col_gcdc).head())
        pd.DataFrame(cur_train_np, columns=col_data).to_csv(os.path.join(path_data, path_fold_dir, cur_train_file), index=None)
        pd.DataFrame(cur_valid_np, columns=col_data).to_csv(os.path.join(path_data, path_fold_dir, cur_valid_file), index=None)
        pd.DataFrame(cur_test_np, columns=col_data).to_csv(os.path.join(path_data, path_fold_dir, cur_test_file), index=None)

        print(welkfjwlekjewlf)  # stop here

        return

# end class
