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
import torch
import numpy as np

from evaluators.eval_base import Eval_Base

import utils
from utils import INT, FLOAT, LONG

from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score

class Eval_Fsc(Eval_Base):
    """ evaluation class for F-score """

    logger = logging.getLogger(__name__)

    #
    def __init__(self, config):
        super().__init__(config)

        self.correct = 0.0
        self.total = 0.0

        self.use_gpu = config.use_gpu

        self.output_size = config.output_size

        self.gcdc_ranges = (1, 3)  # used if it needs to re-scale
        self.pred_list = []
        self.label_list = []
        self.tid_list = []

        self.map_suppl = {}  # storage for supplementary data

        return

    #
    def _convert_to_origin_scale(self, scores):
        """ need to revert to original scale which is scaled for loss function """
        min_rating, max_rating = self.gcdc_ranges
        scores = scores * (max_rating - min_rating) + min_rating

        return scores

    #
    def eval_update(self, model_output, label_y, tid, origin_label=None):
        """ update data in every step for evalaution """

        if self.output_size == 1:
            model_output[model_output >= 0.5] = 1.0
            model_output[model_output < 0.5] = 0.0

            predicted = model_output
        else:
            _, predicted = torch.max(model_output, 1)  # model_output: (batch_size, num_class_out)

        self.correct += (predicted == label_y).sum().item()
        self.total += predicted.size(0)

        # self.pred_list.append(model_output)
        # self.label_list.append(origin_label)

        list_predict = predicted.squeeze().tolist()
        list_label = label_y.squeeze().tolist()
        # list_label = origin_label.squeeze().tolist()
        list_tid = list(tid)
        if not isinstance(list_predict, list): list_predict = [list_predict]
        if not isinstance(list_label, list): list_label = [list_label]
        if not isinstance(list_tid, list): list_tid = [list_tid]

        self.pred_list = self.pred_list + list_predict
        self.label_list = self.label_list + list_label
        self.tid_list = self.tid_list + list_tid


        return

    #
    def eval_update_mse(self, model_output, label_y, origin_label=None):
        # get accuracy

        self.pred_list.append(model_output)
        self.label_list.append(origin_label)

        return

    #
    def eval_measure(self, is_test):
        """ calculate evaluation from stored data """

        ## manual version from original implementation in GCDC paper
        num_correct = 0
        num_total = 0
        tp = 0
        fp = 0
        fn = 0
        type = "f05"
        for index, pred_val in enumerate(self.pred_list):
            gold_val = self.label_list[index]
            if type == "accuracy":
                if pred_val == gold_val:
                    num_correct += 1
            elif type == "f05":
                if pred_val == gold_val:
                    if gold_val == 1:
                        tp += 1
                else:
                    if pred_val == 1:
                        fp += 1
                    else:
                        fn += 1
            num_total += 1
        if type == "f05":
            precision = 0
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            recall = 0
            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            f05 = 0
            if (precision + recall) > 0:
                f05 = (1.25 * precision * recall) / (1.25 * precision + recall)

        # get fscore
        # f05 = fbeta_score(self.label_list, self.pred_list, beta=0.5)
        # f = f1_score(self.label_list, self.pred_list, average='macro', beta=0.5)

        fscore = f05

        # for err analysis
        # cur_pred_list = torch.cat(self.pred_list, dim=0)
        # cur_label_list = torch.cat(self.label_list, dim=0)

        # print(self.pred_list)
        # print(self.label_list)

        # self.pred_list_np = np.array(self.origin_label_list).astype('int32')
        # self.origin_label_np = np.array(self.origin_label_list).astype('int32')
        self.pred_list_np = self.pred_list
        self.origin_label_np = self.label_list
        self.tid_np = self.tid_list

        # self.pred_list_np = None
        # if self.use_gpu:    
        #     self.pred_list_np = cur_pred_list.cpu().numpy()
        #     self.origin_label_np = cur_pred_list.cpu().numpy()
        # else:   
        #     self.pred_list_np = cur_label_list.cpu().numpy()
        #     self.origin_label_np = cur_label_list.numpy()
        

        # reset
        self.eval_reset()

        # store performance for test mode
        if is_test:
            self.eval_history.append(fscore)


        return fscore

    #
    def eval_reset(self):
        self.correct = 0.0
        self.total = 0.0

        self.pred_list = []
        self.label_list = []  # list of float
        self.tid_list = []

        return

    #
    def save_suppl(self, name, supp_data):
        # supp_data = supp_data.squeeze().tolist()  # torch -> list
        if name not in self.map_suppl:
            self.map_suppl[name] = supp_data
        else:
            stored = self.map_suppl[name]
            updated = stored + supp_data
            self.map_suppl[name] = updated

        return