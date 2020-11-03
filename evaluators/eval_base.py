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

class Eval_Base(object):
    logger = logging.getLogger(__name__)

    #
    def __init__(self, config):
        super(Eval_Base, self).__init__()
        self.eval_type = config.eval_type  # accuracy, qwk,
        
        self.eval_history = []

        return

    #
    def eval_update(self, model_output, label_y, origin_label=None):
        raise NotImplementedError
    #
    def eval_measure(self):
        raise NotImplementedError
    #
    def eval_reset(self):
        raise NotImplementedError
