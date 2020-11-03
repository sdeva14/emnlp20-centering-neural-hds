# -*- coding: utf-8 -*-

##
import os
import argparse
import logging
import time

##
import numpy as np
import torch
import torch.nn as nn

##
import build_config
import utils

import corpus.corpus_toefl
import corpus.corpus_nyt

import w2vEmbReader


from models.optim_hugging import AdamW, WarmupLinearSchedule, WarmupCosineSchedule

import torch.nn.functional as F
import torch

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import training

from evaluators import eval_acc, eval_qwk, eval_fscore


from models.model_CoNLL17_Essay import Model_CoNLL17_Essay
from models.model_EMNLP18_Centt import Model_EMNLP18_Centt

from models.model_Latent_Doc_Stru import Model_Latent_Doc_Stru

from models.model_DIS_Avg import Model_DIS_Avg
from models.model_DIS_TT import Model_DIS_TT
from models.model_Cent_Hds import Coh_Model_Cent_Hds

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from corpus.dataset_toefl import Dataset_TOEFL
from corpus.dataset_nyt import Dataset_NYT

########################################################

# global parser for arguments
parser = argparse.ArgumentParser()
arg_lists = []

###########################################
###########################################

#
def get_w2v_emb(config, corpus_target):
    embReader = w2vEmbReader.W2VEmbReader(config=config, corpus_target=corpus_target)
    
    return embReader                                              

#
def get_corpus_target(config):
    corpus_target = None
    logger = logging.getLogger()

    if config.corpus_target.lower() == "toefl":
        logger.info("Corpus: TOEFL")
        corpus_target = corpus.corpus_toefl.CorpusTOEFL(config)
    elif config.corpus_target.lower() == "nyt":
        logger.info("Corpus: NYT")
        corpus_target = corpus.corpus_nyt.CorpusNYT(config)

    return corpus_target
# end get_corpus_target

#
def get_dataset(config, id_corpus, pad_id):
    dataloader_train = None
    dataloader_valid = None
    dataloader_test = None

    if config.corpus_target.lower() == "toefl":
        dataset_train = Dataset_TOEFL(id_corpus["train"], config, pad_id)
        dataset_valid = Dataset_TOEFL(id_corpus["valid"], config, pad_id)
        dataset_test = Dataset_TOEFL(id_corpus["test"], config, pad_id)
    elif config.corpus_target.lower() == "nyt":
        dataset_train = Dataset_NYT(id_corpus["train"], config, pad_id)
        dataset_valid = Dataset_NYT(id_corpus["valid"], config, pad_id)
        dataset_test = Dataset_NYT(id_corpus["test"], config, pad_id)


    return dataset_train, dataset_valid, dataset_test

#
def get_model_target(config, corpus_target, embReader):
    model = None
    logger = logging.getLogger()

    if config.target_model.lower().startswith("emnlp18"):
        logger.info("Model: EMNLP18")
        model = Model_EMNLP18_Centt(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower().startswith("conll17"):
        logger.info("Model: CoNLL17")
        model = Model_CoNLL17_Essay(config=config, corpus_target=corpus_target, embReader=embReader)

    elif config.target_model.lower() == "latent_doc_stru":
        logger.info("Model: Latent_Doc_Stru")
        model = Model_Latent_Doc_Stru(config=config, corpus_target=corpus_target, embReader=embReader)

    elif config.target_model.lower() == "dis_avg":
        logger.info("Model: DIS_Simple Avg")
        model = Model_DIS_Avg(config=config, corpus_target=corpus_target, embReader=embReader)
    elif config.target_model.lower() == "dis_tt":
        logger.info("Model: DIS_Tree_Transformer")
        model = Model_DIS_TT(config=config, corpus_target=corpus_target, embReader=embReader)

    elif config.target_model.lower() == "cent_hds":
        logger.info("Model: Centering Structure")
        model = Coh_Model_Cent_Hds(config=config, corpus_target=corpus_target, embReader=embReader)


    return model

#
def get_optimizer(config, model, len_trainset):
    # basic style
    model_opt = model.module if hasattr(model, 'module') else model  # take care of parallel
    optimizer = model_opt.get_optimizer(config)
    scheduler = None  # we do not use scheduler in this work
   

    return optimizer, scheduler

    
#
def exp_model(config):
    ## Pre-processing

    # read corpus then generate id-sequence vector
    corpus_target = get_corpus_target(config)  # get corpus class
    corpus_target.read_kfold(config)

    # get embedding class
    embReader = get_w2v_emb(config, corpus_target)

    # update config depending on environment
    config.max_num_sents = corpus_target.max_num_sents  # the maximum number of sentences in document (i.e., document length)
    config.max_len_sent = corpus_target.max_len_sent  # the maximum length of sentence (the number of words)

    config.total_num_sents = corpus_target.total_num_sents  # the total number of sentences in the dataset

    # convert to id-sequence for given k-fold
    cur_fold = config.cur_fold
    id_corpus, max_len_doc, avg_len_doc, max_num_para, max_num_sents = corpus_target.get_id_corpus(cur_fold)
    config.max_len_doc = max_len_doc
    config.avg_len_doc = avg_len_doc

    config.avg_num_sents = corpus_target.avg_num_sents

    ## Model
    # prepare batch form
    dataset_train, dataset_valid, dataset_test = get_dataset(config, id_corpus, embReader.pad_id)

    #### prepare model
    if torch.cuda.is_available():   config.use_gpu = True
    else: config.use_gpu = False
    model = get_model_target(config, corpus_target, embReader)  # get model class
    optimizer, scheduler = get_optimizer(config, model, len(dataset_train))


    # distributed
    device = "cuda"
    if config.local_rank == -1 or not config.use_gpu:  # when it is not distributed mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.n_gpu = torch.cuda.device_count()  # will be 1 or 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # if config.use_parallel:
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", config.local_rank)
        # torch.distributed.init_process_group(backend='nccl')
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        config.world_size = torch.distributed.get_world_size()
        # config.n_gpu = 1

    if config.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        optimizer = model.module.get_optimizer(config)

    if config.use_gpu:
        model.to(device)

    ##
    evaluator = None
    if config.eval_type.lower() == "fscore":
            evaluator = eval_fscore.Eval_Fsc(config)
    elif config.eval_type.lower() == "accuracy":
            evaluator = eval_acc.Eval_Acc(config)
    elif config.eval_type.lower() == "qwk":
            min_rating, max_rating = corpus_target.score_ranges[corpus_target.prompt_id_test]  # in case of MSELoss
            evaluator = eval_qwk.Eval_Qwk(config, min_rating, max_rating, corpus_target) 

    ##
    # training and evaluating
    final_eval_best, final_valid = training.train(model,
                        optimizer,
                        scheduler,
                        dataset_train=dataset_train,
                        dataset_valid=dataset_valid,
                        dataset_test=dataset_test,
                        config=config,
                        evaluator=evaluator)


    return final_eval_best, final_valid
# end exp_model

###################################################

if __name__=='__main__':
    ## prepare config
    build_config.process_config()
    config, _ = build_config.get_config()
    utils.prepare_dirs_loggers(config, os.path.basename(__file__))
    logger = logging.getLogger() 
    
    # # torch.manual_seed(args.seed)  # we do not set a manual seed in this work

    # automatically extract target corpus from dataset path
    if len(config.corpus_target) == 0:
        cur_corpus_name = os.path.basename(os.path.normpath(config.data_dir))
        config.corpus_target = cur_corpus_name

    # domain information for printing
    cur_domain_train = None
    cur_domain_test = None
    if config.corpus_target.lower() == "toefl":
        cur_domain_train = config.essay_prompt_id_train
        cur_domain_test = config.essay_prompt_id_test

    ## Run model
    list_cv_valid=[]
    list_cv_attempts=[]
    target_attempts = config.cv_attempts
    
    if config.cur_fold > -1:  # test for specific fold
        if cur_domain_train is not None:
            logger.info("Source domain: {}, Target domain: {}, Cur_fold {}".format(cur_domain_train, cur_domain_test, config.cur_fold))
        eval_best_fold, valid_fold = exp_model(config)
        logger.info("{}-fold eval {}".format(config.cur_fold, eval_best_fold))
    else:
        for cur_attempt in range(target_attempts):  # CV only works when whole k-fold eval mode

            ##
            logger.info("Whole k-fold eval mode")
            list_eval_fold = []
            list_valid_fold = []
            for cur_fold in range(config.num_fold):
                config.cur_fold = cur_fold
                if cur_domain_train is not None:
                    logger.info("Source domain: {}, Target domain: {}, Cur_fold {}".format(cur_domain_train, cur_domain_test, config.cur_fold))
                cur_eval_best_fold, cur_valid_fold = exp_model(config)
                list_eval_fold.append(cur_eval_best_fold)
                list_valid_fold.append(cur_valid_fold)

            avg_cv_valid = sum(list_valid_fold) / float(len(list_valid_fold))
            logger.info("Final k-fold valid {}".format(avg_cv_valid))
            logger.info(list_valid_fold)
            list_cv_valid.append(avg_cv_valid)
            
            avg_cv_eval = sum(list_eval_fold) / float(len(list_eval_fold))
            logger.info("Final k-fold eval {}".format(avg_cv_eval))
            logger.info(list_eval_fold)
            list_cv_attempts.append(avg_cv_eval)

    #
    if target_attempts > 1 and len(list_cv_attempts) > 0:
        avg_cv_valid = sum(list_cv_valid) / float(len(list_cv_valid))
        logger.info("Final Valid CV exp result {}".format(avg_cv_valid))
        logger.info(list_cv_valid)

        logger.info("")

        avg_cv_attempt = sum(list_cv_attempts) / float(len(list_cv_attempts))
        logger.info("Final Test CV exp result {}".format(avg_cv_attempt))
        logger.info(list_cv_attempts)

        for cur_score in list_cv_valid:
            print(cur_score)

        print("\n")

        for cur_score in list_cv_attempts:
            print(cur_score)


