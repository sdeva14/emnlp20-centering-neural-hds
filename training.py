# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import csv

import utils
from utils import INT, FLOAT, LONG

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import logging
logger = logging.getLogger()

# from apex import amp
#from parallel import DataParallelModel, DataParallelCriterion  # parallel huggingface

#
def get_loss_func(config, pad_id=None):
    loss_func = None
    if config.loss_type.lower() == 'crossentropyloss':
        print("Use CrossEntropyLoss")
        # loss_func = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss_func = nn.CrossEntropyLoss()   
    elif config.loss_type.lower() == 'nllloss':
        print("Use NLLLoss")
        loss_func = nn.NLLLoss(ignore_index=pad_id)
    elif config.loss_type.lower() == 'multilabelsoftmarginloss':
        print("MultiLabelSoftMarginLoss")
        loss_func = nn.MultiLabelSoftMarginLoss()
    elif config.loss_type.lower() == 'mseloss':
        print("MSELoss")
        loss_func = nn.MSELoss()
    elif config.loss_type.lower() == 'bceloss':
        print("BCELoss")
        loss_func = nn.BCELoss()

    return loss_func

# end get_loss_func

#
def validate(model, evaluator, dataset_test, config, loss_func, is_test=False):
    model.eval()
    losses = []

    sampler_test = SequentialSampler(dataset_test) if config.local_rank == -1 else DistributedSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, sampler=sampler_test, batch_size=config.batch_size)

    adj_list = []
    root_ds_list = []
    seg_map_list = []
    cp_ind_list = []
    num_sents_list = []
    cp_seq_list = []

    tid_list = []
    label_list = []
    for text_inputs, label_y, *remains in dataloader_test:
        mask_input = remains[0]
        len_seq = remains[1]
        len_sents = remains[2]
        tid = remains[3]
        cur_origin_score = remains[-1]  # it might not be origin score when it is not needed for the dataset, then will be ignored

        text_inputs = utils.cast_type(text_inputs, LONG, config.use_gpu)
        mask_input = utils.cast_type(mask_input, FLOAT, config.use_gpu)
        len_seq = utils.cast_type(len_seq, FLOAT, config.use_gpu)

        with torch.no_grad():
            model_outputs = model(text_inputs=text_inputs, mask_input=mask_input, len_seq=len_seq, len_sents=len_sents, tid=tid, mode="") 
            coh_score = model_outputs[0]

            if config.output_size == 1:
                coh_score = coh_score.view(text_inputs.shape[0])
            else:
                coh_score = coh_score.view(text_inputs.shape[0], -1)

            if config.output_size == 1:
                label_y = utils.cast_type(label_y, FLOAT, config.use_gpu)
            else:
                label_y = utils.cast_type(label_y, LONG, config.use_gpu)
            label_y = label_y.view(text_inputs.shape[0])

            if loss_func is not None:
                loss = loss_func(coh_score, label_y)

                losses.append(loss.item())

            evaluator.eval_update(coh_score, label_y, tid, cur_origin_score)

            # for the project of centering transformer
            if config.gen_logs and config.target_model.lower() == "cent_attn":
                batch_adj_list = model_outputs[1]
                batch_root_ds = model_outputs[2]
                batch_seg_map = model_outputs[3]
                batch_cp_ind = model_outputs[4]
                batch_num_sents = model_outputs[5]

                adj_list = adj_list + batch_adj_list
                root_ds_list = root_ds_list + batch_root_ds
                seg_map_list = seg_map_list + batch_seg_map
                cp_ind_list = cp_ind_list + batch_cp_ind
                num_sents_list = num_sents_list + batch_num_sents

                tid_list = tid_list + tid.flatten().tolist()
                label_list = label_list + label_y.flatten().tolist()

                cp_seq_list = update_cp_seq(len_seq, len_sents, config, model_outputs[4], text_inputs, cp_seq_list)

        # end with torch.no_grad()
    # end for batch_num

    eval_measure = evaluator.eval_measure(is_test)
    eval_best_val = None
    if is_test:
        eval_best_val = max(evaluator.eval_history)

    if loss_func is not None:
        valid_loss = sum(losses) / len(losses)
        if is_test:
            logger.info("Total valid loss {}".format(valid_loss))
    else:
        valid_loss = np.inf
    
    if is_test:
        logger.info("{} on Test {}".format(evaluator.eval_type, eval_measure))
        logger.info("Best {} on Test {}".format(evaluator.eval_type, eval_best_val))
    else:
        logger.info("{} on Valid {}".format(evaluator.eval_type, eval_measure))


    valid_itpt = [tid_list, label_list, adj_list, root_ds_list, seg_map_list, cp_seq_list]

    return valid_loss, eval_measure, eval_best_val, valid_itpt
# end validate

def update_cp_seq(len_seq, len_sents, config, cur_cp_ind, text_inputs, cp_seq_list):
    sent_mask = torch.sign(len_sents)  # (batch_size, len_sent)
    sent_mask = utils.cast_type(sent_mask, FLOAT, config.use_gpu)
    num_sents = sent_mask.sum(dim=1)  # (batch_size)
    for batch_ind, cur_doc_cp in enumerate(cur_cp_ind):
        cur_doc_len = int(len_seq[batch_ind])
        cur_sent_len = len_sents[batch_ind]
        cur_sent_num = int(num_sents[batch_ind])  # (max_sent_num)
        end_loc = 0
        prev_loc = 0

        cur_cp_seq = []
        cur_text_seq = text_inputs[batch_ind]
        for cur_sent_ind in range(cur_sent_num):
            end_loc += int(cur_sent_len[cur_sent_ind])
            cur_sent_seq = cur_text_seq[prev_loc:end_loc]  # get the current sentence seq from a doc

            cur_sent_cp = cur_doc_cp[cur_sent_ind]
            cur_cp_seq.append(int(cur_sent_seq[cur_sent_cp]))  # get seq id of Cp
            
            prev_loc = end_loc
        # end for
        cp_seq_list.append(cur_cp_seq)

    # end for

    return cp_seq_list

#
def train(model, optimizer, scheduler, dataset_train, dataset_valid, dataset_test, config, evaluator):  # valid_feed can be None

    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    best_eval_valid = 0.0
    final_eval_best = 0.0
    final_valid = 0.0  # valid performance when the model achieves the best eval on the test set

    sampler_train = RandomSampler(dataset_train) if config.local_rank == -1 else DistributedSampler(dataset_train)
    # sampler_train = SequentialSampler(dataset_train) if config.local_rank == -1 else DistributedSampler(dataset_train)  # for debugging
    dataloader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=config.batch_size)

    batch_cnt = 0
    ckpt_step = len(dataloader_train.dataset) // dataloader_train.batch_size
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    loss_func = None
    # if config.use_parallel and not config.use_apex:
    # if config.n_gpu > 1 and not config.use_apex:
    if config.n_gpu > 1 and config.local_rank == -1:
        loss_func = get_loss_func(config=config, pad_id=model.module.pad_id)
    else:
        loss_func = get_loss_func(config=config, pad_id=model.pad_id)

    if config.use_gpu:
        loss_func.cuda()

    is_valid_sens = True
    if config.corpus_target.lower() == "yelp13" or config.corpus_target.lower() == "nyt":
        is_valid_sens = False

    # epoch loop
    model.train()
    for cur_epoch in range(config.max_epoch):

        # loop until traverse all batches
        tid_list = []
        label_list = []
        adj_list = []
        root_ds_list = []
        seg_map_list = []
        cp_ind_list = []
        num_sents_list = []

        cp_seq_list = []

        for text_inputs, label_y, *remains in dataloader_train:
            mask_input = remains[0]
            len_seq = remains[1]
            len_sents = remains[2]
            tid = remains[3]

            text_inputs = utils.cast_type(text_inputs, LONG, config.use_gpu)
            mask_input = utils.cast_type(mask_input, FLOAT, config.use_gpu)
            len_seq = utils.cast_type(len_seq, FLOAT, config.use_gpu)

            # training for this batch
            optimizer.zero_grad()
            
            # coh_score = model(text_inputs=text_inputs, mask_input=mask_input, len_seq=len_seq, len_sents=len_sents, tid=tid, mode="") 
            model_outputs = model(text_inputs=text_inputs, mask_input=mask_input, len_seq=len_seq, len_sents=len_sents, tid=tid, mode="") 
            coh_score = model_outputs[0]
            
            if config.output_size == 1:
                coh_score = coh_score.view(text_inputs.shape[0])
            else:
                coh_score = coh_score.view(text_inputs.shape[0], -1)

#           # get loss
            if config.output_size == 1:
                label_y = utils.cast_type(label_y, FLOAT, config.use_gpu)
            else:
                label_y = utils.cast_type(label_y, LONG, config.use_gpu)
            label_y = label_y.view(text_inputs.shape[0])

            loss = loss_func(coh_score, label_y)
            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            loss.backward()
            # with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # update optimizer and scheduler
            optimizer.step()
            if scheduler is not None:
                # scheduler.step()
                scheduler.step(loss)
            batch_cnt = batch_cnt + 1

            # print train process
            if batch_cnt % config.print_step == 0:
                logger.info("{}/{}-({:.3f})".format(batch_cnt % config.ckpt_step, config.ckpt_step, loss))

            ## log handling
            if config.gen_logs and config.target_model.lower() == "cent_attn":
                tid_list = tid_list + tid.flatten().tolist()
                label_list = label_list + label_y.flatten().tolist()
                
                adj_list = adj_list + model_outputs[1]
                root_ds_list = root_ds_list + model_outputs[2]
                seg_map_list = seg_map_list + model_outputs[3]
                cp_ind_list = cp_ind_list + model_outputs[4]
                num_sents_list = num_sents_list + model_outputs[5]

                cp_seq_list = update_cp_seq(len_seq, len_sents, config, model_outputs[4], text_inputs, cp_seq_list)

            ##########
            ## validation
            if batch_cnt % ckpt_step == 0:  # manual epoch printing
                model.eval()
                logger.info("\n=== Evaluating Model ===")

                # validation
                eval_cur_valid = -1
                if dataset_valid is not None and is_valid_sens:
                    loss_valid, eval_cur_valid, _, valid_itpt = validate(model, evaluator, dataset_valid, config, loss_func, is_test=False)
                    logger.info("")

                if eval_cur_valid >= best_eval_valid or dataset_valid is None:
                    logger.info("Best {} on Valid {}".format(evaluator.eval_type, eval_cur_valid))
                    best_eval_valid = eval_cur_valid

                    valid_loss, eval_last, eval_best, valid_itpt = validate(model, evaluator, dataset_test, config, loss_func, is_test=True)

                    if config.target_model.lower() == "cent_attn":
                        tid_list = tid_list + valid_itpt[0]
                        label_list = label_list + valid_itpt[1]
                        adj_list = adj_list + valid_itpt[2]
                        root_ds_list = root_ds_list + valid_itpt[3]
                        seg_map_list = seg_map_list + valid_itpt[4]
                        

                        cp_seq_list = cp_seq_list + valid_itpt[5]

                    if eval_best > final_eval_best: 
                        final_eval_best = eval_best

                        final_valid = eval_cur_valid

                        # save model
                        if config.save_model:
                            logger.info("Model Saved.")
                            torch.save(model.state_dict(), os.path.join(config.session_dir, "model"))

                        # save prediction log for error analysis

                        if config.gen_logs:
                            # log for prediction label
                            pred_log_name = "log_pred_" + str(config.essay_prompt_id_train) + "_" + str(config.essay_prompt_id_test) + "_" + str(config.cur_fold) + ".log"
                            if config.eval_type.lower() == "qwk":
                              pred_out = np.stack((evaluator.rescaled_pred, evaluator.origin_label_np, evaluator.tid_np))
                              np.savetxt(os.path.join(config.session_dir, pred_log_name), pred_out, fmt ='%.0f')
                            elif config.eval_type.lower() == "accuracy":
                              pred_out = np.stack((evaluator.pred_list_np, evaluator.origin_label_np, evaluator.tid_np))
                              pred_out = pred_out.T
                              np.savetxt(os.path.join(config.session_dir, pred_log_name), pred_out, fmt ='%.0f')

                            # log for structure
                            if config.target_model.lower() == "cent_attn":
                                stru_log_name = "log_stru_" + config.corpus_target.lower() + "_" + str(config.essay_prompt_id_train) + "_" + str(config.cur_fold) + ".log"
                                col = ["tid", "label", "adj", "root", "seg_map"]
                                tid_list = tid_list + valid_itpt[0]
                                label_list = label_list + valid_itpt[1]
                                adj_list = adj_list + valid_itpt[2]
                                root_ds_list = root_ds_list + valid_itpt[3]
                                seg_map_list = seg_map_list + valid_itpt[4]

                                cp_seq_list = cp_seq_list + valid_itpt[5]

                                # print(len(adj_list))
                                with open(os.path.join(config.session_dir, stru_log_name), "w") as log_file:
                                    writer = csv.DictWriter(log_file, fieldnames=col)
                                    for i in range(len(adj_list)):
                                        writer.writerow({'tid': tid_list[i], 'label': label_list[i], 'adj': adj_list[i], 'root': root_ds_list[i], 'seg_map':seg_map_list[i]})
                                    
                evaluator.map_suppl={}  # reset

                # exit eval model
                model.train()
                logger.info("\n**** Epoch {}/{} ****".format(cur_epoch, config.max_epoch))
            # end valdation

            if config.use_gpu and config.empty_cache:
                torch.cuda.empty_cache()    # due to memory shortage
        # end batch loop

        ## convert Cp indexes in sentences to sequence number
        if config.gen_logs and config.target_model.lower() == "cent_attn":
            cp_log_name = "log_cp_" + config.corpus_target.lower() + "_" + str(config.essay_prompt_id_train) + ".log"
            col = ["tid", "doc_cp_seq"]
            with open(os.path.join(config.session_dir, cp_log_name), "w") as log_file:
                writer = csv.DictWriter(log_file, fieldnames=col)
                for i in range(len(cp_seq_list)):
                    writer.writerow({'tid': tid_list[i], 'doc_cp_seq': cp_seq_list[i]})

        # end if

    # end epoch loop
    logger.info("Best {} on Test {}".format(evaluator.eval_type, final_eval_best))
    logger.info("")

    return final_eval_best, final_valid
# end train




