# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Run BERT on SQuAD."""

# %%
from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
import pickle
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)


from util import (read_squad_examples, convert_examples_to_features)

logger = logging.getLogger(__name__)


def load_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default='./bert/bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default='./debug_squad', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--train_file", default='./squad/simple/train-v1.1.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='./squad/simple/dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=2,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=1,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")
    args, unknown = parser.parse_known_args()


    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('=======================================================================')
    print(args)
    print('=======================================================================')
    return args


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        # print('---------------------------------------------------')
        # print('sequence_output: ', sequence_output.size())
        # print('pool: ', pooled_output.size())
        # print('org logits: ', logits.size())
        # print('org start_logits: ', start_logits.size())
        # print('org end_logits: ', end_logits.size())
        # print('org start_positions: ', start_positions.size())
        # print('org end_positions: ', end_positions.size())
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            # if len(start_positions.size()) > 1:
            #     start_positions = start_positions.squeeze(-1)
            # if len(end_positions.size()) > 1:
            #     end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # ignored_index = start_logits.size(1)
            # start_positions.clamp_(0, ignored_index)
            # end_positions.clamp_(0, ignored_index)

            # print('start_logits: ', start_logits.size(), start_logits)
            # print('end_logits: ', end_logits.size(), end_logits)
            # print('start_positions: ', start_positions.size(), start_positions)
            # print('end_positions: ', end_positions, end_positions.size(), end_positions)
            # print('ignored_index:', ignored_index)
            # print('---------------------------------------------------')
            # loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # loss_fct = BCELoss()
            loss_fct = BCEWithLogitsLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


# %%

args = load_args()


tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case)

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = read_squad_examples(
        input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

# Prepare model
model = BertForQuestionAnswering.from_pretrained(args.bert_model,
                                                    cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))

model.to(args.device)

# Prepare optimizer
if args.do_train:
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)


# %% training
global_step = 0
if args.do_train:
    cached_train_features_file = args.train_file+'_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))
    train_features = None
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info(
                "  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
   logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps) 
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long)

    # all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    # all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

    all_start_positions = torch.zeros(
        [len(train_features), args.max_seq_length], dtype=torch.float)
    all_end_positions = torch.zeros(
        [len(train_features), args.max_seq_length], dtype=torch.float)
    for i, f in enumerate(train_features):
        all_start_positions[i][f.start_position] = 1
        all_end_positions[i][f.end_position] = 1

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
            if args.n_gpu == 1:
                # multi-gpu does scattering it-self
                batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch

            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # print(start_positions)
            # print(end_positions)
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            loss = model(input_ids, segment_ids, input_mask,
                            start_positions, end_positions)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
# %%
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForQuestionAnswering.from_pretrained(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
else:
    model = BertForQuestionAnswering.from_pretrained(args.bert_model)

model.to(args.device)

# %% eavaluating
if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    eval_examples = read_squad_examples(
        input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(
        all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(
                input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                            start_logits=start_logits,
                                            end_logits=end_logits))
    output_prediction_file = os.path.join(
        args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(
        args.output_dir, "null_odds.json")
    write_predictions(eval_examples, eval_features, all_results,
                        args.n_best_size, args.max_answer_length,
                        args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)
