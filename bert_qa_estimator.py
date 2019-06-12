
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


class BertQAModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQAModel, self).__init__(config)
        self.bert = BertModel(config)

        self.sigmoid = nn.Sigmoid()
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        logits = self.sigmoid(logits)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            loss_fct = BCELoss()
            # loss_fct = BCEWithLogitsLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class BertQAEstimator(object):

    def __init__(self,
                 bert_base_model='bert-base-uncased',
                 max_seq_length=384,
                 doc_stride=128,
                 max_query_length=64,
                 max_answer_length=30,
                 do_lower_case=True,
                 with_negative=False,
                 learning_rate=5e-5,
                 warmup_proportion=0.1,
                 gradient_accumulation_steps=1,
                 seed=42):
        """
        Args:
            bert_base_model: (str), Bert pre-trained model path,
                selected in the list:
                    bert-base-uncased, bert-large-uncased, bert-base-cased,
                    bert-large-cased, bert-base-multilingual-uncased,
                    bert-base-multilingual-cased, bert-base-chinese.
            max_seq_length: (int), The maximum total input sequence length
                after WordPiece tokenization. Sequences longer than this will
                be truncated, and sequences shorter than this will be padded.
            doc_stride: (int), When splitting up a long document into chunks,
                how much stride to take between chunks.
            max_query_length: (int), The maximum number of tokens for the
                question. Questions longer than this will be truncated to
                this length.
            max_answer_length: (int), The maximum length of an answer that can
                be generated. This is needed because the start and end
                predictions are not conditioned on one another.
            do_lower_case: (bool), Whether to lower case the input text.
                True for uncased models, False for cased models.
            with_negative: (bool), whether examples contain some that do not
                have an answer.
            learning_rate: (float), The initial learning rate for Adam.
            warmup_proportion: (float), Proportion of training to perform
                linear learning rate warmup for. E.g., 0.1 = 10%% of training.
            gradient_accumulation_steps: (int), Number of updates steps to
                accumulate before performing a backward/update pass.
            seed: (bool) random seed for initialization
        """

        self.bert_base_model = bert_base_model
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.max_answer_length = max_answer_length
        self.do_lower_case = do_lower_case
        self.learning_rate = learning_rate
        self.with_negative = with_negative
        self.warmup_proportion = warmup_proportion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and not self.no_cuda else 'cpu')
        self.n_gpu = torch.cuda.device_count()

        logger.info('device: {} n_gpu: {}'.format(self.device, self.n_gpu))

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        self.model = None
        self.tokenizer = None
        return

    def _load_feature(self, train_file, train_examples):
        cached_train_features_file = train_file + '_{0}_{1}_{2}_{3}'.format(
            self.bert_base_model,
            str(self.max_seq_length),
            str(self.doc_stride),
            str(self.max_query_length))
        train_features = None
        try:
            with open(cached_train_features_file, 'rb') as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=True)
            logger.info('Saving train features into cached file %s',
                        cached_train_features_file)
            with open(cached_train_features_file, 'wb') as writer:
                pickle.dump(train_features, writer)
        return train_features

    def _init_optimizer(self, train_steps):
        param_optimizer = list(self.model.named_parameters())
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
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            warmup=self.warmup_proportion,
            t_total=train_steps)
        return optimizer

    def fit(self,
            train_file,
            dev_file,
            epochs,
            batch_size,
            pretrained_model_path):

        train_examples = read_squad_examples(
            input_file=train_file,
            is_training=True,
            version_2_with_negative=self.with_negative)
        train_steps = int(len(train_examples) / batch_size /
                          self.gradient_accumulation_steps) * epochs

        self.model = BertQAModel.from_pretrained(pretrained_model_path)
        self.model.to(self.device)

        optimizer = self._init_optimizer(train_steps)
        train_features = self._load_feature(train_file, train_examples)

        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.zeros(
            [len(train_features), self.max_seq_length], dtype=torch.float)
        all_end_positions = torch.zeros(
            [len(train_features), self.max_seq_length], dtype=torch.float)
        for i, f in enumerate(train_features):
            all_start_positions[i][f.start_position] = 1
            all_end_positions[i][f.end_position] = 1

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=batch_size)

        global_step = 0
        self.model.train()
        for _ in trange(int(epochs), desc="Epochs"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Batchs")):
                if self.n_gpu == 1:
                    # multi-gpu does scattering it-self
                    batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch

                # print(start_positions)
                # print(end_positions)
                # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                loss = self.model(input_ids, segment_ids, input_mask,
                                  start_positions, end_positions)
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
        return

    def save(self, output_dir):
        """ Save a trained model, configuration and tokenizer
        Args:
            output_dir: (str), directory to save model, config, and tokenizers
        """

        # Only save the model it-self
        model_to_save = model.module if hasattr(model, 'module') else model

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)
        return

    def restore(self, output_dir):
        """ Load a trained model and vocabulary that you have fine-tuned
        Args:
            output_dir: (str), the directory where you save models
        """

        self.model = BertQAModel.from_pretrained(output_dir)
        self.tokenizer = BertTokenizer.from_pretrained(
            output_dir, do_lower_case=self.do_lower_case)
        return

    def evaluate(self, dev):

        return

    def predict(self, data):
        return

# %%


def main():
    model = BertQAEstimator()
    model.fit(train_file='./squad/simple/train-v1.1.json',
              dev_file='./squad/simple/dev-v1.1.json',
              epochs=1,
              batch_size=1)

    return


main()

# %%
