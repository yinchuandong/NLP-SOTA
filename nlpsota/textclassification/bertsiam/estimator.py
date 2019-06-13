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

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import (
    PYTORCH_PRETRAINED_BERT_CACHE,
    WEIGHTS_NAME,
    CONFIG_NAME
)
from pytorch_pretrained_bert.modeling import (
    BertModel,
    BertPreTrainedModel,
    BertConfig)
from pytorch_pretrained_bert.optimization import (
    BertAdam,
    WarmupLinearSchedule
)
from pytorch_pretrained_bert.tokenization import (
    BasicTokenizer,
    BertTokenizer,
    whitespace_tokenize
)


class BertSiamModel(BertPreTrainedModel):
    """
    A Siamese BERT text classification model:
        question -> BERT -> pooled_output0
                                           \
                                             concat --> 0/1
                                           /
        sentence -> BERT -> pooled_output1
    where the weights of BERT are shared between question and sentence
    """
    def __init__(self, config):
        super(BertQAModel, self).__init__(config)
        self.bert = BertModel(config)

        self.linear = nn.Linear(config.hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_bert_weights)
        return

    def forward(self,
                question_input_ids,
                sentence_input_ids,
                targets=None):
        _, question_pooled_output = self.bert(
            question_input_ids,
            output_all_encoded_layers=False)

        _, sentence_pooled_output = self.bert(
            sentence_input_ids,
            output_all_encoded_layers=False)

        pooled_output = torch.cat([
            question_pooled_output,
            sentence_pooled_output], dim=1)
        logits = self.linear(pooled_output)
        prob = self.sigmoid(logits)

        if targets is not None:
            loss_fn = BCELoss()
            loss = loss_fn(prob, targets)
            return loss
        else:
            return prob


class BertSiamEstimator(object):
    """
    Please refer to my another project:
        https://github.com/yinchuandong/sentiment-analysis
    """

    def __init__(self):

        return

    def fit(self, X, y, eval=(X_eval, y_eval)):

        return

    def evaluate(self, X_eval, y_eval):

        return

    def predict(self, X):

        return
