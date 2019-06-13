# %%
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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)


# %%
bert_model_dir = './bert/bert-base-uncased'

model = BertForQuestionAnswering.from_pretrained(bert_model_dir)

# %%
param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]


train_examples = 100
train_batch_size = 20
gradient_accumulation_steps = 1
num_train_epochs = 10
learning_rate = 0.00005
warmup_proportion = 0.1

num_train_optimization_steps = int(len(
    train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs


optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)
# %%

with open('./squad/full/dev-v1.1.json', 'r') as f:
    dev_data = json.load(f)

with open('./squad/simple/dev-v1.1.json', 'w') as f:
    dev_data['data'] = dev_data['data'][:1]
    dev_data['data'][0]['paragraphs'] = dev_data['data'][0]['paragraphs'][:10]
    json.dump(dev_data, f, indent=2)

len(dev_data['data'])
# %%
with open('./squad/full/train-v1.1.json', 'r') as f:
    train_data = json.load(f)

with open('./squad/simple/train-v1.1.json', 'w') as f:
    train_data['data'] = train_data['data'][:1]
    train_data['data'][0]['paragraphs'] = train_data['data'][0]['paragraphs'][:10]
    json.dump(train_data, f, indent=2)

len(train_data['data'])
# %%
tokens_count = list(map(lambda x: len(x['context'].split()), train_data['data'][0]['paragraphs']))
max(tokens_count)
min(tokens_count)
# %%

from torch import nn
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
input

# target = torch.empty(3, dtype=torch.long).random_(5)
# target

target = torch.LongTensor([0, 1, 3])
# target = target.view(-1, 1)
target

output = loss(input, target)
output.backward()
output
# %%


m = nn.Sigmoid()
loss = nn.BCELoss()
# loss = nn.BCELoss(reduction='none')
target = torch.zeros([2, 4], dtype=torch.float32)  # 64 classes, batch size = 10
target[0, :5] = 1

target = torch.tensor([0, 3], dtype=torch.long)

input = torch.full([2, 4], 0.999, requires_grad=True)
output = loss(m(input), target)
# output.backward()
output
# %%


t1 = torch.zeros(2, 4)
t1

idx = [[0, 2, 3], [1, 3]]


pos = [(i, j) for i, row in enumerate(idx) for j in row]

rows, cols =  zip(*pos)

t1[rows, cols] = 1

t1
# %%

t0 = torch.zeros(2, 4)
t1 = torch.ones(2, 4)

torch.cat([t0, t1],dim=1)

# %%
