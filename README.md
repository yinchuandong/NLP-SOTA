# A General NLP Library (Under Construction)


### Install
```
git clone git@github.com:yinchuandong/NLP-SOTA.git
cd NLP-SOTA
pip install .
```


### Features

- [x] question answering
  - [x] BERT
    - [x] data preprocessing
    - [x] model build
    - [ ] estimator build (not yet completed)
- [x] text classification    
  - [x] BERT Siamese network
    - [ ] data preprocessing
    - [x] model build
    - [ ] estimator build



### Pre-trained BERT Setup

1. Download pretrained BERT model from [https://github.com/google-research/bert#pre-trained-models](https://github.com/google-research/bert#pre-trained-models).

    ```
    mkdir bert & cd bert
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    `unzip uncased_L-12_H-768_A-12.zip
    mv uncased_L-12_H-768_A-12 bert-base-uncased
    ]
    ```

2. Convert tensorflow model file to pytorch model file

    ```
     ./convert_tf_checkpoint_to_pytorch ./bert/bert-base-uncased
    ```


### Question Answering
The problem formulation is very similar to Stanford Question Answering Dataset (**SQUAD**) [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/). Note that there is also a big difference. In SQUAD dataset, a question either has only one possible answer or is non-answerable. However, our dataset can have multiple answers for a certain question, which makes our dataset more difficult. My solution will focus on this formulation using BERT.


The code below shows how to train a model
``` python

from nlpsota.questionanswering.bert.estimator import BertQAEstimator

# initialize an estimator
estimator = BertQAEstimator()

# train the model
estimator.fit(train_file='./squad/simple/train-v1.1.json',
              eval_file='./squad/simple/dev-v1.1.json',
              epochs=1,
              batch_size=2,
              pretrained_model_path='./bert/bert-base-uncased')

# save model weights and vocabulary
estimator.save('./output/trained-model')
```

``` python
from nlpsota.questionanswering.bert.estimator import BertQAEstimator

# initialize an estimator
estimator = BertQAEstimator()

# load weights and vocabulary from directory
estimator.restore('./output/trained-model')

# predict
estimator.predict(....)
```
