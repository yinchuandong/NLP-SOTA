# A General NLP Library (Under Construction)


## Install
```
git clone git@github.com:yinchuandong/NLP-SOTA.git
cd NLP-SOTA
pip install .
```

Please refer to my another project for end-to-end ML solution, including analysis, model training, model deployment, Flask, Docker, etc. [https://github.com/yinchuandong/sentiment-analysis](https://github.com/yinchuandong/sentiment-analysis).


## Features

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



## Setup pre-trained BERT

1. Download pretrained BERT model from [https://github.com/google-research/bert#pre-trained-models](https://github.com/google-research/bert#pre-trained-models).

    ```
    mkdir .bert & cd .bert
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    `unzip uncased_L-12_H-768_A-12.zip
    mv uncased_L-12_H-768_A-12 bert-base-uncased
    ```

2. Convert tensorflow model file to pytorch model file

    ```
     ./convert_tf_checkpoint_to_pytorch .bert/bert-base-uncased
    ```


## Data Analysis
Dive into the given dataset and do some visualisation. Please visit [analysis.ipynb](https://github.com/yinchuandong/NLP-SOTA/blob/master/analysis.ipynb)
for more information.


## Question Answering

The problem formulation is very similar to Stanford Question Answering Dataset (**SQUAD**) [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/). Given a document and a question, try to extract the possible answers (several tokens or a whole sentence) from the document. Note that there is also a big difference. In SQUAD dataset, a question either has only one possible answer or is non-answerable. However, our dataset can have multiple answers for a certain question, which makes our dataset more difficult. My solution will focus on this formulation using BERT.

### Data format

The following data is the format of SQUAD dataset. We also comply with the SQUAD data format and preprocess the our dataset to match it.
``` json
{
  "data": [
    {
      "title": "Super_Bowl_50",
      "paragraphs": [
        {
          "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
          "qas": [
            {
              "answers": [
                {
                  "answer_start": 403,
                  "text": "Santa Clara, California"
                },
                {
                  "answer_start": 355,
                  "text": "Levi's Stadium"
                },
                {
                  "answer_start": 355,
                  "text": "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
                }
              ],
              "question": "Where did Super Bowl 50 take place?",
              "id": "56be4db0acb8001400a502ee"
            }
          ]
        }
      ]
    }
  ]
}
```


### Model Training
``` python

from nlpsota.questionanswering.bert import BertQAEstimator

# initialize an estimator
estimator = BertQAEstimator()

# train the model
estimator.fit(train_file='./example_data/squad_simple/train-v1.1.json',
              eval_file='./example_data/squad_simple/dev-v1.1.json',
              epochs=1,
              batch_size=2,
              pretrained_model_path='.bert/bert-base-uncased')

# save model weights and vocabulary
estimator.save('./output/trained-model')
```

### Restore a well-trained model
``` python
from nlpsota.questionanswering.bert import BertQAEstimator

# initialize an estimator
estimator = BertQAEstimator()

# load weights and vocabulary from directory
estimator.restore('./output/trained-model')

# predict
estimator.predict(....)
```


## Text Classification
This dataset can also be formulated as a text classification problem. Given a question and a sentence, output a probability that the sentence is the answer of the question. However, rather than directly using existing models, **we propose a new text classification model based BERT and Siamese network in this repository**. We feed a question and a sentence into two separate BERT models, and get two pooled outputs from BERT models, respectively. But the two BERT models are sharing weights. Then, we concatenate two outputs and feed into a sigmoid layer. The code below simply shows how the the model works. Please check my another repository [https://github.com/yinchuandong/sentiment-analysis](https://github.com/yinchuandong/sentiment-analysis) for more details.


```python

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
```
