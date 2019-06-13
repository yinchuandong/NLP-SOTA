# A General NLP Library

### Pre-trained BERT Setup

1. Download pretrained BERT model from [https://github.com/google-research/bert#pre-trained-models](https://github.com/google-research/bert#pre-trained-models).
  - `mkdir bert & cd bert`
  - `wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip`
  - `unzip uncased_L-12_H-768_A-12.zip`
  - `mv uncased_L-12_H-768_A-12 bert-base-uncased`

2. Convert tensorflow model file to pytorch model file
  - `./convert_tf_checkpoint_to_pytorch ./bert/bert-base-uncased`

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


### Question Answering
