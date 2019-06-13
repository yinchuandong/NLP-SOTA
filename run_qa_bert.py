# %%
import numpy as np

from nlpsota.questionanswering.bert import (
    BertQAEstimator
)


# %%
estimator = BertQAEstimator()


estimator.fit(train_file='./example_data/squad_simple/train-v1.1.json',
              eval_file='./example_data/squad_simple/dev-v1.1.json',
              epochs=1,
              batch_size=2,
              pretrained_model_path='.bert/bert-base-uncased')

# %%
estimator.save('./output/model')



# %%