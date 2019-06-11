export SQUAD_DIR=./squad/full
export PYTORCH_PRETRAINED_BERT_CACHE=./bert

# --do_train \
python run_squad_v0.py \
  --bert_model $PYTORCH_PRETRAINED_BERT_CACHE/bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./debug_squad/