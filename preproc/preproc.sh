#!/bin/bash

if (! [[ -f "$DATA_DIR/train.7z" ]]) | (! [[ -f "$DATA_DIR/test.7z" ]]); then
  kaggle competitions download -c tensorflow-speech-recognition-challenge -p $DATA_DIR
fi

cd $DATA_DIR
if ! [[ -d "$DATA_DIR/train" ]]; then
  7z train.7z
fi
if ! [[ -d "$DATA_DIR/test" ]]; then
  7z test.7z
fi

python preproc.py --dataset_path $DATA_DIR --batch_size 4 --log_every 50
