#!/bin/bash

get_data(){
  SUBSET="$1"
  if ! [[ -d "$DATA_DIR/$SUBSET" ]]; then
    if ! [[ -f "$DATA_DIR/$SUBSET.7z" ]]; then
      kaggle competitions download -c tensorflow-speech-recognition-challenge -p $DATA_DIR
    fi
    7z x -o $DATA_DIR $DATA_DIR/$SUBSET.7z
  fi

  if [[ -f "$DATA_DIR/$SUBSET.7z" ]]; then
    rm $DATA_DIR/$SUBSET.7z
  fi
}

get_data train
# commenting out because of size restrictions on internal cluster
# get_data test

# the test call might have accidentally redownoladed train.7z. Get rid of it just in case
if [[ -f "$DATA_DIR/train.7z" ]]; then
  rm $DATA_DIR/train.7z
fi

SUBSETS=( train valid ptest ) # test
for SUBSET in "${SUBSETS[@]}"; do
  if ! [[ -f "$DATA_DIR/$SUBSET.tfrecords" ]]; then
    python /workspace/preproc/preproc.py --dataset_path $DATA_DIR --batch_size 4 --log_every 50 --subset $SUBSET
  fi
done
