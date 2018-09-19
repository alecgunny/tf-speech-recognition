#!/bin/bash

if [[ -z "$(ls $DATA_DIR)" ]]; then
  kaggle competitions download -c tensorflow-speech-recognition-challenge -p $DATA_DIR
  cd $DATA_DIR
  7z train.7z
  7z test.7z
fi
