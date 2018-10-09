#!/bin/bash
SHORT_OPTS="d:k:h"
LONG_OPTS="data_dir:,kaggle:,help"
OPTS=`getopt -o $SHORT_OPTS --long $LONG_OPTS -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

CMD="jupyter notebook --allow-root --ip=0.0.0.0"
KAGGLE_CONFIG_DIR=$PWD/.kaggle
while true; do
  case "$1" in
    -d | --data_dir   ) DATA_DIR="$2"; shift; shift ;;
    -k | --kaggle     ) KAGGLE_CONFIG_DIR="$2"; shift; shift ;;
    -h | --help       ) HELP=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if ! [ -z "$HELP" ]; then
  echo "Build dataset and launch slideshow for training a model on Kaggle TensorFlow Speech Recognition Challenge"
  echo "Parameters"
  echo "----------"
  echo "    -d, --data_dir  : path to data (or where to save data if running preprocessing)"
  echo "    -k, --kaggle    : path to directory containing kaggle API key JSON"
  echo "    -h, --help      : show this help"
  exit 0
fi

if [ -z "$DATA_DIR" ]; then
  echo "Must specify data directory!"
  exit 1
fi

docker run \
  --rm \
  -it \
  --runtime=nvidia \
  -v $DATA_DIR:/data \
  -v $KAGGLE_CONFIG_DIR:/workspace/.kaggle/ \
  -p 8888:8888 \
  -p 6006:6006 \
  $USER/tf-src:allin
