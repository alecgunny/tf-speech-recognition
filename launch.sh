#!/bin/bash
SHORT_OPTS="d:ph"
LONG_OPTS="data_dir:,preproc,help"
OPTS=`getopt -o $SHORT_OPTS --long $LONG_OPTS -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

CMD="jupyter notebook --ip=0.0.0.0"
while true; do
  case "$1" in
    -d | --data_dir ) DATA_DIR="$2"; shift; shift ;;
    -p | --preproc  ) CMD="preproc/preproc.sh"; shift ;;
    -h | --help     ) HELP=true; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -z "$HELP" ]; then
  echo "Build dataset and launch slideshow for training a model on Kaggle TensorFlow Speech Recognition Challenge"
  echo "Parameters"
  echo "----------"
  echo "    -d, --data_dir  : path to data (or where to save data if running preprocessing)"
  echo "    -p, --preproc   : whether to run preprocessing data generation script. If set, notebook won't launch"
  echo "    -h, --help      : show this help"
  return 0
fi

if [ -z "$DATA_DIR" ]; then
  echo "Must specify data directory!"
  return 1
fi

docker run \
  --rm \
  -it \
  --runtime=nvidia \
  -v $DATA_DIR:/data \
  -v $PWD:/workspace \
  --workdir /workspace \
  -p 8888:8888 \
  -p 6006:6006 \
  -u $(id -u):$(id -g) \
  $USER/tf-src \
  $CMD
