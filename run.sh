#!/bin/bash
TF_IMAGE=$USER/tf-speech-recognition
DATASET_PATH=/datasets/agunny/speech-recognition

docker volume create modelstore
docker volume create tensorboard

docker run \
  --rm \
  -d \
  --runtime nvidia \
  -v tensorboard:/tensorboard \
  --name tensorboard \
  -p 6006:6006 \
  $TF_IMAGE \
  tensorboard \
  --logdir=/tensorboard \
  --host=0.0.0.0

docker run \
  --rm \
  -it \
  --runtime nvidia \
  -v $PWD/src:/workspace \
  -v modelstore:/modelstore \
  -v tensorboard:/tensorboard \
  -v $DATASET_PATH:/data \
  --name tf-speech-recognition \
  $TF_IMAGE \
  python main.py \
  --train_data /data/train.tfrecords \
  --valid_data /data/valid.tfrecords \
  --pixel_wise_stats /data/stats.tfrecords \
  --labels /data/labels.txt \
  --input_shape 99 161 \
  --learning_rate 1e-5 \
  --batch_size 256 \
  --num_epochs 1 \
  --num_gpus 4 \
  --model_store_dir /modelstore \
  --model_name my_tf_model \
  --model_version 0 \
  --count 4

docker run \
  --rm \
  -it \
  --runtime nvidia \
  -v $PWD/src:/workspace \
  -v modelstore:/modelstore \
  $TF_IMAGE \
  ./convert_to_trt.sh

docker run \
  --rm \
  -it \
  --runtime nvidia \
  -v modelstore:/modelstore \
  -p 8000-8002:8000-8002 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  --name trtserver \
  nvcr.io/nvidia/tensorrtserver:19.01-py3 \
  trtserver \
  --model-store=/modelstore \
  --strict-model-config=false

# docker kill tensorboard
# docker volume rm tensorboard modelstore
