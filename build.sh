#!/bin/bash
TRT_SERVER_DIR=github.com/nvidia/tensorrtserver#r19.01
docker build \
  -t $USER/tensorrtserver:client \
  --target trtserver_build \
  --build-arg BUILD_CLIENTS_ONLY=1 \
  --build-arg PYVER=3.5 \
  $TRT_SERVER_DIR
docker build \
  -t $USER/tf-speech-recognition \
  --build-arg trtisclient=$USER/tensorrtserver:client \
  --build-arg tag=19.01-py3 \
  .
