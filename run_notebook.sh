#!/bin/bash
docker run \
  --rm \
  -it \
  --runtime=nvidia \
  -v /datasets/agunny/speech-recognition2/:/data \
  -v $PWD:/workspace \
  --workdir /workspace \
  -p 8888:8888 \
  -p 6006:6006 \
  -u $(id -u):$(id -g) \
  $USER/tf-src \
  jupyter-notebook \
  --ip=0.0.0.0
