ARG tag=19.01-py3
FROM nvcr.io/nvidia/tensorflow:$TAG

ENV MODELSTORE=/modelstore TENSORBOARD=/tensorboard KAGGLE_CONFIG_DIR=/tmp/.kaggle
VOLUME $MODELSTORE $TENSORBOARD $KAGGLE_CONFIG_DIR

RUN apt-get update && \
      apt-get install -y --no-install-recommends p7zip-full ffmpeg && \
      pip install kaggle && \
      rm -rf /var/lib/apt/lists/*
