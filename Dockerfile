FROM tensorflow/tensorflow:latest-gpu

ARG API_KEY
ARG DATA_DIR=/data/

ENV KAGGLE_CONFIG_DIR=/workspace/.kaggle/ DATA_DIR=$DATA_DIR

COPY $API_KEY $KAGGLE_CONFIG_DIR
COPY run.sh /workspace/

RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg vim && \
  pip install kaggle && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
