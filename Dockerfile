FROM tensorflow/tensorflow:latest-gpu

ARG API_KEY
ENV KAGGLE_CONFIG_DIR=/workspace/.kaggle/
COPY $API_KEY $KAGGLE_CONFIG_DIR

RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg vim && \
  pip install kaggle pandas && \
  rm -rf /var/lib/apt/lists/*

VOLUME /data/
ENV DATA_DIR=/data/

VOLUME /workspace/
WORKDIR /workspace/
