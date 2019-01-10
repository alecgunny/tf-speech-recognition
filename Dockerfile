ARG TAG=18.12-py3
FROM nvcr.io/nvidia/tensorflow:$TAG

RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg && \
  pip install kaggle pandas RISE jupyter_contrib_nbextensions && \
  rm -rf /var/lib/apt/lists/*

# Notebook specific installations
RUN jupyter contrib nbextension install --sys-prefix && \
  jupyter nbextension install rise --py --sys-prefix && \
  jupyter nbextension enable splitcell/splitcell --sys-prefix 

# RUN mkdir /work/ && cd /work/ && git clone https://github.com/alecgunny/tf-speech-recognition.git
COPY img/ preproc/ Slideshow.ipynb /workspace/

EXPOSE 8888
EXPOSE 6006

VOLUME /data/ /tmp/.kaggle/

ENV DATA_DIR=/data/ KAGGLE_CONFIG_DIR=/tmp/.kaggle/ SHELL=/bin/bash PASSWORD=nvidia

ENTRYPOINT ./preproc/preproc.sh && jupyter-notebook --allow-root --ip=0.0.0.0
