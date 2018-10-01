FROM tensorflow/tensorflow:nightly-gpu-py3

ARG API_KEY
ENV KAGGLE_CONFIG_DIR=/workspace/.kaggle/
COPY $API_KEY $KAGGLE_CONFIG_DIR

RUN apt-get update && \
  apt-get install -y --no-install-recommends p7zip-full ffmpeg vim && \
  pip install kaggle pandas RISE jupyter_contrib_nbextensions jupyter_nbextensions_configurator && \
  jupyter contrib nbextension install --sys-prefix && \
  jupyter-nbextension install rise --py --sys-prefix && \
  jupyter nbextensions_configurator enable --sys-prefix && \
  mkdir /.local && chmod 777 /.local && \
  sed -i "s/^#c.NotebookApp.password = ''.*/c.NotebookApp.password = 'nvidia'/" /root/.jupyter/jupyter_notebook_config.py && \
  rm -rf /var/lib/apt/lists/*

RUN jupyter nbextension enable splitcell && mkdir /.jupyter && chmod 777 /.jupyter

ENV SHELL=/bin/bash

EXPOSE 8888
EXPOSE 6006

VOLUME /data/
ENV DATA_DIR=/data/

VOLUME /workspace/
WORKDIR /workspace/
